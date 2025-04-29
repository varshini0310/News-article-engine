from flask import Flask, render_template, request
import requests
from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__, template_folder="templates")

nltk.download("stopwords")
nltk.download("wordnet")

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess_query(query):
    query = query.lower()
    query = re.sub(r'[^\w\s]', '', query)
    words = query.split()
    filtered_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    processed_query = '+'.join(filtered_words)
    return processed_query

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

def scrape_news(query):
    processed_query = preprocess_query(query)
    search_url = f"https://www.bing.com/news/search?q={processed_query}"
    headers = {"User-Agent": "Mozilla/5.0"}

    response = requests.get(search_url, headers=headers)
    if response.status_code != 200:
        return [{"title": "Error fetching news", "link": "#", "description": ""}]

    soup = BeautifulSoup(response.text, "html.parser")
    articles = soup.select("div.news-card")

    results = []
    seen_links = set()  # To track already added articles by link

    for article in articles[:10]:
        title_tag = article.select_one("a.title")
        desc_tag = article.select_one("div.snippet")

        if title_tag and title_tag.get("href"):
            title = title_tag.text.strip()
            link = title_tag["href"]
            description = desc_tag.text.strip() if desc_tag else ""

            # Check if the article link is already in the results
            if link not in seen_links:
                results.append({"title": title, "link": link, "description": description})
                seen_links.add(link)  # Mark this article as seen

    return results

def rank_articles_by_tfidf(articles, query):
    texts = [clean_text(article["description"]) for article in articles]
    query_cleaned = clean_text(query)

    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform([query_cleaned] + texts)

    query_vector = tfidf_matrix[0]
    article_vectors = tfidf_matrix[1:]
    scores = cosine_similarity(query_vector, article_vectors).flatten()

    for i, score in enumerate(scores):
        articles[i]["score"] = score

    return sorted(articles, key=lambda x: x["score"], reverse=True)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/search", methods=["GET"])
def search():
    query = request.args.get("query", "").strip()
    if not query:
        return render_template("index.html", articles=[])

    articles = scrape_news(query)
    if articles and articles[0].get("title") != "Error fetching news":
        articles = rank_articles_by_tfidf(articles, query)

    return render_template("index.html", articles=articles)

if __name__ == "__main__":
    app.run(debug=True)
