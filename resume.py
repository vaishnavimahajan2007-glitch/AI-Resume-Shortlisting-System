from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Candidate Resumes
resumes = [
    {
        "name": "vaishnavi Mahajan",
        "resume": "Python Machine Learning Data Science SQL NLP Deep Learning"
    },
    {
        "name": "Khadija Patel",
        "resume": "Python NLP AI Machine Learning TensorFlow Flask"
    }
]

# Job Description
job_description = """
Looking for a candidate with Python, NLP, Machine Learning,
AI, TensorFlow, and Data Science skills.
"""

# Extract resume texts
resume_texts = [r["resume"] for r in resumes]

# Combine resumes with job description
all_documents = resume_texts + [job_description]

# Convert text into numerical vectors using TF-IDF
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(all_documents)

# Get job description vector
job_vector = tfidf_matrix[-1]

# Get resume vectors
resume_vectors = tfidf_matrix[:-1]

# Calculate cosine similarity
similarity_scores = cosine_similarity(resume_vectors, job_vector)

# Store ranking results
results = []

for i in range(len(resumes)):
    results.append({
        "name": resumes[i]["name"],
        "score": similarity_scores[i][0]
    })

# Sort candidates by highest score
results = sorted(results, key=lambda x: x["score"], reverse=True)

# Display Output
print("\n========== AI Resume Shortlisting System ==========\n")

for rank, candidate in enumerate(results, start=1):
    print(f"Rank {rank}")
    print(f"Candidate Name : {candidate['name']}")
    print(f"Matching Score : {round(candidate['score'] * 100, 2)}%")
    print("-" * 45)

print("\nResume Shortlisting Completed Successfully!")
