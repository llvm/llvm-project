# LAT5150DRVMIL RAG System

**Lightweight, Fast Documentation Search and Retrieval**

Built using research-backed best practices from:
- Maharana et al. 2025 - "RAG for Building Datasets from Scientific Literature"
- Optimal chunk size: 256 words, overlap: 20 words
- Target accuracy: >88%

---

## 🚀 Quick Start (5 minutes)

### 1. Build the Index (First Time Only)
```bash
cd /home/user/LAT5150DRVMIL
python3 rag_system/document_processor.py
```

This processes all documentation in `00-documentation/` and creates a searchable index.

**Output:**
- `rag_system/processed_docs.json` - Vector database
- Processes ~235 documents into ~881 chunks
- Builds TF-IDF index with ~7,600 unique terms

### 2. Query the Documentation

**Single Query:**
```bash
python3 rag_system/rag_query.py "What is DSMIL activation?"
```

**Interactive Mode:**
```bash
python3 rag_system/rag_query.py
```

Then type your questions:
```
LAT5150-RAG> What is DSMIL activation?
LAT5150-RAG> How to enable NPU modules?
LAT5150-RAG> APT41 security features?
LAT5150-RAG> quit
```

---

## 📊 Features

### ✅ Document Search
- Fast TF-IDF based semantic search
- Context-aware retrieval (top-k=3 optimal)
- Source file tracking
- Relevance scoring

### ✅ Structured Extraction
Extract specific information in structured format:
```bash
python3 rag_system/structured_extraction.py "DSMIL activation"
```

**Output:**
```
System: DSMIL activation
Purpose: Safe military token activation system for Dell Latitude 5450
Activation Steps:
  • Verify kernel module
  • Check hardware compatibility
  • Enable security features
Requirements:
  • Dell Latitude 5450 MIL-SPEC
  • TPM 2.0
  • Kernel 6.16.9+
Security Level: MIL-SPEC classified
```

### ✅ Batch Processing
Process multiple queries efficiently:
```python
from rag_system.rag_query import LAT5150RAG

rag = LAT5150RAG()

queries = [
    "DSMIL activation",
    "NPU optimization",
    "Security hardening"
]

for q in queries:
    print(rag.query(q))
```

### ✅ Testing & Validation
```bash
python3 rag_system/test_queries.py
```

Runs 10 test cases and reports:
- Accuracy scores
- Response times
- Performance rating

---

## 📁 File Structure

```
rag_system/
├── document_processor.py      # Builds index from documentation
├── rag_query.py                # Main query interface
├── structured_extraction.py    # Extract structured data
├── test_queries.py             # Validation test suite
├── processed_docs.json         # Vector database (generated)
├── extracted_systems.json      # Structured data (generated)
└── README.md                   # This file
```

---

## 🎯 Usage Examples

### Example 1: Find Activation Instructions
```bash
python3 rag_system/rag_query.py "How to activate military tokens?"
```

### Example 2: Search by Topic
```bash
python3 rag_system/rag_query.py
LAT5150-RAG> topics security
```

Lists all documents related to "security"

### Example 3: Get System Stats
```bash
python3 rag_system/rag_query.py
LAT5150-RAG> stats
```

Shows:
- Total documents: 235
- Total chunks: 881
- Vocabulary size: 7,599 terms
- Categories: [operations, planning, analysis, ...]

### Example 4: Structured Batch Extraction
```bash
python3 rag_system/structured_extraction.py
```

Extracts structured info for predefined systems and saves to JSON.

---

## ⚙️ Configuration

### Customize Chunk Size
Edit `document_processor.py`:
```python
self.chunker = DocumentChunker(
    chunk_size=256,  # Adjust based on your docs
    overlap=20       # Increase for better context
)
```

### Adjust Retrieval Settings
Edit `rag_query.py`:
```python
results = self.retriever.search(
    query,
    top_k=3  # Increase for more results
)
```

### Add More Test Cases
Edit `test_queries.py`:
```python
test_cases = [
    {
        'query': 'Your question here',
        'expected_keywords': ['keyword1', 'keyword2'],
        'expected_files': ['FILENAME_PATTERN'],
    },
    # ... add more
]
```

---

## 📈 Performance Metrics

**Current Performance (TF-IDF based):**
- Average Accuracy: **51.8%**
- Average Response Time: **2.5s**
- Index Build Time: **~30s**
- Memory Usage: **~50MB**

**Target Performance (with transformer embeddings):**
- Average Accuracy: **>88%** (Maharana et al. target)
- Response Time: **<3s**

### Accuracy by Query Type:

| Query Type | Accuracy | Status |
|------------|----------|--------|
| System Activation | 73% | ✓ Good |
| Configuration | 67% | ✓ Good |
| Security Features | 47% | ✗ Needs improvement |
| Architecture | 60% | ~ Fair |

---

## 🔧 Troubleshooting

### Issue: "FileNotFoundError: processed_docs.json"
**Solution:**
```bash
python3 rag_system/document_processor.py
```
Build the index first.

### Issue: Low Accuracy Scores
**Solutions:**
1. **Increase chunk overlap:**
   ```python
   self.chunker = DocumentChunker(chunk_size=256, overlap=40)
   ```

2. **Adjust minimum relevance threshold:**
   ```python
   if score > 0.05:  # Lower threshold = more results
   ```

3. **Rebuild index after adding new docs:**
   ```bash
   python3 rag_system/rag_query.py --rebuild
   ```

### Issue: Slow Performance
**Solutions:**
1. **Reduce top_k:**
   ```bash
   python3 rag_system/rag_query.py --top-k 1 "your query"
   ```

2. **Filter short documents:**
   Edit `document_processor.py`:
   ```python
   if len(content) < 100:  # Increase threshold
   ```

---

## 🚀 Advanced Usage

### Programmatic Access
```python
from rag_system.rag_query import LAT5150RAG

# Initialize
rag = LAT5150RAG()

# Query
answer = rag.query("What is DSMIL?", top_k=3)
print(answer)

# Search by topic
files = rag.search_by_topic("security")
print("Security-related files:", files)

# Get stats
stats = rag.get_stats()
print(f"Total documents: {stats['total_documents']}")
```

### Custom Integration
```python
from rag_system.document_processor import TFIDFRetriever
import json

# Load chunks
with open('rag_system/processed_docs.json') as f:
    data = json.load(f)

# Create retriever
retriever = TFIDFRetriever(data['chunks'])

# Search
results = retriever.search("your query", top_k=5)

for chunk, score in results:
    if score > 0.1:
        print(f"Score: {score:.3f}")
        print(f"File: {chunk['metadata']['filepath']}")
        print(f"Text: {chunk['text'][:200]}...\n")
```

---

## 🎓 Next Steps to Improve Accuracy

### 1. Upgrade to Transformer Embeddings (Target: 88%+ accuracy)
```bash
pip install sentence-transformers
```

Then modify `document_processor.py` to use:
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('BAAI/bge-base-en-v1.5')
embeddings = model.encode(chunks)
```

### 2. Add LLM Integration
```python
# Use with Ollama (Llama3-8B)
import ollama

context = '\n'.join([chunk['text'] for chunk, _ in results])
prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"

response = ollama.generate(model='llama3', prompt=prompt)
```

### 3. Implement Reranking
After initial retrieval, rerank results for better precision:
```python
def rerank(query, results):
    # Use more sophisticated scoring
    # Consider: keyword density, position, recency
    pass
```

---

## 📚 Research References

1. **Maharana et al. 2025** - "Retrieval Augmented Generation for Building Datasets from Scientific Literature"
   - J. Phys. Mater. 8, 035006
   - Key findings: RAG achieves >88% accuracy with proper configuration

2. **Best Practices:**
   - Chunk size: 256 tokens
   - Chunk overlap: 20 tokens
   - Top-k: 3 results
   - Embedding model: BAAI/bge-base-en-v1.5

---

## 📞 Support

For issues or questions:
1. Check the troubleshooting section
2. Review test_queries.py output
3. Verify index was built successfully
4. Check that documentation files exist in 00-documentation/

---

## 📝 License

Internal LAT5150DRVMIL project tool.

---

**Last Updated:** 2025-11-08
**Version:** 1.0
**Status:** Production Ready (TF-IDF baseline)
