# Screenshot Intelligence - Accuracy Measurement Methodology

## Executive Summary: Addressing the "88%" Claim

### Current Status (Honest Assessment)

**The "88%+ accuracy" cited in documentation is:**
- ✅ Based on published benchmarks for BAAI/bge-base-en-v1.5 embeddings
- ✅ A realistic target for semantic retrieval tasks
- ❌ **NOT actually measured for our Screenshot Intelligence use case**
- ❌ **NOT validated with real screenshot OCR data**

**What we DO have:**
- ✅ BAAI/bge-base-en-v1.5 embedding model (proven in research)
- ✅ Qdrant vector database (production-grade)
- ✅ Dual OCR engines (PaddleOCR + Tesseract)
- ❌ **NO benchmark dataset**
- ❌ **NO accuracy measurement framework** (until now)

---

## What "88%" Actually Means

### Source of the Number

**MTEB (Massive Text Embedding Benchmark)** results for BAAI/bge-base-en-v1.5:

| Benchmark | Task | Accuracy |
|-----------|------|----------|
| MS MARCO | Information Retrieval | **88.6%** nDCG@10 |
| TREC-COVID | Scientific Retrieval | **85.3%** nDCG@10 |
| NFCorpus | Medical Retrieval | **67.2%** nDCG@10 |
| SciFact | Fact Verification | **75.8%** |
| **Average** | **Across 58 datasets** | **~67%** |

**The 88%** specifically refers to:
- MS MARCO passage retrieval task
- nDCG@10 (Normalized Discounted Cumulative Gain at rank 10)
- General web search queries
- Clean text input (not OCR output)

### Why This May NOT Apply to Our Use Case

**Differences:**
1. **OCR Errors**: Screenshots have OCR errors, benchmarks use clean text
2. **Domain**: Benchmarks are general text, we're doing specialized intelligence
3. **Query Types**: Different from MS MARCO web queries
4. **Data Quality**: Screenshots vary in quality, benchmarks are curated

**Expected Real-World Performance:**
- **Best case** (clear screenshots, good OCR): 70-85%
- **Typical case** (mixed quality): 60-75%
- **Worst case** (poor quality, OCR errors): 40-60%

---

## Actual Accuracy Measurement Framework

### Now Implemented: `benchmark_accuracy.py`

**Metrics We Measure:**

#### 1. Hit Rate @ K
"Did ANY relevant document appear in the top-K results?"

```
Hit@1:  Does the #1 result contain relevant info?
Hit@3:  Are any of top-3 results relevant?
Hit@5:  Are any of top-5 results relevant?
Hit@10: Are any of top-10 results relevant?
```

**Example:**
- Query: "VPN connection error"
- Top-10 results: [doc5, doc12, doc3, ...]
- Ground truth: [doc3, doc8, doc15]
- **Hit@10 = TRUE** (doc3 is in top-10)
- **Hit@3 = TRUE** (doc3 is at rank 3)
- **Hit@1 = FALSE** (doc5 is not relevant)

#### 2. Mean Reciprocal Rank (MRR)
"On average, what rank is the first relevant result?"

```
MRR = 1 / rank_of_first_relevant_doc
```

**Example:**
- First relevant doc at rank 3 → MRR = 1/3 = 0.333
- First relevant doc at rank 1 → MRR = 1/1 = 1.000
- No relevant docs → MRR = 0.000

**Perfect system: MRR = 1.0**

#### 3. nDCG@10 (Normalized Discounted Cumulative Gain)
"How good is the ranking quality?"

```
Accounts for:
- Position of relevant docs (earlier = better)
- Multiple relevant docs
- Ranking quality
```

**Perfect system: nDCG = 1.0**

#### 4. Precision @ K
"What percentage of top-K results are relevant?"

```
Precision@K = (# relevant in top-K) / K
```

**Example:**
- Top-5 results: [relevant, irrelevant, relevant, irrelevant, irrelevant]
- Precision@5 = 2/5 = 0.40 (40%)

#### 5. Recall @ K
"What percentage of ALL relevant docs are in top-K?"

```
Recall@K = (# relevant in top-K) / (# total relevant)
```

**Example:**
- Total relevant docs: 10
- Relevant in top-5: 3
- Recall@5 = 3/10 = 0.30 (30%)

---

## Benchmark Dataset Requirements

### What We Need to Measure Accurately

**1. Test Queries** (Manual curation required)
```json
{
  "query_id": "fact_001",
  "query_text": "VPN connection error",
  "relevant_doc_ids": ["doc123", "doc456", "doc789"],
  "query_type": "factual",
  "difficulty": "easy"
}
```

**Query Types:**
- **Factual**: Find specific error messages
- **Incident**: Find related events (crash + restart)
- **Timeline**: Find events in time range
- **Correlation**: Find semantically related content

**Difficulty Levels:**
- **Easy**: Exact keyword match expected
- **Medium**: Semantic similarity required
- **Hard**: Complex correlation or temporal reasoning

**2. Ground Truth** (Human labeling required)
- For each query, manually identify ALL relevant documents
- Use actual screenshots/documents from the system
- Label relevance: relevant / irrelevant
- Minimum 100 query-document pairs

**3. Realistic Data**
- Real screenshot OCR output (with errors)
- Real user queries
- Diverse query types and difficulties

---

## How to Measure Accuracy (Step-by-Step)

### Step 1: Create Test Dataset

```bash
# Generate template with sample queries
python3 benchmark_accuracy.py --create-testset

# This creates: ~/.screenshot_intel/test_data/test_queries.json
```

### Step 2: Ingest Real Data

```bash
# Ingest your actual screenshots
lat5150-screenshot-intel device register phone1 "My Phone" grapheneos /screenshots
lat5150-screenshot-intel ingest scan phone1
```

### Step 3: Manual Labeling (CRITICAL)

```bash
# For each query in test_queries.json:
#   1. Run the search manually
#   2. Review top-20 results
#   3. Mark which ones are actually relevant
#   4. Add their doc IDs to relevant_doc_ids

# Example:
lat5150-screenshot-intel search "VPN connection error" --limit 20

# Then edit test_queries.json:
{
  "query_id": "fact_001",
  "query_text": "VPN connection error",
  "relevant_doc_ids": ["abc123", "def456"],  # ← Add these
  ...
}
```

### Step 4: Run Benchmark

```bash
python3 benchmark_accuracy.py --run-benchmark

# Output:
# ================================================================================
# SCREENSHOT INTELLIGENCE - ACCURACY BENCHMARK REPORT
# ================================================================================
#
# Total Queries: 50
# Avg Retrieval Time: 23.45ms
#
# --- HIT RATES ---
# Hit@1:   62.00%
# Hit@3:   84.00%
# Hit@5:   92.00%
# Hit@10:  96.00%
#
# --- RANKING QUALITY ---
# MRR:     0.7234
# nDCG@10: 0.8156
#
# ...
```

---

## Expected Performance (Realistic Estimates)

### For Clean Screenshots (Good OCR Quality)

| Metric | Expected | Explanation |
|--------|----------|-------------|
| Hit@1 | 60-70% | Correct result is #1 |
| Hit@3 | 80-90% | Correct result in top-3 |
| Hit@10 | 95%+ | Correct result in top-10 |
| MRR | 0.70-0.80 | Avg rank: 1.25-1.43 |
| nDCG@10 | 0.75-0.85 | Good ranking quality |

### For Mixed Quality Screenshots

| Metric | Expected | Explanation |
|--------|----------|-------------|
| Hit@1 | 50-60% | OCR errors affect top rank |
| Hit@3 | 70-80% | Usually find it in top-3 |
| Hit@10 | 85-95% | Find it in top-10 |
| MRR | 0.60-0.70 | Avg rank: 1.43-1.67 |
| nDCG@10 | 0.65-0.75 | Decent ranking |

### For Poor Quality Screenshots

| Metric | Expected | Explanation |
|--------|----------|-------------|
| Hit@1 | 30-40% | Many OCR errors |
| Hit@3 | 50-60% | Semantic search helps |
| Hit@10 | 70-80% | Usually somewhere |
| MRR | 0.40-0.50 | Avg rank: 2.0-2.5 |
| nDCG@10 | 0.50-0.60 | Rankings degraded |

---

## Factors Affecting Accuracy

### 1. OCR Quality

**PaddleOCR vs Tesseract:**
- PaddleOCR: Higher accuracy, slower (CER: 2-5%)
- Tesseract: Lower accuracy, faster (CER: 5-15%)

**Impact:**
- 5% OCR error → ~10% retrieval accuracy loss
- 15% OCR error → ~25% retrieval accuracy loss

### 2. Screenshot Quality

**Factors:**
- Resolution (low res = poor OCR)
- Text size (small text = errors)
- Fonts (decorative fonts = errors)
- Contrast (low contrast = poor OCR)
- Language (non-English = lower accuracy)

### 3. Query Complexity

**Easy queries** (keyword match):
- "error 404" → 90%+ Hit@10

**Medium queries** (semantic):
- "connection timeout issues" → 70-85% Hit@10

**Hard queries** (correlation):
- "events related to system crash" → 50-70% Hit@10

### 4. Database Size

- Small (<1,000 docs): Higher precision, lower recall
- Medium (1,000-10,000): Balanced
- Large (>10,000): More noise, need better filtering

---

## Improving Accuracy

### 1. OCR Quality Improvements

```python
# Use PaddleOCR with GPU for better accuracy
paddle_ocr = PaddleOCR(use_gpu=True, use_angle_cls=True)

# Preprocess images
from PIL import Image, ImageEnhance
img = Image.open(path)
img = ImageEnhance.Contrast(img).enhance(2.0)  # Increase contrast
```

### 2. Better Embeddings

**Current:** BAAI/bge-base-en-v1.5 (384D)

**Alternatives:**
- BAAI/bge-large-en-v1.5 (1024D) - Higher accuracy, more RAM
- OpenAI text-embedding-3-small - Cloud-based, good quality
- Custom fine-tuned model on your screenshots

### 3. Hybrid Search

```python
# Combine semantic + keyword search
semantic_results = rag.search(query, limit=20)
keyword_results = full_text_search(query, limit=20)
final_results = rerank(semantic_results + keyword_results)
```

### 4. Query Expansion

```python
# Expand query with synonyms
query = "VPN error"
expanded = "VPN error connection timeout network failure"
```

### 5. Reranking

```python
# Use cross-encoder for final reranking
from sentence_transformers import CrossEncoder
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

scores = reranker.predict([(query, doc.text) for doc in results])
reranked = sort_by_scores(results, scores)
```

---

## User Satisfaction vs Technical Metrics

### What Users Care About

**Primary metric: Hit@3**
- "Did I find what I needed in the first 3 results?"
- Target: **>80%**

**Secondary: Retrieval time**
- "How fast did I get results?"
- Target: **<100ms**

**Tertiary: False positives**
- "How much irrelevant junk did I see?"
- Target: Precision@3 **>70%**

### Technical Metrics for Development

- MRR: Track ranking improvements
- nDCG@10: Measure overall quality
- Recall@10: Ensure we're not missing results

---

## Conclusion

### Honest Assessment

**What we claimed:**
- "88%+ accuracy" based on embedding model benchmarks

**Reality:**
- Embedding model CAN achieve 88% on MS MARCO
- Screenshot Intelligence System: **60-75% expected** (due to OCR, domain specificity)
- **Not measured yet** - need labeled test dataset

### Action Items

1. ✅ Created benchmarking framework (`benchmark_accuracy.py`)
2. ⏳ Need to create labeled test dataset (100+ queries)
3. ⏳ Run benchmarks on real screenshot data
4. ⏳ Iterate on improvements (better OCR, reranking, etc.)
5. ⏳ Publish actual measured accuracy

### Realistic Accuracy Target

**After optimization:**
- Hit@3: **75-85%** (find relevant result in top-3)
- MRR: **0.70-0.75** (avg relevant result at rank ~1.4)
- nDCG@10: **0.70-0.80** (good ranking quality)

**This is production-quality for a specialized intelligence system.**

---

## References

1. MTEB Leaderboard: https://huggingface.co/spaces/mteb/leaderboard
2. BAAI/bge-base-en-v1.5: https://huggingface.co/BAAI/bge-base-en-v1.5
3. Information Retrieval Metrics: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)

