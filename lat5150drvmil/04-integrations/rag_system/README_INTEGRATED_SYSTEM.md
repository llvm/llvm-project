# LAT5150DRVMIL Integrated RAG System

Complete unified RAG (Retrieval-Augmented Generation) system for cyber forensics with Jina Embeddings v3 integration.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    RAG Orchestrator                          │
│              (Main Driver & Coordination)                    │
└────────┬─────────────────────────────────────────┬──────────┘
         │                                         │
    ┌────▼────────┐                          ┌────▼─────────┐
    │  Ingestion  │                          │    Search    │
    │   Pipeline  │                          │   Pipeline   │
    └────┬────────┘                          └────┬─────────┘
         │                                         │
    ┌────▼──────────────┐                    ┌────▼──────────────┐
    │  1. Chunking      │                    │  1. Query Encoding│
    │     - Fixed       │                    │  2. Vector Search │
    │     - Semantic    │                    │  3. BM25 Search   │
    │     - Jina v3     │                    │  4. Hybrid Fusion │
    │     - Late        │                    │  5. Reranking     │
    └────┬──────────────┘                    └───────────────────┘
         │
    ┌────▼──────────────┐
    │  2. Embedding     │
    │     - BGE         │
    │     - Jina v3     │
    │     - E5          │
    └────┬──────────────┘
         │
    ┌────▼──────────────┐
    │  3. Vector Store  │
    │     - Qdrant      │
    │     - Quantization│
    │     - Multi-vector│
    └───────────────────┘
```

## Components

### Core Components

1. **RAG Orchestrator** (`rag_orchestrator.py`)
   - Main entry point and driver
   - Component initialization and coordination
   - Automatic fallback handling
   - Performance monitoring

2. **Configuration Management** (`rag_config.py`)
   - Centralized configuration
   - Preset configurations
   - Environment variable support
   - Runtime configuration

3. **Embedding Models** (`query_aware_embeddings.py`)
   - BGE (baseline): 768D, 512 tokens
   - Jina v3: 256-2084D, 8192 tokens
   - E5: Multilingual support
   - Task-specific adapters

4. **Chunking Strategies** (`chunking.py`, `late_chunking.py`)
   - Fixed-size chunking
   - Semantic chunking
   - Jina v3 optimized (1024-2048 tokens)
   - Late chunking (+3-4% nDCG)

5. **Vector Storage** (`jina_v3_qdrant_config.py`)
   - Qdrant integration
   - High-dimensional support (up to 65K)
   - Scalar quantization (4x compression)
   - Multi-vector ColBERT support

6. **Reranking** (`reranker.py`, `jina_reranker.py`)
   - Cross-encoder (local)
   - Jina Reranker v2 (local/API)
   - LLM-based reranking
   - Automatic fallback

7. **Hybrid Search** (`hybrid_search.py`)
   - Vector + BM25 fusion
   - Reciprocal Rank Fusion (RRF)
   - Configurable weighting
   - +5-8% overall improvement

## Quick Start

### Installation

```bash
# Navigate to RAG system directory
cd 04-integrations/rag_system

# Install dependencies
pip install -r requirements.txt

# Start Qdrant (required)
docker run -p 6333:6333 qdrant/qdrant
```

### Basic Usage

```python
from rag_orchestrator import RAGOrchestrator

# Create RAG system with preset configuration
rag = RAGOrchestrator.from_preset("jina_standard")

# Index documents
rag.index_document(
    text="VPN connection failed with timeout...",
    doc_id="vpn_log_001",
    metadata={"source": "vpn_logs.txt", "timestamp": "2024-01-15"}
)

# Batch indexing
documents = [
    {"text": "...", "id": "doc1", "metadata": {...}},
    {"text": "...", "id": "doc2", "metadata": {...}},
]
result = rag.index_document_batch(documents)

# Search
results = rag.search("VPN authentication timeout", top_k=10)

for result in results:
    print(f"Score: {result.score:.3f}")
    print(f"Text: {result.text}")
    print(f"Source: {result.source}")
```

### Configuration Presets

```python
# Available presets
presets = [
    "baseline",             # BGE-base, standard settings
    "jina_standard",        # Jina v3 1024D, balanced (recommended)
    "jina_high_accuracy",   # Jina v3 full 2084D, late chunking
    "jina_memory_optimized",# Jina v3 512D, quantization
    "jina_colbert",         # Multi-vector ColBERT mode
    "production",           # Production-ready configuration
]

# Use preset
rag = RAGOrchestrator.from_preset("jina_standard")

# Or customize
from rag_config import get_preset_config

config = get_preset_config("jina_standard")
config.search.use_reranking = False  # Disable reranking
config.embedding.use_gpu = False     # Use CPU

rag = RAGOrchestrator(config=config)
```

### Advanced Configuration

```python
from rag_config import RAGSystemConfig, EmbeddingConfig, VectorStoreConfig, ChunkingConfig, SearchConfig, RerankerConfig

# Custom configuration
config = RAGSystemConfig(
    embedding=EmbeddingConfig(
        model_name="jinaai/jina-embeddings-v3",
        dimension=1024,
        max_length=8192,
        matryoshka_dim=1024,
        use_gpu=True
    ),
    vector_store=VectorStoreConfig(
        store_type="qdrant",
        host="localhost",
        port=6333,
        collection_name="custom_collection",
        use_quantization=True,
        hnsw_m=32,
        hnsw_ef_search=128
    ),
    chunking=ChunkingConfig(
        strategy="late_chunking",
        chunk_size=512,
        chunk_overlap=100
    ),
    search=SearchConfig(
        mode="hybrid",
        vector_top_k=50,
        use_reranking=True
    ),
    reranker=RerankerConfig(
        model_name="jinaai/jina-reranker-v2-base-multilingual",
        top_k=10,
        use_gpu=True
    )
)

rag = RAGOrchestrator(config=config)
```

## Configuration Reference

### Preset Comparison

| Preset | Embedding | Dim | Context | Chunking | Reranking | Use Case |
|--------|-----------|-----|---------|----------|-----------|----------|
| baseline | BGE-base | 768D | 512 | Hybrid | Cross-encoder | Quick start |
| jina_standard | Jina v3 | 1024D | 8192 | Jina v3 | Jina Reranker | **Recommended** |
| jina_high_accuracy | Jina v3 | 2084D | 8192 | Late chunking | Jina Reranker | Best accuracy |
| jina_memory_optimized | Jina v3 | 512D | 8192 | Jina v3 | None | Large scale |
| jina_colbert | Jina v3 | 128D | 8192 | Late chunking | Jina Reranker | Multi-vector |
| production | Jina v3 | 1024D | 8192 | Jina v3 | Jina Reranker | Production |

### Performance Targets

| Configuration | Accuracy | Memory (1M docs) | Query Latency | Notes |
|---------------|----------|------------------|---------------|-------|
| baseline | 75-83% | ~3.1 GB | ~50ms | Fast, simple |
| jina_standard | 88-92% | ~2.1 GB (quantized) | ~200ms | **Recommended** |
| jina_high_accuracy | 95-97% | ~4.2 GB (quantized) | ~400ms | Best quality |
| jina_memory_optimized | 85-88% | ~1.0 GB (quantized) | ~150ms | Large scale |
| jina_colbert | 93-95% | ~20 GB | ~500ms | Best precision |

## Component Reference

### Embedding Models

```python
# BGE (baseline)
from query_aware_embeddings import QueryAwareEmbedder
embedder = QueryAwareEmbedder("BAAI/bge-base-en-v1.5")

# Jina v3
from query_aware_embeddings import JinaV3Embedder
embedder = JinaV3Embedder(
    output_dim=1024,  # 256, 512, 1024, or None (2084)
    task_adapter="retrieval"
)

# Encode
query_emb = embedder.encode_query("VPN timeout")
doc_embs = embedder.encode_document(["doc1", "doc2"])
```

### Chunking

```python
# Jina v3 optimized (recommended for Jina models)
from chunking import JinaV3Chunker
chunker = JinaV3Chunker(target_size=1024, max_size=2048)

# Late chunking (best accuracy)
from late_chunking import LateChunkingEncoder
chunker = LateChunkingEncoder(
    model_name="jinaai/jina-embeddings-v3",
    chunk_size=512,
    pooling="mean"
)

# Standard chunking
from chunking import create_chunker
chunker = create_chunker("hybrid", chunk_size=400)

# Chunk document
chunks = chunker.chunk(text, metadata={})
```

### Vector Storage

```python
# Qdrant with Jina v3
from jina_v3_qdrant_config import JinaV3QdrantStore, JINA_V3_CONFIGS

store = JinaV3QdrantStore(
    host="localhost",
    port=6333,
    config=JINA_V3_CONFIGS["jina_v3_1024d"]
)

# Index documents
store.index_documents([
    {"id": "doc1", "vector": embedding, "payload": {...}}
])

# Search
results = store.search(query_vector, limit=10)
```

### Reranking

```python
# Unified interface (automatic backend selection)
from reranker import create_reranker

# Cross-encoder (fast, local)
reranker = create_reranker("cross_encoder")

# Jina local (multilingual, local)
reranker = create_reranker("jina_local", use_gpu=True)

# Jina API (multilingual, cloud)
reranker = create_reranker("jina_api", api_key="your_key")

# Rerank
reranked = reranker.rerank(query, documents, top_k=10)
```

### Hybrid Search

```python
from hybrid_search import reciprocal_rank_fusion

# Get results from vector and BM25
vector_results = vector_search(query)
bm25_results = bm25_search(query)

# Merge with RRF
merged = reciprocal_rank_fusion(vector_results, bm25_results, k=60)
```

## Environment Variables

Configure via environment variables:

```bash
# Preset configuration
export RAG_PRESET="jina_standard"

# Model selection
export RAG_EMBEDDING_MODEL="jinaai/jina-embeddings-v3"
export RAG_VECTOR_STORE="qdrant"

# GPU settings
export RAG_USE_GPU="true"

# Qdrant connection
export QDRANT_HOST="localhost"
export QDRANT_PORT="6333"

# Jina API key (for cloud reranking)
export JINA_API_KEY="your_key_here"

# Use in code
from rag_orchestrator import RAGOrchestrator
rag = RAGOrchestrator.from_env()
```

## Examples

### Example 1: Document Ingestion Pipeline

```python
from rag_orchestrator import RAGOrchestrator
from pathlib import Path

# Initialize
rag = RAGOrchestrator.from_preset("jina_standard")

# Ingest directory of documents
doc_dir = Path("./forensics_data")
documents = []

for file_path in doc_dir.glob("*.txt"):
    with open(file_path) as f:
        text = f.read()

    documents.append({
        "text": text,
        "id": file_path.stem,
        "metadata": {
            "source": str(file_path),
            "type": "log",
            "timestamp": file_path.stat().st_mtime
        }
    })

# Batch index
result = rag.index_document_batch(documents, batch_size=10)
print(f"Indexed: {result['succeeded']}/{result['total']}")
```

### Example 2: Advanced Search

```python
# Multi-mode search comparison
query = "malware process injection behavior"

# Vector-only search
vector_results = rag.search(query, mode="vector", top_k=10)

# Hybrid search (recommended)
hybrid_results = rag.search(query, mode="hybrid", top_k=10)

# BM25-only search
bm25_results = rag.search(query, mode="bm25", top_k=10)

# Compare results
for i, (v, h, b) in enumerate(zip(vector_results, hybrid_results, bm25_results), 1):
    print(f"{i}.")
    print(f"  Vector: {v.doc_id} (score: {v.score:.3f})")
    print(f"  Hybrid: {h.doc_id} (score: {h.score:.3f})")
    print(f"  BM25:   {b.doc_id} (score: {b.score:.3f})")
```

### Example 3: Performance Monitoring

```python
# Get system statistics
stats = rag.get_stats()

print("System Statistics:")
print(f"Documents indexed: {stats['documents_indexed']}")
print(f"Queries processed: {stats['queries_processed']}")
print(f"Avg embedding time: {stats['total_embedding_time'] / max(stats['documents_indexed'], 1):.3f}s")
print(f"Avg search time: {stats['total_search_time'] / max(stats['queries_processed'], 1):.3f}s")
print(f"Avg rerank time: {stats['total_rerank_time'] / max(stats['queries_processed'], 1):.3f}s")

# Vector store stats
print("\nVector Store:")
print(f"Total points: {stats['vector_store']['points_count']}")
print(f"Collection: {stats['vector_store']['collection']}")
```

### Example 4: Save/Load Configuration

```python
from pathlib import Path

# Save current configuration
rag.save_config(Path("my_rag_config.json"))

# Load and use saved configuration
from rag_orchestrator import RAGOrchestrator

rag2 = RAGOrchestrator.from_config_file(Path("my_rag_config.json"))
```

## Troubleshooting

### Qdrant Connection Issues

```python
# Check if Qdrant is running
import requests
try:
    response = requests.get("http://localhost:6333/collections")
    print(f"Qdrant status: {response.status_code}")
except:
    print("Qdrant not running. Start with: docker run -p 6333:6333 qdrant/qdrant")
```

### Out of Memory

```python
# Use memory-optimized preset
rag = RAGOrchestrator.from_preset("jina_memory_optimized")

# Or reduce embedding dimension
config = get_preset_config("jina_standard")
config.embedding.matryoshka_dim = 512  # Reduce from 1024 to 512
rag = RAGOrchestrator(config=config)

# Enable quantization
config.vector_store.use_quantization = True
config.vector_store.quantization_type = "scalar"
```

### Slow Search

```python
# Disable reranking for speed
config = get_preset_config("jina_standard")
config.search.use_reranking = False
rag = RAGOrchestrator(config=config)

# Or use smaller top-k for reranking
config.search.vector_top_k = 20  # Reduce from 50
config.reranker.top_k = 5  # Reduce from 10
```

### Low Accuracy

```python
# Use high-accuracy preset
rag = RAGOrchestrator.from_preset("jina_high_accuracy")

# Or increase search parameters
config = get_preset_config("jina_standard")
config.vector_store.hnsw_ef_search = 256  # Increase from 128
config.search.vector_top_k = 100  # More candidates for reranking
```

## Performance Tuning

### Embedding Performance

```python
# Batch size tuning
config.embedding.batch_size = 32  # Larger = faster but more memory

# GPU utilization
config.embedding.use_gpu = True
config.reranker.use_gpu = True
```

### Search Performance

```python
# HNSW tuning (accuracy vs speed)
config.vector_store.hnsw_m = 16           # Lower = faster, less accurate
config.vector_store.hnsw_ef_search = 64   # Lower = faster, less accurate

# Quantization (memory vs accuracy)
config.vector_store.use_quantization = True
config.vector_store.quantization_type = "scalar"  # 4x compression, <1% loss
```

## Best Practices

1. **Start with preset**: Use "jina_standard" for most use cases
2. **Enable quantization**: Saves 75% memory with minimal accuracy loss
3. **Use hybrid search**: +5-8% improvement over vector-only
4. **Enable reranking**: +5-10% precision on top-K results
5. **Batch indexing**: 10-100x faster than one-by-one
6. **Monitor performance**: Track stats and optimize bottlenecks
7. **Use late chunking**: +3-4% nDCG for Jina v3 models

## API Reference

See individual component documentation:
- [Configuration](rag_config.py) - Centralized configuration management
- [Orchestrator](rag_orchestrator.py) - Main driver and API
- [Embeddings](query_aware_embeddings.py) - Embedding models
- [Jina v3 Guide](JINA_V3_INTEGRATION.md) - Complete Jina v3 documentation
- [Chunking](chunking.py) - Chunking strategies
- [Late Chunking](late_chunking.py) - Contextual chunking
- [Vector Store](jina_v3_qdrant_config.py) - Qdrant configuration
- [Reranking](reranker.py) - Reranking strategies
- [Hybrid Search](hybrid_search.py) - Hybrid retrieval

## Support

- **Issues**: Report in project issue tracker
- **Documentation**: See `JINA_V3_INTEGRATION.md` for detailed Jina v3 guide
- **Benchmarks**: Run `benchmark_jina_vs_bge.py` for performance comparisons

---

**Version**: 1.0.0
**Last Updated**: 2025-01-12
**Status**: Production Ready ✅
