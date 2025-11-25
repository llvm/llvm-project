#!/usr/bin/env python3
"""
Unified RAG System Configuration

Central configuration management for all RAG components:
- Embedding models (BGE, Jina v3)
- Vector stores (Qdrant, ChromaDB)
- Rerankers (Cross-encoder, Jina)
- Chunking strategies
- Search modes (vector, hybrid, multi-vector)

Provides:
- Configuration presets for common scenarios
- Environment variable support
- Runtime configuration switching
- Validation and defaults
"""

import os
import json
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict, field
from pathlib import Path
from enum import Enum

logger = logging.getLogger(__name__)


class EmbeddingModel(Enum):
    """Available embedding models"""
    BGE_BASE = "BAAI/bge-base-en-v1.5"
    BGE_LARGE = "BAAI/bge-large-en-v1.5"
    BGE_SMALL = "BAAI/bge-small-en-v1.5"
    JINA_V3 = "jinaai/jina-embeddings-v3"
    E5_BASE = "intfloat/e5-base-v2"
    E5_LARGE = "intfloat/e5-large-v2"


class VectorStore(Enum):
    """Available vector stores"""
    QDRANT = "qdrant"
    CHROMADB = "chromadb"
    MEMORY = "memory"


class RerankerModel(Enum):
    """Available reranker models"""
    CROSS_ENCODER_MINI = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    CROSS_ENCODER_LARGE = "cross-encoder/ms-marco-MiniLM-L-12-v2"
    CROSS_ENCODER_TINY = "cross-encoder/ms-marco-TinyBERT-L-2-v2"
    JINA_RERANKER = "jinaai/jina-reranker-v2-base-multilingual"
    JINA_RERANKER_API = "jina-api"


class ChunkingStrategy(Enum):
    """Available chunking strategies"""
    FIXED = "fixed"
    SENTENCE = "sentence"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"
    JINA_V3 = "jina_v3"
    LATE_CHUNKING = "late_chunking"


class SearchMode(Enum):
    """Search modes"""
    VECTOR_ONLY = "vector"
    HYBRID = "hybrid"
    MULTI_VECTOR = "multi_vector"
    BM25_ONLY = "bm25"


@dataclass
class EmbeddingConfig:
    """Embedding model configuration"""
    model_name: str
    dimension: int
    max_length: int = 512
    use_gpu: bool = True
    matryoshka_dim: Optional[int] = None  # For Jina v3
    task_adapter: str = "retrieval"  # For Jina v3
    normalize: bool = True
    batch_size: int = 32


@dataclass
class VectorStoreConfig:
    """Vector store configuration"""
    store_type: str  # qdrant, chromadb, memory
    host: str = "localhost"
    port: int = 6333
    collection_name: str = "rag_collection"

    # HNSW parameters
    hnsw_m: int = 32
    hnsw_ef_construct: int = 200
    hnsw_ef_search: int = 128

    # Quantization
    use_quantization: bool = True
    quantization_type: str = "scalar"  # scalar or binary

    # Multi-vector
    multi_vector: bool = False
    multi_vector_comparator: str = "max_sim"

    # Persistence
    persist_directory: Optional[str] = None


@dataclass
class RerankerConfig:
    """Reranker configuration"""
    model_name: str
    top_k: int = 10
    use_gpu: bool = True
    api_key: Optional[str] = None  # For Jina API
    batch_size: int = 16
    max_length: int = 512


@dataclass
class ChunkingConfig:
    """Chunking configuration"""
    strategy: str  # fixed, sentence, semantic, hybrid, jina_v3, late_chunking
    chunk_size: int = 400
    chunk_overlap: int = 50
    max_chunk_size: int = 500
    min_chunk_size: int = 100

    # Jina v3 specific
    jina_target_size: int = 1024
    jina_max_size: int = 2048

    # Late chunking
    late_chunking_pooling: str = "mean"  # mean, max, cls


@dataclass
class SearchConfig:
    """Search configuration"""
    mode: str  # vector, hybrid, multi_vector, bm25

    # Vector search
    vector_top_k: int = 50
    vector_score_threshold: float = 0.5

    # BM25 parameters
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    bm25_top_k: int = 50

    # Hybrid fusion
    hybrid_alpha: float = 0.5  # Weight for dense vs sparse (0=sparse, 1=dense)
    rrf_k: int = 60  # Reciprocal rank fusion constant

    # Reranking
    use_reranking: bool = True
    rerank_top_k: int = 10


@dataclass
class RAGSystemConfig:
    """Complete RAG system configuration"""
    embedding: EmbeddingConfig
    vector_store: VectorStoreConfig
    chunking: ChunkingConfig
    search: SearchConfig
    reranker: Optional[RerankerConfig] = None

    # System settings
    log_level: str = "INFO"
    cache_dir: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'RAGSystemConfig':
        """Create from dictionary"""
        return cls(
            embedding=EmbeddingConfig(**config_dict.get('embedding', {})),
            vector_store=VectorStoreConfig(**config_dict.get('vector_store', {})),
            chunking=ChunkingConfig(**config_dict.get('chunking', {})),
            search=SearchConfig(**config_dict.get('search', {})),
            reranker=RerankerConfig(**config_dict['reranker']) if 'reranker' in config_dict else None,
            log_level=config_dict.get('log_level', 'INFO'),
            cache_dir=config_dict.get('cache_dir')
        )

    def save(self, filepath: Path):
        """Save configuration to file"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Configuration saved to {filepath}")

    @classmethod
    def load(cls, filepath: Path) -> 'RAGSystemConfig':
        """Load configuration from file"""
        with open(filepath) as f:
            config_dict = json.load(f)
        logger.info(f"Configuration loaded from {filepath}")
        return cls.from_dict(config_dict)


# ============================================================================
# Preset Configurations
# ============================================================================

def get_preset_config(preset_name: str) -> RAGSystemConfig:
    """
    Get preset configuration by name

    Available presets:
    - baseline: BGE-base + Qdrant + basic settings
    - jina_standard: Jina v3 1024D + optimized settings
    - jina_high_accuracy: Jina v3 full 2084D + late chunking + reranking
    - jina_memory_optimized: Jina v3 512D + quantization
    - jina_colbert: Jina v3 multi-vector ColBERT mode
    - production: Balanced production configuration
    """

    presets = {
        "baseline": _get_baseline_config(),
        "jina_standard": _get_jina_standard_config(),
        "jina_high_accuracy": _get_jina_high_accuracy_config(),
        "jina_memory_optimized": _get_jina_memory_optimized_config(),
        "jina_colbert": _get_jina_colbert_config(),
        "production": _get_production_config(),
    }

    if preset_name not in presets:
        raise ValueError(f"Unknown preset: {preset_name}. Available: {list(presets.keys())}")

    return presets[preset_name]


def _get_baseline_config() -> RAGSystemConfig:
    """Baseline configuration with BGE-base"""
    return RAGSystemConfig(
        embedding=EmbeddingConfig(
            model_name=EmbeddingModel.BGE_BASE.value,
            dimension=768,
            max_length=512,
            use_gpu=True
        ),
        vector_store=VectorStoreConfig(
            store_type=VectorStore.QDRANT.value,
            collection_name="rag_baseline",
            hnsw_m=16,
            hnsw_ef_construct=100,
            hnsw_ef_search=64,
            use_quantization=False
        ),
        chunking=ChunkingConfig(
            strategy=ChunkingStrategy.HYBRID.value,
            chunk_size=400,
            chunk_overlap=50
        ),
        search=SearchConfig(
            mode=SearchMode.HYBRID.value,
            vector_top_k=50,
            use_reranking=True
        ),
        reranker=RerankerConfig(
            model_name=RerankerModel.CROSS_ENCODER_MINI.value,
            top_k=10
        )
    )


def _get_jina_standard_config() -> RAGSystemConfig:
    """Jina v3 standard configuration (balanced)"""
    return RAGSystemConfig(
        embedding=EmbeddingConfig(
            model_name=EmbeddingModel.JINA_V3.value,
            dimension=1024,
            max_length=8192,
            use_gpu=True,
            matryoshka_dim=1024,
            task_adapter="retrieval"
        ),
        vector_store=VectorStoreConfig(
            store_type=VectorStore.QDRANT.value,
            collection_name="rag_jina_1024d",
            hnsw_m=32,
            hnsw_ef_construct=200,
            hnsw_ef_search=128,
            use_quantization=True,
            quantization_type="scalar"
        ),
        chunking=ChunkingConfig(
            strategy=ChunkingStrategy.JINA_V3.value,
            jina_target_size=1024,
            jina_max_size=2048
        ),
        search=SearchConfig(
            mode=SearchMode.HYBRID.value,
            vector_top_k=50,
            use_reranking=True
        ),
        reranker=RerankerConfig(
            model_name=RerankerModel.JINA_RERANKER.value,
            top_k=10,
            use_gpu=True
        )
    )


def _get_jina_high_accuracy_config() -> RAGSystemConfig:
    """Jina v3 high accuracy configuration"""
    return RAGSystemConfig(
        embedding=EmbeddingConfig(
            model_name=EmbeddingModel.JINA_V3.value,
            dimension=2084,
            max_length=8192,
            use_gpu=True,
            matryoshka_dim=None,  # Full 2084D
            task_adapter="retrieval"
        ),
        vector_store=VectorStoreConfig(
            store_type=VectorStore.QDRANT.value,
            collection_name="rag_jina_full",
            hnsw_m=32,
            hnsw_ef_construct=200,
            hnsw_ef_search=256,  # Higher for better recall
            use_quantization=True,
            quantization_type="scalar"
        ),
        chunking=ChunkingConfig(
            strategy=ChunkingStrategy.LATE_CHUNKING.value,
            chunk_size=512,
            chunk_overlap=100,
            late_chunking_pooling="mean"
        ),
        search=SearchConfig(
            mode=SearchMode.HYBRID.value,
            vector_top_k=100,  # More candidates for reranking
            use_reranking=True
        ),
        reranker=RerankerConfig(
            model_name=RerankerModel.JINA_RERANKER.value,
            top_k=10,
            use_gpu=True
        )
    )


def _get_jina_memory_optimized_config() -> RAGSystemConfig:
    """Jina v3 memory-optimized configuration"""
    return RAGSystemConfig(
        embedding=EmbeddingConfig(
            model_name=EmbeddingModel.JINA_V3.value,
            dimension=512,
            max_length=8192,
            use_gpu=True,
            matryoshka_dim=512,
            task_adapter="retrieval"
        ),
        vector_store=VectorStoreConfig(
            store_type=VectorStore.QDRANT.value,
            collection_name="rag_jina_512d",
            hnsw_m=16,
            hnsw_ef_construct=100,
            hnsw_ef_search=64,
            use_quantization=True,
            quantization_type="scalar"
        ),
        chunking=ChunkingConfig(
            strategy=ChunkingStrategy.JINA_V3.value,
            jina_target_size=1024,
            jina_max_size=2048
        ),
        search=SearchConfig(
            mode=SearchMode.HYBRID.value,
            vector_top_k=50,
            use_reranking=False  # Skip reranking for speed
        )
    )


def _get_jina_colbert_config() -> RAGSystemConfig:
    """Jina v3 ColBERT multi-vector configuration"""
    return RAGSystemConfig(
        embedding=EmbeddingConfig(
            model_name=EmbeddingModel.JINA_V3.value,
            dimension=128,  # Typical ColBERT dimension
            max_length=8192,
            use_gpu=True,
            task_adapter="retrieval"
        ),
        vector_store=VectorStoreConfig(
            store_type=VectorStore.QDRANT.value,
            collection_name="rag_jina_colbert",
            hnsw_m=16,
            hnsw_ef_construct=100,
            hnsw_ef_search=64,
            use_quantization=False,
            multi_vector=True,
            multi_vector_comparator="max_sim"
        ),
        chunking=ChunkingConfig(
            strategy=ChunkingStrategy.LATE_CHUNKING.value,
            chunk_size=512,
            chunk_overlap=0,
            late_chunking_pooling="mean"
        ),
        search=SearchConfig(
            mode=SearchMode.MULTI_VECTOR.value,
            vector_top_k=50,
            use_reranking=True
        ),
        reranker=RerankerConfig(
            model_name=RerankerModel.JINA_RERANKER.value,
            top_k=10
        )
    )


def _get_production_config() -> RAGSystemConfig:
    """Production-ready configuration"""
    return RAGSystemConfig(
        embedding=EmbeddingConfig(
            model_name=EmbeddingModel.JINA_V3.value,
            dimension=1024,
            max_length=8192,
            use_gpu=True,
            matryoshka_dim=1024,
            task_adapter="retrieval",
            batch_size=16  # Conservative batch size
        ),
        vector_store=VectorStoreConfig(
            store_type=VectorStore.QDRANT.value,
            collection_name="rag_production",
            hnsw_m=32,
            hnsw_ef_construct=200,
            hnsw_ef_search=128,
            use_quantization=True,
            quantization_type="scalar",
            persist_directory="./data/qdrant"
        ),
        chunking=ChunkingConfig(
            strategy=ChunkingStrategy.JINA_V3.value,
            jina_target_size=1024,
            jina_max_size=2048
        ),
        search=SearchConfig(
            mode=SearchMode.HYBRID.value,
            vector_top_k=50,
            bm25_top_k=50,
            hybrid_alpha=0.6,  # Favor vector search slightly
            use_reranking=True,
            rerank_top_k=10
        ),
        reranker=RerankerConfig(
            model_name=RerankerModel.JINA_RERANKER.value,
            top_k=10,
            use_gpu=True,
            batch_size=16
        ),
        log_level="INFO",
        cache_dir="./cache"
    )


# ============================================================================
# Configuration from Environment
# ============================================================================

def get_config_from_env() -> RAGSystemConfig:
    """
    Create configuration from environment variables

    Environment variables:
    - RAG_PRESET: Preset name (baseline, jina_standard, etc.)
    - RAG_EMBEDDING_MODEL: Embedding model name
    - RAG_VECTOR_STORE: Vector store type
    - RAG_USE_GPU: Use GPU (true/false)
    - QDRANT_HOST: Qdrant host
    - QDRANT_PORT: Qdrant port
    - JINA_API_KEY: Jina API key for reranker
    """
    # Check for preset
    preset = os.getenv("RAG_PRESET")
    if preset:
        config = get_preset_config(preset)
        logger.info(f"Using preset configuration: {preset}")
    else:
        config = get_preset_config("production")
        logger.info("Using default production configuration")

    # Override with environment variables
    if os.getenv("RAG_EMBEDDING_MODEL"):
        config.embedding.model_name = os.getenv("RAG_EMBEDDING_MODEL")

    if os.getenv("RAG_VECTOR_STORE"):
        config.vector_store.store_type = os.getenv("RAG_VECTOR_STORE")

    if os.getenv("RAG_USE_GPU"):
        use_gpu = os.getenv("RAG_USE_GPU").lower() == "true"
        config.embedding.use_gpu = use_gpu
        if config.reranker:
            config.reranker.use_gpu = use_gpu

    if os.getenv("QDRANT_HOST"):
        config.vector_store.host = os.getenv("QDRANT_HOST")

    if os.getenv("QDRANT_PORT"):
        config.vector_store.port = int(os.getenv("QDRANT_PORT"))

    if os.getenv("JINA_API_KEY") and config.reranker:
        config.reranker.api_key = os.getenv("JINA_API_KEY")

    return config


# ============================================================================
# Helpers
# ============================================================================

def print_config(config: RAGSystemConfig):
    """Print configuration in human-readable format"""
    print("\n" + "="*80)
    print("RAG SYSTEM CONFIGURATION")
    print("="*80 + "\n")

    print(f"EMBEDDING MODEL:")
    print(f"  Model: {config.embedding.model_name}")
    print(f"  Dimension: {config.embedding.dimension}D")
    print(f"  Max length: {config.embedding.max_length} tokens")
    print(f"  GPU: {config.embedding.use_gpu}")
    if config.embedding.matryoshka_dim:
        print(f"  Matryoshka: {config.embedding.matryoshka_dim}D")
    print()

    print(f"VECTOR STORE:")
    print(f"  Type: {config.vector_store.store_type}")
    print(f"  Collection: {config.vector_store.collection_name}")
    print(f"  HNSW: m={config.vector_store.hnsw_m}, ef={config.vector_store.hnsw_ef_search}")
    print(f"  Quantization: {config.vector_store.use_quantization} ({config.vector_store.quantization_type})")
    print(f"  Multi-vector: {config.vector_store.multi_vector}")
    print()

    print(f"CHUNKING:")
    print(f"  Strategy: {config.chunking.strategy}")
    if "jina" in config.chunking.strategy:
        print(f"  Target size: {config.chunking.jina_target_size} tokens")
        print(f"  Max size: {config.chunking.jina_max_size} tokens")
    else:
        print(f"  Chunk size: {config.chunking.chunk_size} tokens")
        print(f"  Overlap: {config.chunking.chunk_overlap} tokens")
    print()

    print(f"SEARCH:")
    print(f"  Mode: {config.search.mode}")
    print(f"  Top-K: {config.search.vector_top_k}")
    print(f"  Reranking: {config.search.use_reranking}")
    if config.search.use_reranking and config.reranker:
        print(f"  Reranker: {config.reranker.model_name}")
        print(f"  Rerank top-K: {config.reranker.top_k}")
    print()

    print("="*80 + "\n")


if __name__ == "__main__":
    # Demo all presets
    print("Available preset configurations:\n")

    for preset_name in ["baseline", "jina_standard", "jina_high_accuracy",
                        "jina_memory_optimized", "jina_colbert", "production"]:
        config = get_preset_config(preset_name)
        print(f"\n{'='*80}")
        print(f"PRESET: {preset_name.upper()}")
        print(f"{'='*80}")
        print(f"Embedding: {config.embedding.model_name} ({config.embedding.dimension}D)")
        print(f"Chunking: {config.chunking.strategy}")
        print(f"Search: {config.search.mode}")
        print(f"Reranking: {config.reranker.model_name if config.reranker else 'None'}")

    print("\n" + "="*80 + "\n")

    # Save example config
    config = get_preset_config("production")
    config.save(Path("rag_config_example.json"))
    print("Example configuration saved to: rag_config_example.json")
