#!/usr/bin/env python3
"""
RAG System Orchestrator - Main Driver

Unified interface for the complete RAG system integrating:
- Embedding models (BGE, Jina v3)
- Chunking strategies (fixed, semantic, late chunking)
- Vector stores (Qdrant)
- Search modes (vector, hybrid, multi-vector)
- Reranking (cross-encoder, Jina reranker)

Provides:
- Simple high-level API
- Automatic component initialization
- Configuration management
- Error handling and fallbacks
- Performance monitoring

Usage:
    # Quick start with preset
    rag = RAGOrchestrator.from_preset("jina_standard")
    rag.index_documents(["doc1.txt", "doc2.txt"])
    results = rag.search("VPN timeout error")

    # Advanced configuration
    config = get_preset_config("jina_high_accuracy")
    rag = RAGOrchestrator(config)
    rag.index_document_batch(documents)
    results = rag.search("malware analysis", top_k=10)
"""

import logging
import time
from typing import List, Dict, Optional, Any, Union
from pathlib import Path
from dataclasses import dataclass
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Search result from RAG system"""
    doc_id: str
    text: str
    score: float
    rank: int
    source: str  # vector, bm25, hybrid, reranked
    metadata: Dict[str, Any]

    # Performance metrics
    vector_score: Optional[float] = None
    bm25_score: Optional[float] = None
    rerank_score: Optional[float] = None


class RAGOrchestrator:
    """
    Main RAG System Orchestrator

    Coordinates all components:
    - Document ingestion and chunking
    - Embedding generation
    - Vector storage
    - Search and retrieval
    - Reranking

    Features:
    - Automatic component initialization
    - Graceful degradation
    - Performance monitoring
    - Batch processing
    """

    def __init__(
        self,
        config: Optional['RAGSystemConfig'] = None,
        config_path: Optional[Path] = None,
        preset: str = "production"
    ):
        """
        Initialize RAG orchestrator

        Args:
            config: RAGSystemConfig object
            config_path: Path to configuration file
            preset: Preset name (if config not provided)
        """
        # Load configuration
        if config:
            self.config = config
        elif config_path:
            from rag_config import RAGSystemConfig
            self.config = RAGSystemConfig.load(config_path)
        else:
            from rag_config import get_preset_config
            self.config = get_preset_config(preset)

        # Set logging level
        logging.getLogger().setLevel(self.config.log_level)

        # Initialize components
        self.embedder = None
        self.chunker = None
        self.vector_store = None
        self.bm25_index = None
        self.reranker = None

        # Performance tracking
        self.stats = {
            'documents_indexed': 0,
            'queries_processed': 0,
            'total_embedding_time': 0.0,
            'total_search_time': 0.0,
            'total_rerank_time': 0.0
        }

        # Initialize components
        self._initialize_components()

        logger.info("RAG Orchestrator initialized successfully")

    def _initialize_components(self):
        """Initialize all RAG components"""
        try:
            self._initialize_embedder()
            self._initialize_chunker()
            self._initialize_vector_store()

            # Optional components
            if self.config.search.mode == "hybrid" or self.config.search.mode == "bm25":
                self._initialize_bm25()

            if self.config.search.use_reranking and self.config.reranker:
                self._initialize_reranker()

        except Exception as e:
            logger.error(f"Failed to initialize RAG components: {e}")
            raise

    def _initialize_embedder(self):
        """Initialize embedding model"""
        logger.info("Initializing embedder...")

        model_name = self.config.embedding.model_name

        # Check if Jina v3
        if "jina" in model_name.lower():
            from query_aware_embeddings import JinaV3Embedder

            self.embedder = JinaV3Embedder(
                model_name=model_name,
                use_gpu=self.config.embedding.use_gpu,
                output_dim=self.config.embedding.matryoshka_dim,
                task_adapter=self.config.embedding.task_adapter
            )
        else:
            from query_aware_embeddings import QueryAwareEmbedder

            self.embedder = QueryAwareEmbedder(
                model_name=model_name,
                use_gpu=self.config.embedding.use_gpu
            )

        logger.info(f"✓ Embedder loaded: {model_name} ({self.embedder.get_embedding_dim()}D)")

    def _initialize_chunker(self):
        """Initialize chunking strategy"""
        logger.info("Initializing chunker...")

        strategy = self.config.chunking.strategy

        if strategy == "late_chunking":
            from late_chunking import LateChunkingEncoder

            self.chunker = LateChunkingEncoder(
                model_name=self.config.embedding.model_name,
                chunk_size=self.config.chunking.chunk_size,
                overlap=self.config.chunking.chunk_overlap,
                pooling=self.config.chunking.late_chunking_pooling,
                use_gpu=self.config.embedding.use_gpu
            )
        elif strategy == "jina_v3":
            from chunking import JinaV3Chunker

            self.chunker = JinaV3Chunker(
                target_size=self.config.chunking.jina_target_size,
                max_size=self.config.chunking.jina_max_size
            )
        else:
            from chunking import create_chunker

            self.chunker = create_chunker(
                strategy=strategy,
                chunk_size=self.config.chunking.chunk_size,
                chunk_overlap=self.config.chunking.chunk_overlap,
                max_size=self.config.chunking.max_chunk_size,
                min_size=self.config.chunking.min_chunk_size
            )

        logger.info(f"✓ Chunker initialized: {strategy}")

    def _initialize_vector_store(self):
        """Initialize vector store"""
        logger.info("Initializing vector store...")

        store_type = self.config.vector_store.store_type

        if store_type == "qdrant":
            from jina_v3_qdrant_config import JinaV3QdrantStore, QdrantConfig

            # Create Qdrant config
            qdrant_config = QdrantConfig(
                collection_name=self.config.vector_store.collection_name,
                vector_size=self.config.embedding.dimension,
                hnsw_m=self.config.vector_store.hnsw_m,
                hnsw_ef_construct=self.config.vector_store.hnsw_ef_construct,
                hnsw_ef_search=self.config.vector_store.hnsw_ef_search,
                use_quantization=self.config.vector_store.use_quantization,
                quantization_type=self.config.vector_store.quantization_type
            )

            self.vector_store = JinaV3QdrantStore(
                host=self.config.vector_store.host,
                port=self.config.vector_store.port,
                config=qdrant_config
            )

        else:
            raise ValueError(f"Unsupported vector store: {store_type}")

        logger.info(f"✓ Vector store initialized: {store_type}")

    def _initialize_bm25(self):
        """Initialize BM25 index for hybrid search"""
        logger.info("Initializing BM25 index...")

        from hybrid_search import BM25Index

        self.bm25_index = BM25Index(
            k1=self.config.search.bm25_k1,
            b=self.config.search.bm25_b
        )

        logger.info("✓ BM25 index initialized")

    def _initialize_reranker(self):
        """Initialize reranker"""
        logger.info("Initializing reranker...")

        from reranker import create_reranker

        model_name = self.config.reranker.model_name

        # Determine reranker type
        if "jina" in model_name.lower():
            if self.config.reranker.api_key:
                reranker_type = "jina_api"
            else:
                reranker_type = "jina_local"
        else:
            reranker_type = "cross_encoder"

        self.reranker = create_reranker(
            reranker_type=reranker_type,
            model_name=model_name,
            use_gpu=self.config.reranker.use_gpu,
            api_key=self.config.reranker.api_key
        )

        logger.info(f"✓ Reranker initialized: {model_name}")

    # ========================================================================
    # Document Ingestion
    # ========================================================================

    def index_document(
        self,
        text: str,
        doc_id: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Index a single document

        Args:
            text: Document text
            doc_id: Document ID (generated if not provided)
            metadata: Document metadata

        Returns:
            Ingestion result
        """
        start_time = time.time()

        try:
            # Generate doc ID if not provided
            if doc_id is None:
                import hashlib
                doc_id = hashlib.sha256(text.encode()).hexdigest()[:16]

            # Chunk document
            if hasattr(self.chunker, 'encode_document_with_late_chunking'):
                # Late chunking
                chunks = self.chunker.encode_document_with_late_chunking(
                    text=text,
                    metadata=metadata
                )
                # Chunks already have embeddings
                documents = [
                    {
                        'id': f"{doc_id}_chunk_{chunk.chunk_id}",
                        'vector': chunk.embedding,
                        'payload': {
                            'text': chunk.text,
                            'doc_id': doc_id,
                            'chunk_id': chunk.chunk_id,
                            **(metadata or {})
                        }
                    }
                    for chunk in chunks
                ]
            else:
                # Standard chunking
                chunks = self.chunker.chunk(text, metadata)

                # Generate embeddings
                chunk_texts = [chunk.text for chunk in chunks]
                if hasattr(self.embedder, 'encode_document'):
                    embeddings = self.embedder.encode_document(chunk_texts)
                else:
                    embeddings = [self.embedder.encode_query(t) for t in chunk_texts]

                # Prepare documents
                documents = [
                    {
                        'id': f"{doc_id}_chunk_{i}",
                        'vector': emb,
                        'payload': {
                            'text': chunk.text,
                            'doc_id': doc_id,
                            'chunk_id': i,
                            **(metadata or {})
                        }
                    }
                    for i, (chunk, emb) in enumerate(zip(chunks, embeddings))
                ]

            # Index in vector store
            result = self.vector_store.index_documents(documents)

            # Add to BM25 if available
            if self.bm25_index:
                bm25_docs = [
                    {
                        'id': doc['id'],
                        'text': doc['payload']['text'],
                        **doc['payload']
                    }
                    for doc in documents
                ]
                self.bm25_index.add_documents(bm25_docs)

            # Update stats
            self.stats['documents_indexed'] += 1
            self.stats['total_embedding_time'] += time.time() - start_time

            return {
                'status': 'success',
                'doc_id': doc_id,
                'chunks': len(chunks),
                'time': time.time() - start_time
            }

        except Exception as e:
            logger.error(f"Failed to index document: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }

    def index_document_batch(
        self,
        documents: List[Dict[str, Any]],
        batch_size: int = 10
    ) -> Dict[str, Any]:
        """
        Index multiple documents in batches

        Args:
            documents: List of dicts with 'text', 'id' (optional), 'metadata' (optional)
            batch_size: Batch size for processing

        Returns:
            Batch ingestion results
        """
        results = {
            'total': len(documents),
            'succeeded': 0,
            'failed': 0,
            'errors': []
        }

        for i, doc in enumerate(documents):
            result = self.index_document(
                text=doc.get('text', ''),
                doc_id=doc.get('id'),
                metadata=doc.get('metadata')
            )

            if result['status'] == 'success':
                results['succeeded'] += 1
            else:
                results['failed'] += 1
                results['errors'].append({
                    'index': i,
                    'error': result.get('error')
                })

            if (i + 1) % batch_size == 0:
                logger.info(f"Indexed {i + 1}/{len(documents)} documents")

        logger.info(f"Batch indexing complete: {results['succeeded']}/{results['total']} succeeded")

        return results

    # ========================================================================
    # Search and Retrieval
    # ========================================================================

    def search(
        self,
        query: str,
        top_k: int = 10,
        mode: Optional[str] = None,
        filters: Optional[Dict] = None
    ) -> List[SearchResult]:
        """
        Search for relevant documents

        Args:
            query: Search query
            top_k: Number of results to return
            mode: Search mode (vector, hybrid, bm25) - uses config if not specified
            filters: Additional filters

        Returns:
            List of search results
        """
        start_time = time.time()

        mode = mode or self.config.search.mode

        try:
            # Get initial candidates
            if mode == "vector":
                candidates = self._vector_search(query, top_k * 5, filters)
            elif mode == "hybrid":
                candidates = self._hybrid_search(query, top_k * 5, filters)
            elif mode == "bm25":
                candidates = self._bm25_search(query, top_k * 5)
            else:
                raise ValueError(f"Unknown search mode: {mode}")

            # Rerank if enabled
            if self.config.search.use_reranking and self.reranker and candidates:
                rerank_start = time.time()

                rerank_docs = [
                    {
                        'id': c['id'],
                        'text': c['payload']['text'],
                        'score': c['score'],
                        'metadata': c['payload']
                    }
                    for c in candidates
                ]

                reranked = self.reranker.rerank(query, rerank_docs, top_k=top_k)

                self.stats['total_rerank_time'] += time.time() - rerank_start

                # Convert to SearchResult
                results = [
                    SearchResult(
                        doc_id=r.doc_id,
                        text=r.text,
                        score=r.rerank_score,
                        rank=i + 1,
                        source="reranked",
                        metadata=r.metadata,
                        vector_score=r.initial_score,
                        rerank_score=r.rerank_score
                    )
                    for i, r in enumerate(reranked)
                ]
            else:
                # No reranking, return top-k candidates
                results = [
                    SearchResult(
                        doc_id=c['id'],
                        text=c['payload']['text'],
                        score=c['score'],
                        rank=i + 1,
                        source=mode,
                        metadata=c['payload'],
                        vector_score=c['score'] if mode == "vector" else None,
                        bm25_score=c['score'] if mode == "bm25" else None
                    )
                    for i, c in enumerate(candidates[:top_k])
                ]

            # Update stats
            self.stats['queries_processed'] += 1
            self.stats['total_search_time'] += time.time() - start_time

            return results

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    def _vector_search(
        self,
        query: str,
        limit: int = 50,
        filters: Optional[Dict] = None
    ) -> List[Dict]:
        """Vector similarity search"""
        # Encode query
        query_emb = self.embedder.encode_query(query)

        # Search
        results = self.vector_store.search(
            query_vector=query_emb,
            limit=limit,
            score_threshold=self.config.search.vector_score_threshold,
            filter_conditions=filters
        )

        return results

    def _bm25_search(
        self,
        query: str,
        limit: int = 50
    ) -> List[Dict]:
        """BM25 keyword search"""
        if not self.bm25_index:
            logger.warning("BM25 index not available, falling back to vector search")
            return self._vector_search(query, limit)

        results = self.bm25_index.search(query, top_k=limit)

        return [
            {
                'id': doc_id,
                'score': score,
                'payload': {}  # Would need to fetch from store
            }
            for doc_id, score in results
        ]

    def _hybrid_search(
        self,
        query: str,
        limit: int = 50,
        filters: Optional[Dict] = None
    ) -> List[Dict]:
        """Hybrid search (vector + BM25)"""
        # Get vector results
        vector_results = self._vector_search(query, limit, filters)

        # Get BM25 results if available
        if self.bm25_index:
            bm25_results = self._bm25_search(query, limit)

            # Merge using reciprocal rank fusion
            from hybrid_search import reciprocal_rank_fusion

            merged = reciprocal_rank_fusion(
                vector_results,
                bm25_results,
                k=self.config.search.rrf_k
            )

            return merged[:limit]
        else:
            # Fall back to vector only
            logger.warning("BM25 not available, using vector search only")
            return vector_results

    # ========================================================================
    # Utilities
    # ========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        vector_stats = self.vector_store.get_stats()

        return {
            **self.stats,
            'vector_store': vector_stats,
            'config': {
                'embedding_model': self.config.embedding.model_name,
                'embedding_dim': self.config.embedding.dimension,
                'chunking_strategy': self.config.chunking.strategy,
                'search_mode': self.config.search.mode,
                'reranking_enabled': self.config.search.use_reranking
            }
        }

    def save_config(self, filepath: Path):
        """Save current configuration"""
        self.config.save(filepath)

    # ========================================================================
    # Class Methods
    # ========================================================================

    @classmethod
    def from_preset(cls, preset_name: str) -> 'RAGOrchestrator':
        """
        Create orchestrator from preset configuration

        Args:
            preset_name: Preset name (baseline, jina_standard, etc.)

        Returns:
            RAGOrchestrator instance
        """
        return cls(preset=preset_name)

    @classmethod
    def from_config_file(cls, config_path: Path) -> 'RAGOrchestrator':
        """
        Create orchestrator from configuration file

        Args:
            config_path: Path to JSON config file

        Returns:
            RAGOrchestrator instance
        """
        return cls(config_path=config_path)

    @classmethod
    def from_env(cls) -> 'RAGOrchestrator':
        """
        Create orchestrator from environment variables

        Returns:
            RAGOrchestrator instance
        """
        from rag_config import get_config_from_env
        config = get_config_from_env()
        return cls(config=config)


# ============================================================================
# Convenience Functions
# ============================================================================

def create_rag_system(
    preset: str = "production",
    **config_overrides
) -> RAGOrchestrator:
    """
    Convenience function to create RAG system

    Args:
        preset: Configuration preset
        **config_overrides: Override config values

    Returns:
        RAGOrchestrator instance

    Example:
        rag = create_rag_system("jina_standard")
        rag.index_document("Sample text")
        results = rag.search("query")
    """
    from rag_config import get_preset_config

    config = get_preset_config(preset)

    # Apply overrides
    for key, value in config_overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)

    return RAGOrchestrator(config=config)


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("RAG ORCHESTRATOR DEMO")
    print("="*80 + "\n")

    # Create RAG system with Jina v3 standard preset
    print("Creating RAG system with 'jina_standard' preset...\n")
    rag = RAGOrchestrator.from_preset("jina_standard")

    # Show configuration
    print("Configuration:")
    print(f"  Embedding: {rag.config.embedding.model_name} ({rag.config.embedding.dimension}D)")
    print(f"  Chunking: {rag.config.chunking.strategy}")
    print(f"  Search: {rag.config.search.mode}")
    print(f"  Reranking: {rag.config.reranker.model_name if rag.config.reranker else 'None'}")
    print("\n" + "="*80 + "\n")

    # Index sample documents
    print("Indexing sample documents...\n")

    sample_docs = [
        {
            'id': 'doc1',
            'text': 'VPN connection failed due to authentication timeout. Gateway experienced high CPU load.',
            'metadata': {'source': 'vpn_logs.txt', 'timestamp': '2024-01-15'}
        },
        {
            'id': 'doc2',
            'text': 'Malware analysis revealed suspicious process injection behavior. Memory forensics showed reflective DLL injection.',
            'metadata': {'source': 'malware_report.pdf', 'timestamp': '2024-01-16'}
        },
        {
            'id': 'doc3',
            'text': 'Network packet capture analysis using Wireshark. SMB traffic with NTLM authentication detected.',
            'metadata': {'source': 'pcap_analysis.txt', 'timestamp': '2024-01-17'}
        }
    ]

    result = rag.index_document_batch(sample_docs)
    print(f"Indexing result: {result['succeeded']}/{result['total']} documents indexed\n")

    print("="*80 + "\n")

    # Perform search
    print("Searching for 'VPN authentication error'...\n")

    results = rag.search("VPN authentication error", top_k=3)

    print(f"Found {len(results)} results:\n")

    for i, result in enumerate(results, 1):
        print(f"{i}. [Score: {result.score:.3f}] [Source: {result.source}]")
        print(f"   Doc ID: {result.doc_id}")
        print(f"   Text: {result.text[:100]}...")
        if result.rerank_score:
            print(f"   Rerank score: {result.rerank_score:.3f}")
        print()

    print("="*80 + "\n")

    # Show stats
    stats = rag.get_stats()
    print("System Statistics:")
    print(f"  Documents indexed: {stats['documents_indexed']}")
    print(f"  Queries processed: {stats['queries_processed']}")
    print(f"  Total points in vector store: {stats['vector_store'].get('points_count', 0)}")
    print()

    print("✓ RAG Orchestrator demo complete")
