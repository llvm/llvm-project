#!/usr/bin/env python3
"""
Optimized RAG System - Phase 1 Integration

Combines all Phase 1 optimizations:
1. HNSW parameter tuning (+3-5% accuracy)
2. Query expansion (+3-5% recall)
3. Cross-encoder reranking (+5-10% precision)

Expected total improvement: +11-20% over baseline
Target: Hit@3: 88-90% (from ~84%)

Usage:
    rag = OptimizedRAG()
    results = rag.search("VPN connection error", limit=10)

Components:
- VectorRAGSystem: Base system with optimized HNSW
- QueryEnhancer: Query expansion with synonyms/LLM
- CrossEncoderReranker: Two-stage retrieval
"""

import logging
from typing import List, Dict, Optional
from pathlib import Path
from dataclasses import dataclass
import time

# Import base components
from vector_rag_system import VectorRAGSystem, SearchResult as BaseSearchResult
from query_enhancer import QueryEnhancer, HybridQueryProcessor
from reranker import CrossEncoderReranker, TwoStageRAG, RerankResult

logger = logging.getLogger(__name__)


@dataclass
class OptimizedSearchResult:
    """Enhanced search result with optimization metadata"""
    doc_id: str
    text: str
    filename: str
    filepath: str
    doc_type: str
    timestamp: str

    # Scores
    vector_score: float      # Initial vector similarity
    rerank_score: float      # Cross-encoder rerank score
    final_score: float       # Combined score

    # Optimization metadata
    rank_change: int         # Position change from reranking
    query_expanded: bool     # Was query expanded?
    synonyms_used: List[str] # Synonyms added to query

    # Document metadata
    metadata: Dict


class OptimizedRAG:
    """
    Production-ready optimized RAG system with Phase 1 enhancements

    Features:
    - Optimized HNSW indexing (m=32, ef_construct=200, hnsw_ef=128)
    - Query expansion (synonym + LLM)
    - Two-stage retrieval (vector + cross-encoder)
    - Performance tracking
    - Fallback modes for robustness

    Expected performance:
    - Hit@3: 88-90% (from ~84%)
    - MRR: 0.75-0.78 (from ~0.72)
    - Precision@10: 75-80%
    - Latency: <150ms per query
    """

    def __init__(
        self,
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        collection_name: str = "lat5150_knowledge_base",
        embedding_model: str = "BAAI/bge-base-en-v1.5",

        # Optimization settings
        enable_query_expansion: bool = True,
        enable_reranking: bool = True,
        enable_llm_rewriting: bool = False,  # Optional, requires Ollama

        # Advanced settings
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        use_gpu: bool = True,
    ):
        """
        Initialize optimized RAG system

        Args:
            qdrant_host: Qdrant server host
            qdrant_port: Qdrant server port
            collection_name: Collection name
            embedding_model: Sentence transformer model
            enable_query_expansion: Enable query enhancement
            enable_reranking: Enable cross-encoder reranking
            enable_llm_rewriting: Enable LLM-based query rewriting
            reranker_model: Cross-encoder model
            use_gpu: Use GPU if available
        """
        logger.info("Initializing Optimized RAG System (Phase 1)...")

        # Initialize base vector RAG (with optimized HNSW)
        logger.info("  [1/3] Loading vector RAG system...")
        self.vector_rag = VectorRAGSystem(
            qdrant_host=qdrant_host,
            qdrant_port=qdrant_port,
            collection_name=collection_name,
            embedding_model=embedding_model,
            use_gpu=use_gpu
        )

        # Initialize query enhancer
        self.enable_query_expansion = enable_query_expansion
        if enable_query_expansion:
            logger.info("  [2/3] Loading query enhancer...")
            self.query_processor = HybridQueryProcessor(
                use_llm=enable_llm_rewriting,
                llm_endpoint="http://localhost:11434"
            )
            logger.info("      ✓ Query expansion enabled")
        else:
            self.query_processor = None
            logger.info("  [2/3] Query expansion disabled")

        # Initialize reranker
        self.enable_reranking = enable_reranking
        if enable_reranking:
            logger.info("  [3/3] Loading cross-encoder reranker...")
            try:
                self.two_stage_rag = TwoStageRAG(
                    vector_rag=self.vector_rag,
                    reranker_type="cross-encoder",
                    reranker_model=reranker_model
                )
                logger.info("      ✓ Cross-encoder reranking enabled")
            except Exception as e:
                logger.warning(f"Failed to load reranker: {e}")
                logger.warning("      ! Reranking disabled, using vector search only")
                self.enable_reranking = False
                self.two_stage_rag = None
        else:
            self.two_stage_rag = None
            logger.info("  [3/3] Reranking disabled")

        # Performance tracking
        self.stats = {
            'total_queries': 0,
            'queries_with_expansion': 0,
            'queries_with_reranking': 0,
            'avg_query_time': 0.0,
            'avg_vector_time': 0.0,
            'avg_rerank_time': 0.0,
        }

        logger.info("✓ Optimized RAG System initialized")
        logger.info(f"  Optimizations:")
        logger.info(f"    • HNSW tuning: ✓ (m=32, ef_construct=200, hnsw_ef=128)")
        logger.info(f"    • Query expansion: {'✓' if enable_query_expansion else '✗'}")
        logger.info(f"    • Cross-encoder: {'✓' if enable_reranking else '✗'}")
        logger.info(f"  Expected improvement: +11-20% over baseline")

    def search(
        self,
        query: str,
        limit: int = 10,
        score_threshold: float = 0.0,
        filters: Optional[Dict] = None,

        # Advanced options
        expand_query: Optional[bool] = None,  # Override default
        use_reranking: Optional[bool] = None,  # Override default
        candidate_multiplier: int = 10,  # How many candidates for reranking
    ) -> List[OptimizedSearchResult]:
        """
        Search with all optimizations enabled

        Args:
            query: Search query
            limit: Number of results to return
            score_threshold: Minimum similarity score
            filters: Document filters
            expand_query: Override query expansion setting
            use_reranking: Override reranking setting
            candidate_multiplier: Candidate multiplier for reranking (default: 10x)

        Returns:
            List of optimized search results
        """
        start_time = time.time()
        self.stats['total_queries'] += 1

        # Determine which optimizations to use
        do_expansion = expand_query if expand_query is not None else self.enable_query_expansion
        do_reranking = use_reranking if use_reranking is not None else self.enable_reranking

        # Phase 1: Query expansion
        expanded_query = query
        synonyms_added = []

        if do_expansion and self.query_processor:
            try:
                enhanced = self.query_processor.process(query)
                expanded_query = enhanced.expanded
                synonyms_added = enhanced.synonyms_added
                self.stats['queries_with_expansion'] += 1
                logger.debug(f"Query expanded: '{query}' → '{expanded_query}'")
            except Exception as e:
                logger.warning(f"Query expansion failed: {e}, using original query")
                expanded_query = query

        # Phase 2: Retrieval + Reranking
        if do_reranking and self.two_stage_rag:
            # Two-stage retrieval
            candidate_k = limit * candidate_multiplier
            vector_start = time.time()

            reranked_results = self.two_stage_rag.search(
                query=expanded_query,
                final_k=limit,
                candidate_k=candidate_k,
                score_threshold=score_threshold,
                filters=filters
            )

            vector_time = time.time() - vector_start
            self.stats['queries_with_reranking'] += 1

            # Convert to OptimizedSearchResult
            results = []
            for result in reranked_results:
                opt_result = OptimizedSearchResult(
                    doc_id=result.doc_id,
                    text=result.text,
                    filename=result.metadata.get('filename', ''),
                    filepath=result.metadata.get('filepath', ''),
                    doc_type=result.metadata.get('type', ''),
                    timestamp=result.metadata.get('timestamp', ''),
                    vector_score=result.initial_score,
                    rerank_score=result.rerank_score,
                    final_score=result.rerank_score,  # Use rerank score as final
                    rank_change=result.rank_change,
                    query_expanded=do_expansion,
                    synonyms_used=synonyms_added,
                    metadata=result.metadata
                )
                results.append(opt_result)

        else:
            # Single-stage vector search only
            vector_start = time.time()
            base_results = self.vector_rag.search(
                query=expanded_query,
                limit=limit,
                score_threshold=score_threshold,
                filters=filters
            )
            vector_time = time.time() - vector_start

            # Convert to OptimizedSearchResult
            results = []
            for result in base_results:
                opt_result = OptimizedSearchResult(
                    doc_id=result.document.id,
                    text=result.document.text,
                    filename=result.document.filename,
                    filepath=result.document.filepath,
                    doc_type=result.document.doc_type,
                    timestamp=result.document.timestamp.isoformat(),
                    vector_score=result.score,
                    rerank_score=result.score,  # No reranking
                    final_score=result.score,
                    rank_change=0,
                    query_expanded=do_expansion,
                    synonyms_used=synonyms_added,
                    metadata=result.document.metadata
                )
                results.append(opt_result)

        # Update stats
        total_time = time.time() - start_time
        self.stats['avg_query_time'] = (
            (self.stats['avg_query_time'] * (self.stats['total_queries'] - 1) + total_time) /
            self.stats['total_queries']
        )

        logger.info(
            f"Search completed in {total_time*1000:.1f}ms "
            f"({len(results)} results, expansion={'✓' if do_expansion else '✗'}, "
            f"reranking={'✓' if do_reranking else '✗'})"
        )

        return results

    def ingest_document(self, filepath: Path, **kwargs) -> Dict:
        """Ingest document into system"""
        return self.vector_rag.ingest_document(filepath, **kwargs)

    def ingest_chat_message(self, **kwargs) -> Dict:
        """Ingest chat message"""
        return self.vector_rag.ingest_chat_message(**kwargs)

    def get_stats(self) -> Dict:
        """Get system statistics"""
        base_stats = self.vector_rag.get_stats()

        return {
            **base_stats,
            'optimizations': {
                'hnsw_tuning': True,
                'query_expansion': self.enable_query_expansion,
                'cross_encoder_reranking': self.enable_reranking,
            },
            'performance': self.stats,
            'expected_improvement': '+11-20% over baseline',
            'target_metrics': {
                'hit_at_3': '88-90%',
                'mrr': '0.75-0.78',
                'precision_at_10': '75-80%',
            }
        }

    def get_performance_summary(self) -> str:
        """Get human-readable performance summary"""
        stats = self.stats

        summary = f"""
=== Optimized RAG Performance Summary ===

Total Queries: {stats['total_queries']}

Optimizations Used:
  • Query Expansion:    {stats['queries_with_expansion']} queries ({stats['queries_with_expansion']/max(stats['total_queries'],1)*100:.1f}%)
  • Cross-Encoder:      {stats['queries_with_reranking']} queries ({stats['queries_with_reranking']/max(stats['total_queries'],1)*100:.1f}%)

Performance:
  • Avg Query Time:     {stats['avg_query_time']*1000:.1f}ms
  • Target:             <150ms ({'✓' if stats['avg_query_time']*1000 < 150 else '✗'})

Phase 1 Optimizations:
  ✓ HNSW parameter tuning (m=32, ef_construct=200, hnsw_ef=128)
  {'✓' if self.enable_query_expansion else '✗'} Query expansion (synonym + LLM)
  {'✓' if self.enable_reranking else '✗'} Cross-encoder reranking (two-stage)

Expected Gains:
  • Hit@3:  +4-8% → 88-90%
  • MRR:    +0.03-0.06 → 0.75-0.78
  • P@10:   +5-10% → 75-80%
"""

        return summary


# Example usage and testing
if __name__ == "__main__":
    import sys

    print("=== Optimized RAG System Test ===\n")

    # Initialize system
    print("Initializing optimized RAG...")
    try:
        rag = OptimizedRAG(
            enable_query_expansion=True,
            enable_reranking=True,
            enable_llm_rewriting=False  # Disable for quick test
        )
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        sys.exit(1)

    print("\n" + "="*60 + "\n")

    # Get stats
    print("System Statistics:")
    stats = rag.get_stats()
    print(f"  Collection:       {stats['collection']}")
    print(f"  Total Documents:  {stats['total_documents']}")
    print(f"  Embedding Model:  {stats['embedding_model']}")
    print(f"  Vector Dimension: {stats['vector_dimension']}")
    print(f"\nOptimizations:")
    for key, value in stats['optimizations'].items():
        print(f"  • {key}: {'✓' if value else '✗'}")

    print(f"\nExpected Improvement: {stats['expected_improvement']}")

    # Test search if documents exist
    if stats['total_documents'] > 0:
        print("\n" + "="*60 + "\n")
        print("Test Search:\n")

        test_query = "error message"
        print(f"Query: '{test_query}'\n")

        results = rag.search(test_query, limit=5)

        if results:
            print(f"Found {len(results)} results:\n")
            for i, result in enumerate(results, 1):
                print(f"{i}. [Score: {result.final_score:.3f}]")
                print(f"   File: {result.filename}")
                print(f"   Type: {result.doc_type}")
                if result.rank_change != 0:
                    change = f"+{result.rank_change}" if result.rank_change > 0 else str(result.rank_change)
                    print(f"   Rank change: {change}")
                if result.query_expanded and result.synonyms_used:
                    print(f"   Synonyms: {', '.join(result.synonyms_used[:3])}")
                print(f"   Text: {result.text[:100]}...")
                print()
        else:
            print("No results found")

        # Performance summary
        print("="*60)
        print(rag.get_performance_summary())

    else:
        print("\n⚠️  No documents in collection. Ingest some documents first.")
        print("\nTo ingest documents:")
        print("  from pathlib import Path")
        print("  rag.ingest_document(Path('path/to/file.txt'))")

    print("\n✓ Test complete")
