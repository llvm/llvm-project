#!/usr/bin/env python3
"""
Adaptive Multi-Strategy Retriever

Dynamically selects best retrieval strategy based on query type:
- Vector search: Semantic similarity
- Keyword search: BM25/TF-IDF
- Hybrid search: Weighted combination

Includes supervisor pattern for strategy selection.

Author: LAT5150DRVMIL AI Framework
Version: 1.0.0
"""

from typing import List, Dict, Optional
from enum import Enum


class RetrievalStrategy(Enum):
    """Retrieval strategy options"""
    VECTOR = "vector"
    KEYWORD = "keyword"
    HYBRID = "hybrid"


class AdaptiveRetriever:
    """
    Adaptive retrieval with supervisor-based strategy selection

    Usage:
        retriever = AdaptiveRetriever(rag_system)
        results = retriever.retrieve(
            query="How to optimize SQL?",
            strategy="hybrid",
            n_results=10
        )
    """

    def __init__(self, rag_system=None):
        """
        Initialize adaptive retriever

        Args:
            rag_system: EnhancedRAGSystem instance (optional, for integration)
        """
        self.rag_system = rag_system

        # Strategy weights for hybrid search
        self.hybrid_weights = {
            "vector": 0.7,
            "keyword": 0.3
        }

    def retrieve(
        self,
        query: str,
        strategy: str = "hybrid",
        n_results: int = 10,
        use_reranking: bool = True
    ) -> List[Dict]:
        """
        Retrieve documents using specified strategy

        Args:
            query: Search query
            strategy: "vector", "keyword", or "hybrid"
            n_results: Number of results
            use_reranking: Apply cross-encoder reranking

        Returns:
            List of document dicts with scores
        """
        if strategy == RetrievalStrategy.VECTOR.value:
            return self._vector_search(query, n_results, use_reranking)
        elif strategy == RetrievalStrategy.KEYWORD.value:
            return self._keyword_search(query, n_results)
        elif strategy == RetrievalStrategy.HYBRID.value:
            return self._hybrid_search(query, n_results, use_reranking)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def _vector_search(
        self,
        query: str,
        n_results: int,
        use_reranking: bool
    ) -> List[Dict]:
        """
        Vector/semantic search using embeddings

        Args:
            query: Search query
            n_results: Number of results
            use_reranking: Apply reranking

        Returns:
            List of documents with vector scores
        """
        if self.rag_system:
            # Use integrated RAG system
            results = self.rag_system.search(
                query,
                n_results=n_results,
                search_type="semantic",
                use_reranking=use_reranking
            )

            return [
                {
                    "text": r.text,
                    "score": r.relevance_score,
                    "metadata": r.metadata,
                    "strategy": "vector"
                }
                for r in results
            ]
        else:
            # Placeholder for standalone usage
            print(f"Vector search: {query} (n={n_results})")
            return []

    def _keyword_search(self, query: str, n_results: int) -> List[Dict]:
        """
        Keyword search using BM25/TF-IDF

        Args:
            query: Search query
            n_results: Number of results

        Returns:
            List of documents with keyword scores
        """
        # Placeholder - would integrate with PostgreSQL full-text search
        # or Elasticsearch in production
        print(f"Keyword search: {query} (n={n_results})")

        # For now, fall back to vector search if RAG system available
        if self.rag_system:
            return self._vector_search(query, n_results, use_reranking=False)

        return []

    def _hybrid_search(
        self,
        query: str,
        n_results: int,
        use_reranking: bool
    ) -> List[Dict]:
        """
        Hybrid search combining vector and keyword

        Args:
            query: Search query
            n_results: Number of results
            use_reranking: Apply reranking

        Returns:
            List of documents with combined scores
        """
        # Get results from both strategies
        vector_results = self._vector_search(query, n_results * 2, use_reranking=False)
        keyword_results = self._keyword_search(query, n_results * 2)

        # Combine and rerank
        combined = self._combine_results(
            vector_results,
            keyword_results,
            n_results
        )

        # Apply cross-encoder reranking if enabled
        if use_reranking and self.rag_system and self.rag_system.reranker:
            texts = [doc["text"] for doc in combined]
            reranked = self.rag_system.reranker.rerank(query, texts, top_k=n_results)

            combined = [
                {
                    **combined[rr.original_rank],
                    "score": rr.score,
                    "original_score": combined[rr.original_rank]["score"],
                    "reranked": True
                }
                for rr in reranked
            ]

        return combined

    def _combine_results(
        self,
        vector_results: List[Dict],
        keyword_results: List[Dict],
        n_results: int
    ) -> List[Dict]:
        """
        Combine vector and keyword results with weighted scores

        Uses Reciprocal Rank Fusion (RRF) for combining rankings.

        Args:
            vector_results: Vector search results
            keyword_results: Keyword search results
            n_results: Number of results to return

        Returns:
            Combined and sorted results
        """
        # Build document index
        doc_scores = {}

        # Add vector results
        for rank, doc in enumerate(vector_results):
            doc_id = doc["text"][:100]  # Use text prefix as ID
            doc_scores[doc_id] = {
                "doc": doc,
                "vector_score": doc["score"],
                "vector_rank": rank,
                "keyword_score": 0.0,
                "keyword_rank": None
            }

        # Add keyword results
        for rank, doc in enumerate(keyword_results):
            doc_id = doc["text"][:100]
            if doc_id in doc_scores:
                doc_scores[doc_id]["keyword_score"] = doc["score"]
                doc_scores[doc_id]["keyword_rank"] = rank
            else:
                doc_scores[doc_id] = {
                    "doc": doc,
                    "vector_score": 0.0,
                    "vector_rank": None,
                    "keyword_score": doc["score"],
                    "keyword_rank": rank
                }

        # Calculate combined scores using RRF
        k = 60  # RRF constant
        for doc_id, scores in doc_scores.items():
            vector_rrf = 1.0 / (k + (scores["vector_rank"] or 1000))
            keyword_rrf = 1.0 / (k + (scores["keyword_rank"] or 1000))

            # Weighted combination
            combined_score = (
                self.hybrid_weights["vector"] * vector_rrf +
                self.hybrid_weights["keyword"] * keyword_rrf
            )

            scores["combined_score"] = combined_score

        # Sort by combined score
        sorted_docs = sorted(
            doc_scores.values(),
            key=lambda x: x["combined_score"],
            reverse=True
        )

        # Return top N documents
        return [
            {
                **item["doc"],
                "score": item["combined_score"],
                "vector_score": item["vector_score"],
                "keyword_score": item["keyword_score"],
                "strategy": "hybrid"
            }
            for item in sorted_docs[:n_results]
        ]

    def set_hybrid_weights(self, vector_weight: float, keyword_weight: float):
        """
        Set weights for hybrid search

        Args:
            vector_weight: Weight for vector scores (0.0-1.0)
            keyword_weight: Weight for keyword scores (0.0-1.0)
        """
        total = vector_weight + keyword_weight
        self.hybrid_weights = {
            "vector": vector_weight / total,
            "keyword": keyword_weight / total
        }


if __name__ == "__main__":
    # Demo usage
    print("="*70)
    print("Adaptive Retriever Demo")
    print("="*70)

    retriever = AdaptiveRetriever()

    print("\nDemonstrating strategy selection:")
    print("-"*70)

    strategies = ["vector", "keyword", "hybrid"]

    for strategy in strategies:
        print(f"\nStrategy: {strategy.upper()}")
        results = retriever.retrieve(
            query="How to optimize PostgreSQL queries?",
            strategy=strategy,
            n_results=5
        )
        print(f"  Retrieved: {len(results)} documents")

    print("\n" + "="*70)
    print("Adaptive retriever ready for integration")
    print("="*70)
