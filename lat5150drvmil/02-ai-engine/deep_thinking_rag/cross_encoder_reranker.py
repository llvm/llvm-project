#!/usr/bin/env python3
"""
Cross-Encoder Reranker for RAG Precision

Uses cross-encoder models for high-precision reranking of search results.
Bi-encoders (current RAG) are fast but less accurate. Cross-encoders encode
query+document together for better semantic understanding.

Pipeline:
1. Bi-encoder retrieves top 50-100 documents (fast, high recall)
2. Cross-encoder reranks to top 5-10 (slow, high precision)
3. Send top results to LLM (best quality)

Expected improvement: +10-30% answer quality

Author: LAT5150DRVMIL AI Framework
Version: 1.0.0
"""

import warnings
warnings.filterwarnings('ignore')

from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import numpy as np

# Check for dependencies
try:
    from sentence_transformers import CrossEncoder
    CROSSENCODER_AVAILABLE = True
except ImportError:
    CROSSENCODER_AVAILABLE = False
    print("WARNING: sentence-transformers not installed. Cross-encoder reranking unavailable.")
    print("Install with: pip install sentence-transformers")


@dataclass
class RerankedResult:
    """Reranked search result with cross-encoder score"""
    text: str
    score: float
    original_rank: int
    reranked_rank: int
    metadata: Optional[Dict] = None


class CrossEncoderReranker:
    """
    High-precision reranking using cross-encoder models

    Models available:
    - cross-encoder/ms-marco-MiniLM-L-6-v2: Fast (90MB), good quality
    - cross-encoder/ms-marco-electra-base: Better (440MB), best quality
    - cross-encoder/qnli-electra-base: Question answering optimized

    Usage:
        reranker = CrossEncoderReranker()
        results = reranker.rerank(query, documents, top_k=10)
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: Optional[str] = None
    ):
        """
        Initialize cross-encoder reranker

        Args:
            model_name: HuggingFace cross-encoder model
            device: Device to run on ('cpu', 'cuda', 'npu'). Auto-detect if None
        """
        if not CROSSENCODER_AVAILABLE:
            raise ImportError(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )

        self.model_name = model_name

        # Auto-detect device
        if device is None:
            import torch
            if torch.cuda.is_available():
                device = 'cuda'
            else:
                device = 'cpu'

        self.device = device

        print(f"Loading cross-encoder: {model_name} on {device}")
        self.model = CrossEncoder(model_name, device=device)
        print("✓ Cross-encoder loaded successfully")

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: int = 10,
        return_scores: bool = True,
        batch_size: int = 32
    ) -> List[RerankedResult]:
        """
        Rerank documents using cross-encoder

        Args:
            query: Search query
            documents: List of document texts to rerank
            top_k: Number of top results to return
            return_scores: Include relevance scores in results
            batch_size: Batch size for inference (larger = faster but more memory)

        Returns:
            List of RerankedResult objects sorted by relevance
        """
        if not documents:
            return []

        # Limit top_k to available documents
        top_k = min(top_k, len(documents))

        # Create query-document pairs
        pairs = [[query, doc] for doc in documents]

        # Score with cross-encoder (encodes query+doc together)
        print(f"Reranking {len(documents)} documents with cross-encoder...")
        scores = self.model.predict(
            pairs,
            batch_size=batch_size,
            show_progress_bar=len(documents) > 100
        )

        # Create results with original ranks
        results = []
        for idx, (doc, score) in enumerate(zip(documents, scores)):
            results.append({
                'text': doc,
                'score': float(score),
                'original_rank': idx
            })

        # Sort by score (descending)
        results.sort(key=lambda x: x['score'], reverse=True)

        # Add reranked ranks and convert to dataclass
        reranked_results = []
        for rank, result in enumerate(results[:top_k]):
            reranked_results.append(RerankedResult(
                text=result['text'],
                score=result['score'],
                original_rank=result['original_rank'],
                reranked_rank=rank,
                metadata=result.get('metadata')
            ))

        print(f"✓ Reranked to top {top_k} results")
        return reranked_results

    def rerank_with_metadata(
        self,
        query: str,
        documents: List[Dict],
        text_field: str = 'text',
        top_k: int = 10,
        batch_size: int = 32
    ) -> List[Dict]:
        """
        Rerank documents with metadata preservation

        Args:
            query: Search query
            documents: List of dicts with 'text' field and metadata
            text_field: Field name containing document text
            top_k: Number of top results to return
            batch_size: Batch size for inference

        Returns:
            List of original document dicts sorted by relevance with scores added
        """
        if not documents:
            return []

        # Extract texts
        texts = [doc[text_field] for doc in documents]

        # Score with cross-encoder
        pairs = [[query, text] for text in texts]
        scores = self.model.predict(
            pairs,
            batch_size=batch_size,
            show_progress_bar=len(documents) > 100
        )

        # Add scores to documents
        scored_docs = []
        for idx, (doc, score) in enumerate(zip(documents, scores)):
            doc_copy = doc.copy()
            doc_copy['reranker_score'] = float(score)
            doc_copy['original_rank'] = idx
            scored_docs.append(doc_copy)

        # Sort and return top_k
        scored_docs.sort(key=lambda x: x['reranker_score'], reverse=True)

        # Add reranked rank
        for rank, doc in enumerate(scored_docs[:top_k]):
            doc['reranked_rank'] = rank

        return scored_docs[:top_k]

    def compare_rankings(
        self,
        query: str,
        documents: List[str],
        top_k: int = 10
    ) -> Dict:
        """
        Compare original vs reranked ordering for analysis

        Args:
            query: Search query
            documents: List of document texts
            top_k: Number of top results to analyze

        Returns:
            Dict with comparison statistics
        """
        results = self.rerank(query, documents, top_k=top_k)

        # Calculate rank changes
        rank_changes = []
        for result in results:
            rank_change = result.original_rank - result.reranked_rank
            rank_changes.append(rank_change)

        # Statistics
        avg_rank_change = np.mean([abs(rc) for rc in rank_changes])
        max_rank_jump = max(rank_changes)
        max_rank_drop = min(rank_changes)

        # How many of original top-k stayed in top-k?
        original_top_k = set(range(top_k))
        reranked_top_k = {r.original_rank for r in results}
        overlap = len(original_top_k & reranked_top_k)

        return {
            'total_documents': len(documents),
            'top_k': top_k,
            'average_rank_change': avg_rank_change,
            'max_rank_jump': max_rank_jump,
            'max_rank_drop': max_rank_drop,
            'top_k_overlap': overlap,
            'top_k_overlap_percent': (overlap / top_k) * 100,
            'results': results
        }


def demo():
    """Demonstration of cross-encoder reranking"""
    print("=" * 60)
    print("Cross-Encoder Reranking Demo")
    print("=" * 60)

    # Sample query and documents
    query = "How do I optimize database queries in PostgreSQL?"

    documents = [
        "PostgreSQL supports various index types including B-tree, Hash, GiST, and GIN for query optimization.",
        "Python is a popular programming language for data science and machine learning applications.",
        "Use EXPLAIN ANALYZE to understand query execution plans and identify slow operations in PostgreSQL.",
        "JavaScript frameworks like React and Vue are commonly used for frontend development.",
        "Database normalization is the process of organizing data to reduce redundancy.",
        "The VACUUM command in PostgreSQL helps reclaim storage and update statistics for better query planning.",
        "Docker containers provide isolated environments for running applications consistently.",
        "Creating appropriate indexes on frequently queried columns can dramatically improve PostgreSQL performance.",
        "Git is a distributed version control system used for tracking changes in source code.",
        "Connection pooling with pgBouncer reduces connection overhead in PostgreSQL applications."
    ]

    print(f"\nQuery: {query}\n")
    print(f"Documents to rerank: {len(documents)}\n")

    # Create reranker
    reranker = CrossEncoderReranker()

    # Rerank
    results = reranker.rerank(query, documents, top_k=5)

    print("\n" + "=" * 60)
    print("Top 5 Results After Reranking")
    print("=" * 60)

    for result in results:
        print(f"\n[Rank {result.reranked_rank + 1}] Score: {result.score:.4f} (was rank {result.original_rank + 1})")
        print(f"  {result.text[:100]}...")

    # Compare rankings
    print("\n" + "=" * 60)
    print("Ranking Comparison")
    print("=" * 60)

    comparison = reranker.compare_rankings(query, documents, top_k=5)
    print(f"\nAverage rank change: {comparison['average_rank_change']:.2f} positions")
    print(f"Largest improvement: +{comparison['max_rank_jump']} positions")
    print(f"Largest drop: {comparison['max_rank_drop']} positions")
    print(f"Top-5 overlap: {comparison['top_k_overlap']}/5 documents ({comparison['top_k_overlap_percent']:.0f}%)")


if __name__ == "__main__":
    demo()
