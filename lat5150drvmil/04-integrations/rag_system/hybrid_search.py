#!/usr/bin/env python3
"""
Hybrid Search Module - Dense + Sparse Retrieval

Combines:
1. Dense vector search (semantic) via Qdrant
2. Sparse BM25 search (keyword) via rank-bm25
3. Reciprocal Rank Fusion (RRF) for score combination

Expected gain: +3-5% on keyword queries, +5-8% overall
Research: Pinecone (2024), Elastic (2023), Weaviate (2024)

Why Hybrid Search?
- Dense vectors: Great for semantic similarity
- BM25: Great for exact keyword matches
- Combined: Best of both worlds

Example:
Query: "error code 404"
- Dense: Finds "page not found", "HTTP errors" (semantic)
- BM25: Finds exact "404" mentions (keyword)
- Hybrid: Ranks both appropriately
"""

import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import json
import pickle

logger = logging.getLogger(__name__)

# BM25 implementation
try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    logger.warning("rank-bm25 not available. Install: pip install rank-bm25")

# For tokenization
import re


@dataclass
class HybridSearchResult:
    """Result from hybrid search"""
    doc_id: str
    text: str
    filename: str
    filepath: str
    doc_type: str
    timestamp: str

    # Scores
    dense_score: float  # Vector similarity score
    sparse_score: float  # BM25 score
    hybrid_score: float  # Fused score

    # Ranking info
    dense_rank: int  # Rank in dense results
    sparse_rank: int  # Rank in sparse results

    # Document metadata
    metadata: Dict


class BM25Index:
    """
    BM25 sparse index for keyword-based retrieval

    Uses rank-bm25 (BM25Okapi variant) for efficient keyword matching

    Parameters:
    - k1: Term frequency saturation (default 1.5)
    - b: Length normalization (default 0.75)
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Initialize BM25 index

        Args:
            k1: Term frequency saturation parameter
            b: Length normalization parameter
        """
        if not BM25_AVAILABLE:
            raise ImportError("rank-bm25 required. Install: pip install rank-bm25")

        self.k1 = k1
        self.b = b
        self.bm25 = None
        self.documents = []  # Store document metadata
        self.tokenized_corpus = []  # Store tokenized documents

        logger.info(f"BM25 index initialized (k1={k1}, b={b})")

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text for BM25

        Strategy:
        - Lowercase for case-insensitive matching
        - Keep alphanumeric and technical characters
        - Preserve technical terms (IPs, paths, codes)
        """
        # Lowercase
        text = text.lower()

        # Tokenize: keep alphanumeric, hyphens, underscores, dots, slashes
        tokens = re.findall(r'[\w\-\.\/]+', text)

        # Filter very short tokens (but keep single-letter codes like 'C' in "C drive")
        tokens = [t for t in tokens if len(t) >= 1]

        return tokens

    def add_documents(self, documents: List[Dict]):
        """
        Add documents to BM25 index

        Args:
            documents: List of dicts with 'id', 'text', and metadata
        """
        self.documents = documents

        # Tokenize all documents
        self.tokenized_corpus = [
            self._tokenize(doc.get('text', ''))
            for doc in documents
        ]

        # Build BM25 index
        self.bm25 = BM25Okapi(self.tokenized_corpus, k1=self.k1, b=self.b)

        logger.info(f"BM25 index built: {len(documents)} documents")

    def search(self, query: str, top_k: int = 100) -> List[Tuple[str, float]]:
        """
        Search BM25 index

        Args:
            query: Search query
            top_k: Number of results

        Returns:
            List of (doc_id, score) tuples
        """
        if self.bm25 is None:
            logger.warning("BM25 index not built yet")
            return []

        # Tokenize query
        query_tokens = self._tokenize(query)

        # Get BM25 scores
        scores = self.bm25.get_scores(query_tokens)

        # Get top-k indices
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        # Return (doc_id, score) pairs
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only return docs with non-zero scores
                doc_id = self.documents[idx].get('id', '')
                results.append((doc_id, float(scores[idx])))

        return results

    def save(self, filepath: Path):
        """Save BM25 index to disk"""
        data = {
            'k1': self.k1,
            'b': self.b,
            'documents': self.documents,
            'tokenized_corpus': self.tokenized_corpus,
        }

        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

        logger.info(f"BM25 index saved to {filepath}")

    def load(self, filepath: Path):
        """Load BM25 index from disk"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        self.k1 = data['k1']
        self.b = data['b']
        self.documents = data['documents']
        self.tokenized_corpus = data['tokenized_corpus']

        # Rebuild BM25 index
        self.bm25 = BM25Okapi(self.tokenized_corpus, k1=self.k1, b=self.b)

        logger.info(f"BM25 index loaded from {filepath}: {len(self.documents)} documents")


class ReciprocalRankFusion:
    """
    Reciprocal Rank Fusion (RRF) for combining multiple ranked lists

    Formula: RRF(d) = Σ 1 / (k + rank(d))

    Where:
    - d: document
    - k: constant (default 60, from research)
    - rank(d): rank of document d in a ranked list

    Research: Cormack et al. (2009), Google Scholar uses RRF
    Benefits:
    - No score normalization needed
    - Robust to different score scales
    - Proven effective in practice
    """

    def __init__(self, k: int = 60):
        """
        Initialize RRF

        Args:
            k: Constant for RRF formula (default 60 from research)
        """
        self.k = k

    def fuse(
        self,
        dense_results: List[Tuple[str, float]],
        sparse_results: List[Tuple[str, float]],
        dense_weight: float = 0.5,
        sparse_weight: float = 0.5
    ) -> List[Tuple[str, float]]:
        """
        Fuse dense and sparse results using weighted RRF

        Args:
            dense_results: List of (doc_id, score) from dense search
            sparse_results: List of (doc_id, score) from sparse search
            dense_weight: Weight for dense results (0-1)
            sparse_weight: Weight for sparse results (0-1)

        Returns:
            Fused list of (doc_id, fused_score)
        """
        # Build rank maps
        dense_ranks = {doc_id: rank for rank, (doc_id, _) in enumerate(dense_results, 1)}
        sparse_ranks = {doc_id: rank for rank, (doc_id, _) in enumerate(sparse_results, 1)}

        # Get all unique doc IDs
        all_doc_ids = set(dense_ranks.keys()) | set(sparse_ranks.keys())

        # Calculate RRF scores
        fused_scores = {}
        for doc_id in all_doc_ids:
            rrf_score = 0.0

            # Add weighted RRF from dense
            if doc_id in dense_ranks:
                rrf_score += dense_weight * (1.0 / (self.k + dense_ranks[doc_id]))

            # Add weighted RRF from sparse
            if doc_id in sparse_ranks:
                rrf_score += sparse_weight * (1.0 / (self.k + sparse_ranks[doc_id]))

            fused_scores[doc_id] = rrf_score

        # Sort by fused score
        fused_results = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)

        return fused_results


class HybridSearchSystem:
    """
    Hybrid search combining dense (Qdrant) and sparse (BM25) retrieval

    Architecture:
    1. Query → Dense search (Qdrant) → top-100
    2. Query → Sparse search (BM25) → top-100
    3. Fuse with RRF → top-K results

    Expected improvement: +3-5% on keyword queries, +5-8% overall
    """

    def __init__(
        self,
        vector_rag,  # VectorRAGSystem instance
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
        rrf_k: int = 60,
        dense_weight: float = 0.7,  # Favor dense slightly (better for semantic)
        sparse_weight: float = 0.3,  # BM25 for keyword boosting
    ):
        """
        Initialize hybrid search system

        Args:
            vector_rag: VectorRAGSystem instance
            bm25_k1: BM25 term frequency saturation
            bm25_b: BM25 length normalization
            rrf_k: RRF constant
            dense_weight: Weight for dense retrieval (0-1)
            sparse_weight: Weight for sparse retrieval (0-1)
        """
        self.vector_rag = vector_rag
        self.bm25_index = BM25Index(k1=bm25_k1, b=bm25_b)
        self.rrf = ReciprocalRankFusion(k=rrf_k)
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight

        # Build initial BM25 index from Qdrant
        self._sync_bm25_index()

        logger.info("✓ Hybrid search system initialized")
        logger.info(f"  Dense weight: {dense_weight}")
        logger.info(f"  Sparse weight: {sparse_weight}")
        logger.info(f"  Expected gain: +5-8% over dense-only")

    def _sync_bm25_index(self):
        """Sync BM25 index with Qdrant collection"""
        logger.info("Syncing BM25 index with Qdrant...")

        # Get all documents from Qdrant
        try:
            # Scroll through entire collection
            documents = []
            offset = None

            while True:
                result = self.vector_rag.client.scroll(
                    collection_name=self.vector_rag.collection_name,
                    limit=100,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False
                )

                points, next_offset = result

                if not points:
                    break

                for point in points:
                    documents.append({
                        'id': point.id,
                        'text': point.payload.get('text', ''),
                        'filename': point.payload.get('filename', ''),
                        'filepath': point.payload.get('filepath', ''),
                        'type': point.payload.get('type', ''),
                        'timestamp': point.payload.get('timestamp', ''),
                        'metadata': point.payload
                    })

                if next_offset is None:
                    break

                offset = next_offset

            # Build BM25 index
            if documents:
                self.bm25_index.add_documents(documents)
                logger.info(f"✓ BM25 index synced: {len(documents)} documents")
            else:
                logger.warning("No documents found in Qdrant collection")

        except Exception as e:
            logger.error(f"Failed to sync BM25 index: {e}")

    def search(
        self,
        query: str,
        limit: int = 10,
        candidate_k: int = 100,
        dense_weight: Optional[float] = None,
        sparse_weight: Optional[float] = None,
        filters: Optional[Dict] = None
    ) -> List[HybridSearchResult]:
        """
        Hybrid search combining dense and sparse retrieval

        Args:
            query: Search query
            limit: Number of final results
            candidate_k: Number of candidates from each method
            dense_weight: Override dense weight
            sparse_weight: Override sparse weight
            filters: Document filters (applied to dense search only)

        Returns:
            List of hybrid search results
        """
        # Use provided weights or defaults
        dense_w = dense_weight if dense_weight is not None else self.dense_weight
        sparse_w = sparse_weight if sparse_weight is not None else self.sparse_weight

        # Stage 1: Dense search with Qdrant
        dense_results_raw = self.vector_rag.search(
            query=query,
            limit=candidate_k,
            score_threshold=0.0,
            filters=filters
        )

        dense_results = [
            (r.document.id, r.score)
            for r in dense_results_raw
        ]

        # Stage 2: Sparse search with BM25
        sparse_results = self.bm25_index.search(query, top_k=candidate_k)

        # Stage 3: Fuse with RRF
        fused_results = self.rrf.fuse(
            dense_results=dense_results,
            sparse_results=sparse_results,
            dense_weight=dense_w,
            sparse_weight=sparse_w
        )

        # Get top-K fused results
        top_fused = fused_results[:limit]

        # Build result objects
        results = []

        # Create lookup maps
        dense_map = {doc_id: (rank, score) for rank, (doc_id, score) in enumerate(dense_results, 1)}
        sparse_map = {doc_id: (rank, score) for rank, (doc_id, score) in enumerate(sparse_results, 1)}
        doc_map = {r.document.id: r.document for r in dense_results_raw}

        # Also include docs from BM25 that aren't in dense results
        for doc in self.bm25_index.documents:
            doc_id = doc.get('id', '')
            if doc_id not in doc_map:
                # Create minimal document object
                from vector_rag_system import Document
                from datetime import datetime
                doc_map[doc_id] = Document(
                    id=doc_id,
                    text=doc.get('text', ''),
                    filepath=doc.get('filepath', ''),
                    filename=doc.get('filename', ''),
                    doc_type=doc.get('type', ''),
                    timestamp=datetime.fromisoformat(doc.get('timestamp', datetime.now().isoformat())),
                    metadata=doc.get('metadata', {})
                )

        for doc_id, hybrid_score in top_fused:
            if doc_id not in doc_map:
                continue

            doc = doc_map[doc_id]

            # Get individual scores and ranks
            dense_rank, dense_score = dense_map.get(doc_id, (999, 0.0))
            sparse_rank, sparse_score = sparse_map.get(doc_id, (999, 0.0))

            result = HybridSearchResult(
                doc_id=doc_id,
                text=doc.text,
                filename=doc.filename,
                filepath=doc.filepath,
                doc_type=doc.doc_type,
                timestamp=doc.timestamp.isoformat() if hasattr(doc.timestamp, 'isoformat') else str(doc.timestamp),
                dense_score=dense_score,
                sparse_score=sparse_score,
                hybrid_score=hybrid_score,
                dense_rank=dense_rank,
                sparse_rank=sparse_rank,
                metadata=doc.metadata
            )
            results.append(result)

        logger.info(
            f"Hybrid search: {len(dense_results)} dense + {len(sparse_results)} sparse "
            f"→ {len(results)} fused results"
        )

        return results

    def ingest_document(self, filepath: Path, **kwargs) -> Dict:
        """
        Ingest document into both dense and sparse indices

        Args:
            filepath: Document path
            **kwargs: Additional arguments for VectorRAGSystem.ingest_document

        Returns:
            Ingestion result
        """
        # Ingest into dense index (Qdrant)
        result = self.vector_rag.ingest_document(filepath, **kwargs)

        # Resync BM25 index
        if result.get('status') == 'success':
            self._sync_bm25_index()

        return result

    def get_stats(self) -> Dict:
        """Get system statistics"""
        base_stats = self.vector_rag.get_stats()

        return {
            **base_stats,
            'hybrid_search': {
                'bm25_documents': len(self.bm25_index.documents),
                'dense_weight': self.dense_weight,
                'sparse_weight': self.sparse_weight,
                'fusion_method': 'Reciprocal Rank Fusion (RRF)',
            }
        }


# ============================================================================
# Standalone Utility Functions
# ============================================================================

def reciprocal_rank_fusion(
    results_list_1: List[Dict],
    results_list_2: List[Dict],
    k: int = 60
) -> List[Dict]:
    """
    Merge two result lists using Reciprocal Rank Fusion (RRF)

    RRF formula: score(d) = sum(1 / (k + rank(d)))

    Args:
        results_list_1: First result list (dicts with 'id' and 'score')
        results_list_2: Second result list (dicts with 'id' and 'score')
        k: RRF constant (default 60, from original paper)

    Returns:
        Merged results sorted by RRF score

    Reference:
        Cormack et al. (2009) "Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods"
    """
    # Build rank maps
    rank_map_1 = {r['id']: i + 1 for i, r in enumerate(results_list_1)}
    rank_map_2 = {r['id']: i + 1 for i, r in enumerate(results_list_2)}

    # Get all unique doc IDs
    all_ids = set(rank_map_1.keys()) | set(rank_map_2.keys())

    # Compute RRF scores
    rrf_scores = {}
    for doc_id in all_ids:
        score = 0.0

        if doc_id in rank_map_1:
            score += 1.0 / (k + rank_map_1[doc_id])

        if doc_id in rank_map_2:
            score += 1.0 / (k + rank_map_2[doc_id])

        rrf_scores[doc_id] = score

    # Find documents in original lists
    doc_map = {}
    for r in results_list_1 + results_list_2:
        if r['id'] not in doc_map:
            doc_map[r['id']] = r

    # Build merged results
    merged = []
    for doc_id in sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True):
        doc = doc_map[doc_id].copy()
        doc['score'] = rrf_scores[doc_id]
        merged.append(doc)

    return merged


# Example usage and testing
if __name__ == "__main__":
    import sys
    from vector_rag_system import VectorRAGSystem

    print("=== Hybrid Search System Test ===\n")

    # Initialize vector RAG
    print("Initializing vector RAG...")
    try:
        rag = VectorRAGSystem()
    except Exception as e:
        print(f"❌ Failed to initialize vector RAG: {e}")
        sys.exit(1)

    # Get stats
    stats = rag.get_stats()
    if stats['total_documents'] == 0:
        print("⚠️  No documents in collection. Ingest some documents first.")
        sys.exit(0)

    print(f"✓ Vector RAG initialized: {stats['total_documents']} documents\n")

    # Initialize hybrid search
    print("Initializing hybrid search...")
    try:
        hybrid = HybridSearchSystem(rag)
    except Exception as e:
        print(f"❌ Failed to initialize hybrid search: {e}")
        print("Install rank-bm25: pip install rank-bm25")
        sys.exit(1)

    print("\n" + "="*60 + "\n")

    # Test searches
    test_queries = [
        "error message",
        "VPN connection 404",
        "network timeout",
    ]

    for query in test_queries:
        print(f"Query: '{query}'\n")

        # Compare dense-only vs hybrid
        print("Dense-only results:")
        dense_results = rag.search(query, limit=5)
        for i, r in enumerate(dense_results, 1):
            print(f"  {i}. [Score: {r.score:.3f}] {r.document.filename}")

        print("\nHybrid results:")
        hybrid_results = hybrid.search(query, limit=5)
        for i, r in enumerate(hybrid_results, 1):
            print(f"  {i}. [Hybrid: {r.hybrid_score:.4f}] "
                  f"(Dense: {r.dense_score:.3f}, BM25: {r.sparse_score:.1f}) "
                  f"{r.filename}")

        print("\n" + "="*60 + "\n")

    # Statistics
    hybrid_stats = hybrid.get_stats()
    print("Hybrid Search Statistics:")
    print(f"  Total documents:  {hybrid_stats['total_documents']}")
    print(f"  BM25 documents:   {hybrid_stats['hybrid_search']['bm25_documents']}")
    print(f"  Dense weight:     {hybrid_stats['hybrid_search']['dense_weight']}")
    print(f"  Sparse weight:    {hybrid_stats['hybrid_search']['sparse_weight']}")
    print(f"  Fusion method:    {hybrid_stats['hybrid_search']['fusion_method']}")

    print("\n✓ Hybrid search test complete")
