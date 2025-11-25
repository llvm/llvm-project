#!/usr/bin/env python3
"""
Jina Reranker API Integration

Jina Reranker v2 is a state-of-the-art multilingual reranking model
for precision improvement in RAG systems.

Features:
- 161M parameters, optimized for reranking tasks
- Multilingual support (89+ languages)
- Cross-encoder architecture (query-document interaction)
- API-based (cloud) or local deployment

Performance gain: +5-10% precision on top-K results
Expected use: Second-stage reranker after vector search

Workflow:
1. Vector search → top-100 candidates (recall-focused)
2. Jina Reranker → top-10 results (precision-focused)
3. Total latency: ~50-150ms depending on candidate count

API documentation: https://jina.ai/reranker
"""

import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import requests
import os

logger = logging.getLogger(__name__)


@dataclass
class RerankedResult:
    """Reranked search result"""
    document_id: str
    text: str
    original_score: float  # Vector search score
    rerank_score: float  # Reranker score
    rank: int  # Position after reranking
    metadata: Dict


class JinaReranker:
    """
    Jina Reranker API client for second-stage reranking

    Usage:
        reranker = JinaReranker(api_key="your_key")
        results = reranker.rerank(
            query="VPN connection error",
            documents=["...", "...", "..."],
            top_k=10
        )

    Features:
    - Cross-encoder scoring (query+doc interaction)
    - Multilingual support
    - Batch processing
    - Configurable top-k

    Best practices:
    - Use after vector search (rerank top-50 to top-100)
    - Cache rerank scores for repeated queries
    - Monitor API rate limits
    - Fall back to vector scores if API fails
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "jina-reranker-v2-base-multilingual",
        api_url: str = "https://api.jina.ai/v1/rerank",
        timeout: int = 30
    ):
        """
        Initialize Jina Reranker

        Args:
            api_key: Jina API key (or set JINA_API_KEY env var)
            model: Reranker model name
            api_url: API endpoint
            timeout: Request timeout in seconds
        """
        self.api_key = api_key or os.getenv("JINA_API_KEY")
        if not self.api_key:
            raise ValueError("JINA_API_KEY required (env var or constructor arg)")

        self.model = model
        self.api_url = api_url
        self.timeout = timeout

        logger.info(f"Jina Reranker initialized: {model}")

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: int = 10,
        return_documents: bool = True,
        document_ids: Optional[List[str]] = None,
        original_scores: Optional[List[float]] = None,
        metadata: Optional[List[Dict]] = None
    ) -> List[RerankedResult]:
        """
        Rerank documents using Jina Reranker API

        Args:
            query: Search query
            documents: List of document texts to rerank
            top_k: Return top K results
            return_documents: Include document text in results
            document_ids: Document IDs (generated if not provided)
            original_scores: Original vector search scores
            metadata: Document metadata

        Returns:
            List of RerankedResult objects sorted by rerank score
        """
        if not documents:
            return []

        # Prepare document IDs
        if document_ids is None:
            document_ids = [f"doc_{i}" for i in range(len(documents))]

        # Prepare original scores
        if original_scores is None:
            original_scores = [0.0] * len(documents)

        # Prepare metadata
        if metadata is None:
            metadata = [{}] * len(documents)

        # Call Jina Reranker API
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            payload = {
                "model": self.model,
                "query": query,
                "documents": documents,
                "top_n": top_k,
                "return_documents": return_documents
            }

            response = requests.post(
                self.api_url,
                json=payload,
                headers=headers,
                timeout=self.timeout
            )

            response.raise_for_status()
            result = response.json()

            # Parse results
            reranked = []
            for item in result.get("results", []):
                idx = item.get("index")
                rerank_score = item.get("relevance_score", 0.0)
                rank = item.get("rank", len(reranked) + 1)

                reranked.append(RerankedResult(
                    document_id=document_ids[idx],
                    text=documents[idx],
                    original_score=original_scores[idx],
                    rerank_score=rerank_score,
                    rank=rank,
                    metadata=metadata[idx]
                ))

            logger.info(f"Reranked {len(documents)} docs → top {len(reranked)} results")

            return reranked

        except requests.exceptions.RequestException as e:
            logger.error(f"Jina Reranker API error: {e}")
            # Fall back to original ranking
            logger.warning("Falling back to original vector search ranking")
            return self._fallback_ranking(
                documents, document_ids, original_scores, metadata, top_k
            )

        except Exception as e:
            logger.error(f"Unexpected error in reranking: {e}")
            return self._fallback_ranking(
                documents, document_ids, original_scores, metadata, top_k
            )

    def _fallback_ranking(
        self,
        documents: List[str],
        document_ids: List[str],
        original_scores: List[float],
        metadata: List[Dict],
        top_k: int
    ) -> List[RerankedResult]:
        """
        Fallback to original ranking if API fails

        Args:
            documents: Document texts
            document_ids: Document IDs
            original_scores: Original scores
            metadata: Metadata
            top_k: Top K to return

        Returns:
            RerankedResult list (using original scores)
        """
        results = []
        for i, (doc, doc_id, score, meta) in enumerate(
            zip(documents, document_ids, original_scores, metadata)
        ):
            results.append(RerankedResult(
                document_id=doc_id,
                text=doc,
                original_score=score,
                rerank_score=score,  # Use original score as fallback
                rank=i + 1,
                metadata=meta
            ))

        # Sort by original score
        results.sort(key=lambda x: x.original_score, reverse=True)

        # Return top K
        return results[:top_k]


class LocalJinaReranker:
    """
    Local Jina Reranker using transformers (no API needed)

    For use cases requiring:
    - Offline operation
    - Lower latency
    - No API costs
    - Full control

    Requires:
    - transformers
    - torch
    - GPU recommended for production

    Usage:
        reranker = LocalJinaReranker(use_gpu=True)
        results = reranker.rerank(query, documents, top_k=10)
    """

    def __init__(
        self,
        model_name: str = "jinaai/jina-reranker-v2-base-multilingual",
        use_gpu: bool = True,
        batch_size: int = 16
    ):
        """
        Initialize local Jina Reranker

        Args:
            model_name: HuggingFace model name
            use_gpu: Use GPU if available
            batch_size: Batch size for inference
        """
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            import torch
        except ImportError:
            raise ImportError("transformers and torch required for local reranker")

        self.model_name = model_name
        self.batch_size = batch_size
        self.device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'

        logger.info(f"Loading local Jina Reranker: {model_name} on {self.device}")

        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        logger.info("✓ Local Jina Reranker initialized")

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: int = 10,
        document_ids: Optional[List[str]] = None,
        original_scores: Optional[List[float]] = None,
        metadata: Optional[List[Dict]] = None
    ) -> List[RerankedResult]:
        """
        Rerank documents using local model

        Args:
            query: Search query
            documents: List of document texts
            top_k: Return top K results
            document_ids: Document IDs
            original_scores: Original scores
            metadata: Metadata

        Returns:
            List of RerankedResult objects
        """
        import torch

        if not documents:
            return []

        # Prepare IDs, scores, metadata
        if document_ids is None:
            document_ids = [f"doc_{i}" for i in range(len(documents))]
        if original_scores is None:
            original_scores = [0.0] * len(documents)
        if metadata is None:
            metadata = [{}] * len(documents)

        # Compute rerank scores
        rerank_scores = []

        for i in range(0, len(documents), self.batch_size):
            batch_docs = documents[i:i+self.batch_size]

            # Tokenize query-document pairs
            inputs = self.tokenizer(
                [query] * len(batch_docs),
                batch_docs,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            ).to(self.device)

            # Get scores
            with torch.no_grad():
                outputs = self.model(**inputs)
                scores = outputs.logits.squeeze(-1).cpu().tolist()

                # Handle single score case
                if not isinstance(scores, list):
                    scores = [scores]

                rerank_scores.extend(scores)

        # Create results
        results = []
        for i, (doc, doc_id, orig_score, rerank_score, meta) in enumerate(
            zip(documents, document_ids, original_scores, rerank_scores, metadata)
        ):
            results.append(RerankedResult(
                document_id=doc_id,
                text=doc,
                original_score=orig_score,
                rerank_score=rerank_score,
                rank=i + 1,  # Will be updated after sorting
                metadata=meta
            ))

        # Sort by rerank score
        results.sort(key=lambda x: x.rerank_score, reverse=True)

        # Update ranks
        for i, result in enumerate(results, 1):
            result.rank = i

        # Return top K
        return results[:top_k]


# Example usage
if __name__ == "__main__":
    print("=== Jina Reranker Test ===\n")

    # Test documents (cyber forensics examples)
    query = "VPN authentication timeout error"

    documents = [
        "The VPN connection failed due to authentication timeout. Gateway experienced high load (89% CPU).",
        "User reported slow internet speeds. No VPN issues detected. ISP throttling suspected.",
        "Database query for user credentials took 8.2 seconds, exceeding 5s timeout threshold.",
        "Firewall logs show no blocking rules triggered during VPN connection attempts.",
        "System reboot resolved temporary network connectivity issues. VPN working normally now.",
        "Certificate validation passed successfully. No SSL/TLS errors in logs.",
        "Recommendation: Increase timeout threshold or optimize user lookup query for VPN auth.",
        "Multiple retry attempts with exponential backoff observed in network trace.",
        "Gateway performance monitoring enabled. CPU usage alerts configured at 80%.",
        "User account locked after 3 failed authentication attempts. Password reset required."
    ]

    print(f"Query: '{query}'")
    print(f"Documents: {len(documents)} candidates\n")
    print("="*60 + "\n")

    # Test API reranker (will fail without API key, showing fallback)
    print("Testing Jina Reranker API (with fallback)...\n")

    try:
        reranker = JinaReranker(api_key="test_key_will_fail")
        results = reranker.rerank(
            query=query,
            documents=documents,
            top_k=5,
            original_scores=[1.0 - i*0.1 for i in range(len(documents))]
        )

        print(f"Top {len(results)} reranked results:\n")
        for i, result in enumerate(results, 1):
            print(f"{i}. Rerank score: {result.rerank_score:.3f} (original: {result.original_score:.2f})")
            print(f"   Text: {result.text[:80]}...")
            print()

    except Exception as e:
        print(f"⚠️  API test failed (expected): {e}")
        print("Set JINA_API_KEY environment variable to test API reranking\n")

    print("="*60 + "\n")

    # Test local reranker
    print("Testing Local Jina Reranker...\n")

    try:
        local_reranker = LocalJinaReranker(use_gpu=False)

        results = local_reranker.rerank(
            query=query,
            documents=documents,
            top_k=5,
            original_scores=[1.0 - i*0.1 for i in range(len(documents))]
        )

        print(f"Top {len(results)} locally reranked results:\n")
        for i, result in enumerate(results, 1):
            print(f"{i}. Rerank score: {result.rerank_score:.3f} (original: {result.original_score:.2f})")
            print(f"   Text: {result.text[:80]}...")
            print()

        print("✓ Local reranker test complete")

    except ImportError:
        print("⚠️  Local reranker requires transformers and torch")
        print("Install with: pip install transformers torch")

    print("\nKey benefits:")
    print("- +5-10% precision improvement on top-K results")
    print("- Cross-encoder captures query-document interactions")
    print("- Multilingual support (89+ languages)")
    print("- API or local deployment options")
