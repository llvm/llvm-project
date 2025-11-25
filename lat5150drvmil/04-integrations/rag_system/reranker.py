#!/usr/bin/env python3
"""
Unified Reranking Module for RAG System

Supports multiple reranking backends:
1. Cross-Encoder (local, fast)
2. Jina Reranker v2 (local or API, multilingual)

Implements two-stage retrieval:
Stage 1: Fast vector search with Qdrant (retrieve top-50 to top-100)
Stage 2: Accurate reranking with reranker (return top-10)

Expected gains: +5-10% Precision@10, +3-5% Hit@3

Based on research:
- Pinecone (2024): Cross-encoder reranking improves precision 5-10%
- Nomic AI (2024): Two-stage retrieval balances speed and accuracy
- Jina AI (2024): Multilingual reranking with 161M parameter model
"""

import logging
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum
import time

logger = logging.getLogger(__name__)

# Try to import dependencies
try:
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    CROSS_ENCODER_AVAILABLE = False
    logger.warning("sentence-transformers not available. Install: pip install sentence-transformers")

# Try to import Jina reranker (from our jina_reranker.py)
try:
    from jina_reranker import JinaReranker, LocalJinaReranker
    JINA_RERANKER_AVAILABLE = True
except ImportError:
    JINA_RERANKER_AVAILABLE = False
    logger.warning("jina_reranker module not available")


@dataclass
class RerankResult:
    """Reranked search result"""
    doc_id: str
    text: str
    initial_score: float  # From vector search
    rerank_score: float   # From cross-encoder
    rank_change: int      # Position change (positive = moved up)
    metadata: Dict


class CrossEncoderReranker:
    """
    Cross-encoder reranker for two-stage retrieval

    Stage 1 (Fast): Vector similarity search → top-100 candidates
    Stage 2 (Accurate): Cross-encoder scoring → top-10 results

    Models:
    - cross-encoder/ms-marco-MiniLM-L-6-v2 (fast, 80M params)
    - cross-encoder/ms-marco-MiniLM-L-12-v2 (balanced, 33M params)
    - cross-encoder/ms-marco-TinyBERT-L-2-v2 (fastest, 15M params)
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        use_gpu: bool = True
    ):
        """
        Initialize cross-encoder reranker

        Args:
            model_name: HuggingFace model name
            use_gpu: Use GPU if available
        """
        if not CROSS_ENCODER_AVAILABLE:
            raise ImportError("sentence-transformers required. Install: pip install sentence-transformers")

        device = 'cuda' if use_gpu else 'cpu'
        logger.info(f"Loading cross-encoder: {model_name} on {device}")

        self.model = CrossEncoder(model_name, device=device, max_length=512)
        self.model_name = model_name

        logger.info(f"✓ Cross-encoder loaded: {model_name}")

    def rerank(
        self,
        query: str,
        documents: List[Dict],
        top_k: int = 10
    ) -> List[RerankResult]:
        """
        Rerank documents using cross-encoder

        Args:
            query: Search query
            documents: List of documents with 'text' and 'id' fields
            top_k: Number of top results to return

        Returns:
            List of reranked results with scores
        """
        if not documents:
            return []

        # Create query-document pairs
        pairs = [(query, doc.get('text', '')) for doc in documents]

        # Score with cross-encoder
        start_time = time.time()
        scores = self.model.predict(pairs)
        rerank_time = time.time() - start_time

        # Sort by rerank score
        scored_docs = []
        for i, (doc, score) in enumerate(zip(documents, scores)):
            scored_docs.append({
                'doc': doc,
                'initial_rank': i,
                'initial_score': doc.get('score', 0.0),
                'rerank_score': float(score),
            })

        # Sort by rerank score (descending)
        scored_docs.sort(key=lambda x: x['rerank_score'], reverse=True)

        # Build results
        results = []
        for new_rank, item in enumerate(scored_docs[:top_k]):
            rank_change = item['initial_rank'] - new_rank
            doc = item['doc']

            result = RerankResult(
                doc_id=doc.get('id', ''),
                text=doc.get('text', ''),
                initial_score=item['initial_score'],
                rerank_score=item['rerank_score'],
                rank_change=rank_change,
                metadata=doc.get('metadata', {})
            )
            results.append(result)

        logger.debug(f"Reranked {len(documents)} docs in {rerank_time*1000:.1f}ms")

        return results

    def score_pairs(
        self,
        query: str,
        texts: List[str]
    ) -> List[float]:
        """
        Score query-text pairs

        Args:
            query: Search query
            texts: List of text snippets

        Returns:
            List of relevance scores
        """
        pairs = [(query, text) for text in texts]
        scores = self.model.predict(pairs)
        return scores.tolist()


class LLMReranker:
    """
    LLM-based reranking using local Ollama

    Uses LLM to judge document relevance (slower but potentially more accurate)
    Useful for complex queries requiring deep understanding
    """

    def __init__(self, llm_endpoint: str = "http://localhost:11434", model: str = "llama3.2:3b"):
        """
        Initialize LLM reranker

        Args:
            llm_endpoint: Ollama API endpoint
            model: Model name
        """
        self.llm_endpoint = llm_endpoint
        self.model = model
        logger.info(f"LLM reranker initialized: {model}")

    def rerank(
        self,
        query: str,
        documents: List[Dict],
        top_k: int = 10
    ) -> List[RerankResult]:
        """
        Rerank documents using LLM judgment

        Args:
            query: Search query
            documents: List of documents
            top_k: Number of results to return

        Returns:
            Reranked results
        """
        import requests

        scored_docs = []

        for i, doc in enumerate(documents):
            # Build prompt
            prompt = self._build_relevance_prompt(query, doc.get('text', ''))

            try:
                # Call LLM
                response = requests.post(
                    f"{self.llm_endpoint}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.1,
                            "max_tokens": 50
                        }
                    },
                    timeout=5
                )

                if response.status_code == 200:
                    result = response.json()
                    output = result.get('response', '').strip()

                    # Parse score from output
                    score = self._parse_score(output)
                else:
                    score = 0.0

            except Exception as e:
                logger.warning(f"LLM reranking failed for doc {i}: {e}")
                score = 0.0

            scored_docs.append({
                'doc': doc,
                'initial_rank': i,
                'initial_score': doc.get('score', 0.0),
                'rerank_score': score,
            })

        # Sort by rerank score
        scored_docs.sort(key=lambda x: x['rerank_score'], reverse=True)

        # Build results
        results = []
        for new_rank, item in enumerate(scored_docs[:top_k]):
            rank_change = item['initial_rank'] - new_rank
            doc = item['doc']

            result = RerankResult(
                doc_id=doc.get('id', ''),
                text=doc.get('text', ''),
                initial_score=item['initial_score'],
                rerank_score=item['rerank_score'],
                rank_change=rank_change,
                metadata=doc.get('metadata', {})
            )
            results.append(result)

        return results

    def _build_relevance_prompt(self, query: str, document: str) -> str:
        """Build prompt for relevance judgment"""
        # Truncate document if too long
        if len(document) > 500:
            document = document[:500] + "..."

        prompt = f"""Rate how relevant this document is to the query on a scale of 0-10.

Query: "{query}"

Document: "{document}"

Respond with ONLY a number from 0 (not relevant) to 10 (highly relevant).

Relevance score:"""

        return prompt

    def _parse_score(self, output: str) -> float:
        """Parse score from LLM output"""
        import re

        # Extract first number from output
        match = re.search(r'(\d+\.?\d*)', output)
        if match:
            score = float(match.group(1))
            # Normalize to 0-1 range
            return min(score / 10.0, 1.0)
        else:
            return 0.0


class TwoStageRAG:
    """
    Two-stage retrieval system integrating vector search + reranking

    Usage:
        rag = TwoStageRAG(vector_rag_system, reranker)
        results = rag.search("error message", final_k=10, candidate_k=100)

    Stage 1: Fast vector retrieval (100 candidates in ~10-20ms)
    Stage 2: Accurate reranking (top 10 in ~50-100ms)
    Total: ~60-120ms for high-quality results
    """

    def __init__(
        self,
        vector_rag,  # VectorRAGSystem instance
        reranker_type: str = "cross-encoder",  # 'cross-encoder' or 'llm'
        reranker_model: Optional[str] = None
    ):
        """
        Initialize two-stage RAG

        Args:
            vector_rag: VectorRAGSystem instance
            reranker_type: 'cross-encoder' or 'llm'
            reranker_model: Optional model override
        """
        self.vector_rag = vector_rag

        # Initialize reranker
        if reranker_type == "cross-encoder":
            model = reranker_model or "cross-encoder/ms-marco-MiniLM-L-6-v2"
            self.reranker = CrossEncoderReranker(model_name=model)
        elif reranker_type == "llm":
            self.reranker = LLMReranker(model=reranker_model or "llama3.2:3b")
        else:
            raise ValueError(f"Unknown reranker type: {reranker_type}")

        self.reranker_type = reranker_type

        logger.info(f"✓ Two-stage RAG initialized: {reranker_type}")

    def search(
        self,
        query: str,
        final_k: int = 10,
        candidate_k: int = 100,
        score_threshold: float = 0.0,
        filters: Optional[Dict] = None
    ) -> List[RerankResult]:
        """
        Two-stage search with reranking

        Args:
            query: Search query
            final_k: Number of final results
            candidate_k: Number of candidates from stage 1
            score_threshold: Minimum score for stage 1
            filters: Filters for stage 1

        Returns:
            Reranked results
        """
        # Stage 1: Fast vector retrieval
        stage1_start = time.time()
        candidates = self.vector_rag.search(
            query=query,
            limit=candidate_k,
            score_threshold=score_threshold,
            filters=filters
        )
        stage1_time = time.time() - stage1_start

        if not candidates:
            logger.warning(f"No candidates found for query: {query}")
            return []

        # Prepare documents for reranking
        docs_for_reranking = []
        for result in candidates:
            docs_for_reranking.append({
                'id': result.document.id,
                'text': result.document.text,
                'score': result.score,
                'metadata': result.document.metadata
            })

        # Stage 2: Accurate reranking
        stage2_start = time.time()
        reranked = self.reranker.rerank(
            query=query,
            documents=docs_for_reranking,
            top_k=final_k
        )
        stage2_time = time.time() - stage2_start

        logger.info(
            f"Two-stage search: Stage1={stage1_time*1000:.1f}ms ({len(candidates)} docs), "
            f"Stage2={stage2_time*1000:.1f}ms ({len(reranked)} docs)"
        )

        return reranked

    def get_stats(self) -> Dict:
        """Get system statistics"""
        return {
            'reranker_type': self.reranker_type,
            'reranker_model': getattr(self.reranker, 'model_name', self.reranker.model),
            'vector_rag_stats': self.vector_rag.get_stats()
        }


# ============================================================================
# Unified Reranker Interface
# ============================================================================

class RerankerType(Enum):
    """Available reranker types"""
    CROSS_ENCODER = "cross_encoder"
    JINA_LOCAL = "jina_local"
    JINA_API = "jina_api"
    LLM = "llm"


def create_reranker(
    reranker_type: str = "cross_encoder",
    model_name: Optional[str] = None,
    use_gpu: bool = True,
    api_key: Optional[str] = None,
    **kwargs
):
    """
    Factory function to create appropriate reranker

    Args:
        reranker_type: Type of reranker (cross_encoder, jina_local, jina_api, llm)
        model_name: Model name (optional, uses defaults)
        use_gpu: Use GPU if available
        api_key: API key for Jina API reranker
        **kwargs: Additional arguments for specific rerankers

    Returns:
        Reranker instance

    Examples:
        # Cross-encoder (fast, local)
        reranker = create_reranker("cross_encoder")

        # Jina local (multilingual, local)
        reranker = create_reranker("jina_local", use_gpu=True)

        # Jina API (multilingual, cloud)
        reranker = create_reranker("jina_api", api_key="your_key")

        # LLM-based (slow, accurate)
        reranker = create_reranker("llm", model="llama3.2:3b")
    """
    reranker_type = reranker_type.lower()

    if reranker_type == "cross_encoder" or reranker_type == RerankerType.CROSS_ENCODER.value:
        if not CROSS_ENCODER_AVAILABLE:
            raise ImportError("sentence-transformers required for cross-encoder")

        if model_name is None:
            model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"

        return CrossEncoderReranker(model_name=model_name, use_gpu=use_gpu)

    elif reranker_type == "jina_local" or reranker_type == RerankerType.JINA_LOCAL.value:
        if not JINA_RERANKER_AVAILABLE:
            raise ImportError("jina_reranker module required")

        if model_name is None:
            model_name = "jinaai/jina-reranker-v2-base-multilingual"

        return LocalJinaReranker(
            model_name=model_name,
            use_gpu=use_gpu,
            batch_size=kwargs.get('batch_size', 16)
        )

    elif reranker_type == "jina_api" or reranker_type == RerankerType.JINA_API.value:
        if not JINA_RERANKER_AVAILABLE:
            raise ImportError("jina_reranker module required")

        if model_name is None:
            model_name = "jina-reranker-v2-base-multilingual"

        return JinaReranker(
            api_key=api_key,
            model=model_name,
            api_url=kwargs.get('api_url', "https://api.jina.ai/v1/rerank"),
            timeout=kwargs.get('timeout', 30)
        )

    elif reranker_type == "llm" or reranker_type == RerankerType.LLM.value:
        return LLMReranker(
            llm_endpoint=kwargs.get('llm_endpoint', "http://localhost:11434"),
            model=model_name or kwargs.get('model', "llama3.2:3b")
        )

    else:
        raise ValueError(
            f"Unknown reranker type: {reranker_type}. "
            f"Choose from: {[e.value for e in RerankerType]}"
        )


class UnifiedReranker:
    """
    Unified reranker interface supporting multiple backends

    Automatically falls back to alternatives if primary reranker fails

    Usage:
        reranker = UnifiedReranker(
            primary="jina_local",
            fallback="cross_encoder",
            use_gpu=True
        )

        results = reranker.rerank(query, documents, top_k=10)
    """

    def __init__(
        self,
        primary: str = "cross_encoder",
        fallback: Optional[str] = None,
        use_gpu: bool = True,
        api_key: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize unified reranker

        Args:
            primary: Primary reranker type
            fallback: Fallback reranker type (optional)
            use_gpu: Use GPU
            api_key: API key (for Jina API)
            **kwargs: Additional arguments
        """
        self.primary_type = primary
        self.fallback_type = fallback

        # Initialize primary reranker
        try:
            self.primary = create_reranker(
                reranker_type=primary,
                use_gpu=use_gpu,
                api_key=api_key,
                **kwargs
            )
            logger.info(f"Primary reranker initialized: {primary}")
        except Exception as e:
            logger.error(f"Failed to initialize primary reranker ({primary}): {e}")
            self.primary = None

        # Initialize fallback reranker
        self.fallback = None
        if fallback:
            try:
                self.fallback = create_reranker(
                    reranker_type=fallback,
                    use_gpu=use_gpu,
                    **kwargs
                )
                logger.info(f"Fallback reranker initialized: {fallback}")
            except Exception as e:
                logger.warning(f"Failed to initialize fallback reranker ({fallback}): {e}")

    def rerank(
        self,
        query: str,
        documents: List[Dict],
        top_k: int = 10
    ) -> List[RerankResult]:
        """
        Rerank documents with automatic fallback

        Args:
            query: Search query
            documents: List of documents
            top_k: Number of results

        Returns:
            Reranked results
        """
        if not documents:
            return []

        # Try primary reranker
        if self.primary:
            try:
                return self.primary.rerank(query, documents, top_k)
            except Exception as e:
                logger.error(f"Primary reranker failed: {e}")

        # Try fallback
        if self.fallback:
            try:
                logger.info("Using fallback reranker")
                return self.fallback.rerank(query, documents, top_k)
            except Exception as e:
                logger.error(f"Fallback reranker failed: {e}")

        # No reranking available, return original order
        logger.warning("No reranker available, returning original order")
        return [
            RerankResult(
                doc_id=doc.get('id', f'doc_{i}'),
                text=doc.get('text', ''),
                initial_score=doc.get('score', 0.0),
                rerank_score=doc.get('score', 0.0),
                rank_change=0,
                metadata=doc.get('metadata', {})
            )
            for i, doc in enumerate(documents[:top_k])
        ]


# Example usage and testing
if __name__ == "__main__":
    print("=== Cross-Encoder Reranker Test ===\n")

    # Test cross-encoder
    if CROSS_ENCODER_AVAILABLE:
        reranker = CrossEncoderReranker()

        query = "VPN connection timeout error"
        documents = [
            {'id': '1', 'text': 'VPN connection failed with timeout after 30 seconds', 'score': 0.85},
            {'id': '2', 'text': 'Network error occurred during file download', 'score': 0.82},
            {'id': '3', 'text': 'VPN authentication succeeded but connection dropped', 'score': 0.80},
            {'id': '4', 'text': 'System memory usage is high', 'score': 0.75},
            {'id': '5', 'text': 'VPN service restarted due to timeout', 'score': 0.78},
        ]

        print(f"Query: {query}\n")
        print("Initial ranking (by vector similarity):")
        for i, doc in enumerate(documents, 1):
            print(f"{i}. [Score: {doc['score']:.2f}] {doc['text']}")

        print("\n" + "="*60 + "\n")

        # Rerank
        reranked = reranker.rerank(query, documents, top_k=5)

        print("After reranking (by cross-encoder):")
        for i, result in enumerate(reranked, 1):
            change_str = f"+{result.rank_change}" if result.rank_change > 0 else str(result.rank_change)
            print(f"{i}. [Score: {result.rerank_score:.3f}] [{change_str}] {result.text}")

        print("\n" + "="*60 + "\n")

        # Show most improved
        improvements = [r for r in reranked if r.rank_change > 0]
        if improvements:
            print("Biggest improvements:")
            for result in sorted(improvements, key=lambda x: x.rank_change, reverse=True)[:3]:
                print(f"  • {result.text[:60]}...")
                print(f"    Moved up {result.rank_change} positions (rerank: {result.rerank_score:.3f})")
        else:
            print("No significant improvements in ranking")

    else:
        print("⚠️  sentence-transformers not installed")
        print("Install: pip install sentence-transformers")
