#!/usr/bin/env python3
"""
Self-Improving Agentic RAG System
==================================
Based on concepts from "Building a Self-Improving Agentic RAG System"

Features:
- Feedback Loop Learning: Improves retrieval based on user feedback
- Adaptive Chunking: Dynamically adjusts chunk sizes based on content
- Query Refinement: Learns to rewrite queries for better retrieval
- Relevance Scoring: Self-calibrating relevance thresholds
- Cache Optimization: Learns frequently accessed patterns

Architecture:
- Retrieval Agent: Autonomous document retrieval with learning
- Quality Evaluator: Assesses retrieval quality and triggers improvements
- Adaptation Engine: Continuously improves retrieval strategies

Author: LAT5150DRVMIL AI Platform
Version: 1.0.0
"""

import json
import hashlib
import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict

logger = logging.getLogger(__name__)


class FeedbackType(str, Enum):
    """Types of user feedback"""
    POSITIVE = "positive"    # Good retrieval
    NEGATIVE = "negative"    # Poor retrieval
    PARTIAL = "partial"      # Partially useful
    IRRELEVANT = "irrelevant"  # Completely irrelevant


class AdaptationType(str, Enum):
    """Types of adaptations"""
    QUERY_REWRITE = "query_rewrite"
    CHUNK_SIZE = "chunk_size"
    THRESHOLD_ADJUST = "threshold_adjust"
    EMBEDDING_WEIGHT = "embedding_weight"


@dataclass
class RetrievalResult:
    """Result from RAG retrieval"""
    id: str
    content: str
    source: str
    relevance_score: float
    chunk_index: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FeedbackEntry:
    """User feedback on retrieval"""
    query: str
    result_ids: List[str]
    feedback_type: FeedbackType
    feedback_score: float  # -1 to 1
    user_comment: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class AdaptationRecord:
    """Record of an adaptation"""
    adaptation_type: AdaptationType
    old_value: Any
    new_value: Any
    trigger: str
    improvement: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class QueryPattern:
    """Learned query pattern"""
    pattern: str
    successful_rewrites: List[str] = field(default_factory=list)
    success_rate: float = 0.5
    usage_count: int = 0


class QueryRefiner:
    """
    Learns to rewrite queries for better retrieval.

    Techniques:
    - Expansion: Add related terms
    - Simplification: Remove noise words
    - Decomposition: Split complex queries
    - Specialization: Add domain-specific terms
    """

    def __init__(self):
        self.patterns: Dict[str, QueryPattern] = {}
        self.rewrite_history: List[Tuple[str, str, float]] = []  # (original, rewritten, score)

        # Domain-specific expansions
        self.domain_expansions = {
            'api': ['endpoint', 'REST', 'HTTP', 'request', 'response'],
            'error': ['exception', 'bug', 'issue', 'failure', 'crash'],
            'config': ['configuration', 'settings', 'options', 'parameters'],
            'auth': ['authentication', 'authorization', 'login', 'credentials'],
            'ml': ['machine learning', 'model', 'training', 'inference'],
            'db': ['database', 'SQL', 'query', 'table', 'schema'],
        }

        # Noise words to potentially remove
        self.noise_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'how', 'what', 'can', 'do'}

    def refine(self, query: str, context: Optional[Dict] = None) -> List[str]:
        """
        Generate refined query variants.

        Returns list of query variants sorted by expected effectiveness.
        """
        variants = [query]  # Original first

        query_lower = query.lower()
        words = query_lower.split()

        # 1. Domain expansion
        for term, expansions in self.domain_expansions.items():
            if term in query_lower:
                for expansion in expansions[:2]:
                    variants.append(f"{query} {expansion}")

        # 2. Noise word removal
        filtered = [w for w in words if w not in self.noise_words]
        if len(filtered) >= 2:
            variants.append(' '.join(filtered))

        # 3. Check learned patterns
        for pattern_key, pattern in self.patterns.items():
            if pattern_key in query_lower and pattern.successful_rewrites:
                # Use most successful rewrite
                best_rewrite = pattern.successful_rewrites[0]
                variants.append(query_lower.replace(pattern_key, best_rewrite))

        # Remove duplicates while preserving order
        seen = set()
        unique_variants = []
        for v in variants:
            if v not in seen:
                seen.add(v)
                unique_variants.append(v)

        return unique_variants[:5]

    def learn_from_feedback(self, original: str, rewritten: str, score: float):
        """Learn from feedback on query rewrites"""
        self.rewrite_history.append((original, rewritten, score))

        # Update patterns if improvement was significant
        if score > 0.7:
            # Find what changed
            orig_words = set(original.lower().split())
            new_words = set(rewritten.lower().split())

            added = new_words - orig_words
            for word in added:
                if word not in self.patterns:
                    self.patterns[word] = QueryPattern(pattern=word)
                self.patterns[word].success_rate = (
                    self.patterns[word].success_rate * 0.8 + score * 0.2
                )


class RelevanceCalibrator:
    """
    Self-calibrating relevance scoring.

    Learns optimal thresholds based on user feedback.
    """

    def __init__(self):
        self.threshold = 0.5  # Default relevance threshold
        self.min_threshold = 0.2
        self.max_threshold = 0.9
        self.feedback_history: List[Tuple[float, FeedbackType]] = []
        self.calibration_count = 0

    def should_include(self, score: float) -> bool:
        """Check if score meets threshold"""
        return score >= self.threshold

    def record_feedback(self, score: float, feedback: FeedbackType):
        """Record feedback for calibration"""
        self.feedback_history.append((score, feedback))

        # Calibrate periodically
        if len(self.feedback_history) % 10 == 0:
            self._calibrate()

    def _calibrate(self):
        """Adjust threshold based on feedback"""
        if len(self.feedback_history) < 10:
            return

        recent = self.feedback_history[-50:]

        # Calculate optimal threshold
        positive_scores = [s for s, f in recent if f == FeedbackType.POSITIVE]
        negative_scores = [s for s, f in recent if f in [FeedbackType.NEGATIVE, FeedbackType.IRRELEVANT]]

        if positive_scores and negative_scores:
            min_positive = min(positive_scores)
            max_negative = max(negative_scores)

            # Set threshold between them
            optimal = (min_positive + max_negative) / 2
            self.threshold = max(self.min_threshold, min(self.max_threshold, optimal))
            self.calibration_count += 1

            logger.info(f"Calibrated threshold to {self.threshold:.2f}")


class AdaptiveChunker:
    """
    Adaptive document chunking based on content type and feedback.
    """

    def __init__(self):
        self.default_chunk_size = 512
        self.overlap = 50

        # Content-type specific sizes
        self.content_type_sizes = {
            'code': 256,      # Smaller for code
            'documentation': 768,  # Larger for docs
            'conversation': 384,
            'default': 512
        }

        # Learned adjustments
        self.adjustments: Dict[str, int] = {}

    def get_chunk_size(self, content_type: str, source: Optional[str] = None) -> int:
        """Get optimal chunk size for content"""
        base = self.content_type_sizes.get(content_type, self.default_chunk_size)

        # Apply learned adjustments
        if source and source in self.adjustments:
            base += self.adjustments[source]

        return max(128, min(2048, base))

    def learn_from_feedback(self, source: str, feedback: FeedbackType, chunk_size: int):
        """Adjust chunk sizes based on feedback"""
        if source not in self.adjustments:
            self.adjustments[source] = 0

        if feedback == FeedbackType.POSITIVE:
            # This size works well, remember it
            pass
        elif feedback == FeedbackType.NEGATIVE:
            # Try different size - alternate between smaller and larger
            if self.adjustments[source] <= 0:
                self.adjustments[source] += 64
            else:
                self.adjustments[source] -= 64


class SelfImprovingRAG:
    """
    Self-improving RAG system that learns from feedback.

    Components:
    - Query Refiner: Learns better query formulations
    - Relevance Calibrator: Self-adjusting thresholds
    - Adaptive Chunker: Content-aware chunking
    - Feedback Loop: Continuous improvement
    """

    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path.home() / ".dsmil" / "rag_learning"
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Components
        self.query_refiner = QueryRefiner()
        self.relevance_calibrator = RelevanceCalibrator()
        self.adaptive_chunker = AdaptiveChunker()

        # History and learning
        self.feedback_history: List[FeedbackEntry] = []
        self.adaptation_history: List[AdaptationRecord] = []
        self.query_success_cache: Dict[str, List[str]] = {}  # query -> successful doc IDs

        # Metrics
        self.total_queries = 0
        self.successful_queries = 0
        self.adaptations_made = 0

        self._load()

    def retrieve(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        embedding_fn: Optional[callable] = None,
        k: int = 5
    ) -> List[RetrievalResult]:
        """
        Retrieve relevant documents with self-improvement.

        Args:
            query: User query
            documents: List of documents to search
            embedding_fn: Function to generate embeddings
            k: Number of results to return

        Returns:
            List of RetrievalResults
        """
        self.total_queries += 1

        # Refine query
        query_variants = self.query_refiner.refine(query)

        all_results = []

        for q_variant in query_variants[:3]:
            # Search with this variant
            results = self._search(q_variant, documents, embedding_fn, k * 2)
            all_results.extend(results)

        # Deduplicate and filter
        seen_ids = set()
        filtered_results = []
        for result in all_results:
            if result.id not in seen_ids:
                if self.relevance_calibrator.should_include(result.relevance_score):
                    seen_ids.add(result.id)
                    filtered_results.append(result)

        # Sort by relevance
        filtered_results.sort(key=lambda r: r.relevance_score, reverse=True)

        # Check cache for known good results
        cache_key = self._query_hash(query)
        if cache_key in self.query_success_cache:
            # Boost cached successful results
            cached_ids = set(self.query_success_cache[cache_key])
            for result in filtered_results:
                if result.id in cached_ids:
                    result.relevance_score = min(1.0, result.relevance_score + 0.1)

        return filtered_results[:k]

    def _search(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        embedding_fn: Optional[callable],
        k: int
    ) -> List[RetrievalResult]:
        """Perform actual search"""
        results = []
        query_lower = query.lower()
        query_terms = set(query_lower.split())

        for doc in documents:
            doc_id = doc.get('id', hashlib.md5(doc.get('content', '')[:100].encode()).hexdigest())
            content = doc.get('content', '')
            source = doc.get('source', 'unknown')

            # Simple keyword relevance
            content_lower = content.lower()
            content_terms = set(content_lower.split())

            # Calculate relevance
            term_overlap = len(query_terms & content_terms)
            relevance = term_overlap / max(len(query_terms), 1)

            # Boost for exact phrase match
            if query_lower in content_lower:
                relevance += 0.3

            # Boost for source matches in query
            if source.lower() in query_lower:
                relevance += 0.1

            relevance = min(1.0, relevance)

            if relevance > 0:
                results.append(RetrievalResult(
                    id=doc_id,
                    content=content[:1000],  # Truncate
                    source=source,
                    relevance_score=relevance,
                    chunk_index=0,
                    metadata=doc.get('metadata', {})
                ))

        results.sort(key=lambda r: r.relevance_score, reverse=True)
        return results[:k]

    def record_feedback(
        self,
        query: str,
        results: List[RetrievalResult],
        feedback_type: FeedbackType,
        feedback_score: float = 0.0,
        comment: Optional[str] = None
    ):
        """
        Record user feedback and trigger learning.

        Args:
            query: Original query
            results: Results that were returned
            feedback_type: Type of feedback
            feedback_score: Score from -1 (bad) to 1 (good)
            comment: Optional user comment
        """
        result_ids = [r.id for r in results]

        entry = FeedbackEntry(
            query=query,
            result_ids=result_ids,
            feedback_type=feedback_type,
            feedback_score=feedback_score,
            user_comment=comment
        )
        self.feedback_history.append(entry)

        # Update success metrics
        if feedback_type == FeedbackType.POSITIVE:
            self.successful_queries += 1

            # Cache successful query-document pairs
            cache_key = self._query_hash(query)
            if cache_key not in self.query_success_cache:
                self.query_success_cache[cache_key] = []
            self.query_success_cache[cache_key].extend(result_ids[:3])

        # Update calibrator
        for result in results:
            self.relevance_calibrator.record_feedback(result.relevance_score, feedback_type)

        # Update chunker
        for result in results:
            self.adaptive_chunker.learn_from_feedback(
                result.source,
                feedback_type,
                len(result.content)
            )

        # Periodic self-improvement
        if len(self.feedback_history) % 20 == 0:
            self._self_improve()

        self._save()

    def _self_improve(self):
        """Analyze feedback and make improvements"""
        if len(self.feedback_history) < 20:
            return

        recent = self.feedback_history[-50:]

        # Analyze failure patterns
        negative_queries = [f.query for f in recent if f.feedback_type in [FeedbackType.NEGATIVE, FeedbackType.IRRELEVANT]]
        positive_queries = [f.query for f in recent if f.feedback_type == FeedbackType.POSITIVE]

        # Find common patterns in failures
        if len(negative_queries) > 5:
            # Simple pattern: word frequency in failing queries
            failing_words = defaultdict(int)
            for q in negative_queries:
                for word in q.lower().split():
                    failing_words[word] += 1

            # Words that appear often in failures but not successes
            successful_words = set()
            for q in positive_queries:
                successful_words.update(q.lower().split())

            problematic = {w: c for w, c in failing_words.items()
                         if c >= 3 and w not in successful_words}

            if problematic:
                logger.info(f"Identified problematic query patterns: {problematic}")

        self.adaptations_made += 1

    def _query_hash(self, query: str) -> str:
        """Generate hash for query caching"""
        return hashlib.md5(query.lower().strip().encode()).hexdigest()[:12]

    def get_improvement_stats(self) -> Dict[str, Any]:
        """Get self-improvement statistics"""
        success_rate = self.successful_queries / self.total_queries if self.total_queries > 0 else 0

        return {
            'total_queries': self.total_queries,
            'successful_queries': self.successful_queries,
            'success_rate': success_rate,
            'feedback_entries': len(self.feedback_history),
            'adaptations_made': self.adaptations_made,
            'relevance_threshold': self.relevance_calibrator.threshold,
            'cached_patterns': len(self.query_success_cache),
            'query_patterns_learned': len(self.query_refiner.patterns)
        }

    def _save(self):
        """Save learning state"""
        state = {
            'relevance_threshold': self.relevance_calibrator.threshold,
            'query_patterns': {k: {
                'pattern': v.pattern,
                'success_rate': v.success_rate,
                'usage_count': v.usage_count
            } for k, v in self.query_refiner.patterns.items()},
            'chunk_adjustments': self.adaptive_chunker.adjustments,
            'query_cache': dict(list(self.query_success_cache.items())[-100:]),  # Keep recent
            'stats': {
                'total_queries': self.total_queries,
                'successful_queries': self.successful_queries,
                'adaptations_made': self.adaptations_made
            }
        }

        state_file = self.storage_path / "rag_learning.json"
        state_file.write_text(json.dumps(state, indent=2))

    def _load(self):
        """Load learning state"""
        state_file = self.storage_path / "rag_learning.json"
        if not state_file.exists():
            return

        try:
            state = json.loads(state_file.read_text())

            self.relevance_calibrator.threshold = state.get('relevance_threshold', 0.5)

            for k, v in state.get('query_patterns', {}).items():
                self.query_refiner.patterns[k] = QueryPattern(
                    pattern=v['pattern'],
                    success_rate=v['success_rate'],
                    usage_count=v['usage_count']
                )

            self.adaptive_chunker.adjustments = state.get('chunk_adjustments', {})
            self.query_success_cache = state.get('query_cache', {})

            stats = state.get('stats', {})
            self.total_queries = stats.get('total_queries', 0)
            self.successful_queries = stats.get('successful_queries', 0)
            self.adaptations_made = stats.get('adaptations_made', 0)

            logger.info(f"Loaded RAG learning state: {self.get_improvement_stats()}")

        except Exception as e:
            logger.error(f"Failed to load RAG state: {e}")


# Singleton instance
_rag_system: Optional[SelfImprovingRAG] = None


def get_self_improving_rag() -> SelfImprovingRAG:
    """Get or create singleton RAG system"""
    global _rag_system
    if _rag_system is None:
        _rag_system = SelfImprovingRAG()
    return _rag_system


if __name__ == "__main__":
    # Test the self-improving RAG
    rag = get_self_improving_rag()

    print("Self-Improving RAG Test")
    print("=" * 60)

    # Sample documents
    documents = [
        {'id': 'doc1', 'content': 'The API endpoint accepts POST requests with JSON body', 'source': 'api_docs'},
        {'id': 'doc2', 'content': 'Authentication uses JWT tokens in the Authorization header', 'source': 'auth_docs'},
        {'id': 'doc3', 'content': 'To fix the connection error, check your network settings', 'source': 'troubleshoot'},
        {'id': 'doc4', 'content': 'Database queries should use parameterized statements', 'source': 'db_docs'},
        {'id': 'doc5', 'content': 'The configuration file is located at /etc/app/config.yaml', 'source': 'config_docs'},
    ]

    # Test queries
    queries = [
        "How do I authenticate API requests?",
        "Where is the config file?",
        "Fix connection problems",
    ]

    for query in queries:
        print(f"\nQuery: {query}")
        results = rag.retrieve(query, documents, k=3)

        for i, r in enumerate(results, 1):
            print(f"  {i}. [{r.source}] {r.content[:60]}... (score: {r.relevance_score:.2f})")

        # Simulate positive feedback
        if results:
            rag.record_feedback(query, results, FeedbackType.POSITIVE, 0.8)

    print("\n" + "=" * 60)
    print("Improvement Stats:", rag.get_improvement_stats())
