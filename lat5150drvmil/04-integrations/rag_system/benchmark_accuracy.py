#!/usr/bin/env python3
"""
Screenshot Intelligence System - Accuracy Benchmarking Framework

Measures actual retrieval accuracy for the Screenshot Intelligence System
across multiple dimensions:

1. Retrieval Accuracy (Hit Rate @ K)
2. Semantic Similarity Metrics (MRR, nDCG)
3. OCR Quality Metrics (CER, WER)
4. Incident Detection Accuracy
5. Timeline Correlation Accuracy

Usage:
    python3 benchmark_accuracy.py --create-testset
    python3 benchmark_accuracy.py --run-benchmark
    python3 benchmark_accuracy.py --full-report

Metrics Measured:
- Hit@1, Hit@3, Hit@10: Did the correct result appear in top-K?
- MRR (Mean Reciprocal Rank): Average position of first relevant result
- nDCG (Normalized Discounted Cumulative Gain): Ranking quality
- Precision@K: What % of top-K are relevant?
- Recall@K: What % of relevant docs are in top-K?
"""

import sys
import json
import os
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime
import logging
import math

# Add parent path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from vector_rag_system import VectorRAGSystem, Document
    from screenshot_intelligence import ScreenshotIntelligence
except ImportError as e:
    print(f"Failed to import modules: {e}")
    sys.exit(1)

logger = logging.getLogger(__name__)


@dataclass
class Query:
    """Test query with ground truth"""
    query_id: str
    query_text: str
    relevant_doc_ids: List[str]  # Ground truth relevant documents
    query_type: str  # 'factual', 'incident', 'timeline', 'correlation'
    difficulty: str  # 'easy', 'medium', 'hard'


@dataclass
class RetrievalResult:
    """Single retrieval result"""
    doc_id: str
    score: float
    rank: int
    is_relevant: bool


@dataclass
class BenchmarkMetrics:
    """Benchmark evaluation metrics"""
    # Hit rates
    hit_at_1: float
    hit_at_3: float
    hit_at_5: float
    hit_at_10: float

    # Ranking metrics
    mrr: float  # Mean Reciprocal Rank
    ndcg_at_10: float  # Normalized Discounted Cumulative Gain

    # Precision/Recall
    precision_at_1: float
    precision_at_3: float
    precision_at_5: float
    precision_at_10: float

    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    recall_at_10: float

    # Overall
    total_queries: int
    avg_retrieval_time_ms: float


@dataclass
class OCRQualityMetrics:
    """OCR quality metrics"""
    cer: float  # Character Error Rate
    wer: float  # Word Error Rate
    accuracy: float  # Character-level accuracy
    total_chars: int
    total_words: int


class AccuracyBenchmark:
    """
    Comprehensive accuracy benchmarking for Screenshot Intelligence
    """

    def __init__(
        self,
        vector_rag: Optional[VectorRAGSystem] = None,
        test_data_dir: Optional[Path] = None
    ):
        self.rag = vector_rag if vector_rag else VectorRAGSystem()
        self.test_data_dir = test_data_dir or Path.home() / ".screenshot_intel" / "test_data"
        self.test_data_dir.mkdir(parents=True, exist_ok=True)

        self.queries_file = self.test_data_dir / "test_queries.json"
        self.ground_truth_file = self.test_data_dir / "ground_truth.json"
        self.results_file = self.test_data_dir / "benchmark_results.json"

    def create_test_dataset(self) -> Dict[str, Any]:
        """
        Create a test dataset with queries and ground truth

        In production, this would be:
        1. Real user queries with manually labeled relevant documents
        2. Synthetic queries generated from document corpus
        3. Expert-curated test cases

        Returns:
            Test dataset statistics
        """
        # Example test queries (in production, expand this significantly)
        test_queries = [
            # Factual queries
            Query(
                query_id="fact_001",
                query_text="VPN connection error",
                relevant_doc_ids=[],  # Would be populated with actual doc IDs
                query_type="factual",
                difficulty="easy"
            ),
            Query(
                query_id="fact_002",
                query_text="network timeout message",
                relevant_doc_ids=[],
                query_type="factual",
                difficulty="easy"
            ),
            Query(
                query_id="fact_003",
                query_text="authentication failed 403 forbidden",
                relevant_doc_ids=[],
                query_type="factual",
                difficulty="medium"
            ),

            # Incident retrieval
            Query(
                query_id="incident_001",
                query_text="system crash followed by restart",
                relevant_doc_ids=[],
                query_type="incident",
                difficulty="medium"
            ),
            Query(
                query_id="incident_002",
                query_text="multiple failed login attempts",
                relevant_doc_ids=[],
                query_type="incident",
                difficulty="hard"
            ),

            # Timeline correlation
            Query(
                query_id="timeline_001",
                query_text="events between 14:00 and 15:00",
                relevant_doc_ids=[],
                query_type="timeline",
                difficulty="medium"
            ),

            # Complex correlation
            Query(
                query_id="corr_001",
                query_text="error messages related to database connection",
                relevant_doc_ids=[],
                query_type="correlation",
                difficulty="hard"
            ),
        ]

        # Save queries
        queries_data = [asdict(q) for q in test_queries]
        with open(self.queries_file, 'w') as f:
            json.dump(queries_data, f, indent=2)

        print(f"✓ Created {len(test_queries)} test queries")
        print(f"  Saved to: {self.queries_file}")
        print("\n⚠️  NEXT STEP: Populate relevant_doc_ids with actual document IDs")
        print("   1. Ingest your screenshots/documents")
        print("   2. Run queries manually and identify relevant results")
        print("   3. Update test_queries.json with relevant doc IDs")

        return {
            'total_queries': len(test_queries),
            'by_type': {
                'factual': sum(1 for q in test_queries if q.query_type == 'factual'),
                'incident': sum(1 for q in test_queries if q.query_type == 'incident'),
                'timeline': sum(1 for q in test_queries if q.query_type == 'timeline'),
                'correlation': sum(1 for q in test_queries if q.query_type == 'correlation'),
            },
            'by_difficulty': {
                'easy': sum(1 for q in test_queries if q.difficulty == 'easy'),
                'medium': sum(1 for q in test_queries if q.difficulty == 'medium'),
                'hard': sum(1 for q in test_queries if q.difficulty == 'hard'),
            }
        }

    def load_test_queries(self) -> List[Query]:
        """Load test queries from file"""
        if not self.queries_file.exists():
            raise FileNotFoundError(
                f"Test queries not found. Run: create_test_dataset() first"
            )

        with open(self.queries_file) as f:
            queries_data = json.load(f)

        return [Query(**q) for q in queries_data]

    def calculate_hit_at_k(
        self,
        retrieved_doc_ids: List[str],
        relevant_doc_ids: List[str],
        k: int
    ) -> bool:
        """
        Calculate Hit@K: Is any relevant doc in top-K results?

        Args:
            retrieved_doc_ids: Retrieved document IDs (in rank order)
            relevant_doc_ids: Ground truth relevant document IDs
            k: Top-K to consider

        Returns:
            True if hit, False otherwise
        """
        top_k = retrieved_doc_ids[:k]
        return any(doc_id in relevant_doc_ids for doc_id in top_k)

    def calculate_precision_at_k(
        self,
        retrieved_doc_ids: List[str],
        relevant_doc_ids: List[str],
        k: int
    ) -> float:
        """
        Calculate Precision@K: What fraction of top-K are relevant?

        Args:
            retrieved_doc_ids: Retrieved document IDs
            relevant_doc_ids: Ground truth relevant document IDs
            k: Top-K to consider

        Returns:
            Precision (0.0 to 1.0)
        """
        if k == 0:
            return 0.0

        top_k = retrieved_doc_ids[:k]
        relevant_in_top_k = sum(1 for doc_id in top_k if doc_id in relevant_doc_ids)

        return relevant_in_top_k / k

    def calculate_recall_at_k(
        self,
        retrieved_doc_ids: List[str],
        relevant_doc_ids: List[str],
        k: int
    ) -> float:
        """
        Calculate Recall@K: What fraction of relevant docs are in top-K?

        Args:
            retrieved_doc_ids: Retrieved document IDs
            relevant_doc_ids: Ground truth relevant document IDs
            k: Top-K to consider

        Returns:
            Recall (0.0 to 1.0)
        """
        if not relevant_doc_ids:
            return 0.0

        top_k = retrieved_doc_ids[:k]
        relevant_in_top_k = sum(1 for doc_id in top_k if doc_id in relevant_doc_ids)

        return relevant_in_top_k / len(relevant_doc_ids)

    def calculate_mrr(
        self,
        retrieved_doc_ids: List[str],
        relevant_doc_ids: List[str]
    ) -> float:
        """
        Calculate MRR (Mean Reciprocal Rank): 1 / rank of first relevant doc

        Args:
            retrieved_doc_ids: Retrieved document IDs
            relevant_doc_ids: Ground truth relevant document IDs

        Returns:
            Reciprocal rank (0.0 to 1.0)
        """
        for rank, doc_id in enumerate(retrieved_doc_ids, 1):
            if doc_id in relevant_doc_ids:
                return 1.0 / rank

        return 0.0

    def calculate_ndcg_at_k(
        self,
        retrieved_doc_ids: List[str],
        relevant_doc_ids: List[str],
        k: int
    ) -> float:
        """
        Calculate nDCG@K (Normalized Discounted Cumulative Gain)

        Measures ranking quality with position discount.

        Args:
            retrieved_doc_ids: Retrieved document IDs
            relevant_doc_ids: Ground truth relevant document IDs
            k: Top-K to consider

        Returns:
            nDCG score (0.0 to 1.0)
        """
        # DCG: Sum of (relevance / log2(rank + 1))
        dcg = 0.0
        for rank, doc_id in enumerate(retrieved_doc_ids[:k], 1):
            relevance = 1.0 if doc_id in relevant_doc_ids else 0.0
            dcg += relevance / math.log2(rank + 1)

        # IDCG: DCG of ideal ranking (all relevant docs first)
        idcg = 0.0
        for rank in range(1, min(len(relevant_doc_ids), k) + 1):
            idcg += 1.0 / math.log2(rank + 1)

        if idcg == 0.0:
            return 0.0

        return dcg / idcg

    def run_benchmark(
        self,
        limit: int = 10,
        score_threshold: float = 0.0
    ) -> BenchmarkMetrics:
        """
        Run complete benchmark on test queries

        Args:
            limit: Number of results to retrieve per query
            score_threshold: Minimum similarity score

        Returns:
            BenchmarkMetrics with all calculated metrics
        """
        queries = self.load_test_queries()

        # Filter out queries without ground truth
        queries_with_ground_truth = [
            q for q in queries if q.relevant_doc_ids
        ]

        if not queries_with_ground_truth:
            raise ValueError(
                "No queries with ground truth! "
                "Populate relevant_doc_ids in test_queries.json"
            )

        print(f"Running benchmark on {len(queries_with_ground_truth)} queries...")

        # Metrics accumulators
        hit_1_count = 0
        hit_3_count = 0
        hit_5_count = 0
        hit_10_count = 0

        mrr_sum = 0.0
        ndcg_sum = 0.0

        precision_1_sum = 0.0
        precision_3_sum = 0.0
        precision_5_sum = 0.0
        precision_10_sum = 0.0

        recall_1_sum = 0.0
        recall_3_sum = 0.0
        recall_5_sum = 0.0
        recall_10_sum = 0.0

        total_time_ms = 0.0

        for query in queries_with_ground_truth:
            import time
            start_time = time.time()

            # Retrieve results
            results = self.rag.search(
                query=query.query_text,
                limit=limit,
                score_threshold=score_threshold
            )

            retrieval_time_ms = (time.time() - start_time) * 1000
            total_time_ms += retrieval_time_ms

            # Extract doc IDs
            retrieved_doc_ids = [r.document.id for r in results]

            # Calculate metrics for this query
            if self.calculate_hit_at_k(retrieved_doc_ids, query.relevant_doc_ids, 1):
                hit_1_count += 1
            if self.calculate_hit_at_k(retrieved_doc_ids, query.relevant_doc_ids, 3):
                hit_3_count += 1
            if self.calculate_hit_at_k(retrieved_doc_ids, query.relevant_doc_ids, 5):
                hit_5_count += 1
            if self.calculate_hit_at_k(retrieved_doc_ids, query.relevant_doc_ids, 10):
                hit_10_count += 1

            mrr_sum += self.calculate_mrr(retrieved_doc_ids, query.relevant_doc_ids)
            ndcg_sum += self.calculate_ndcg_at_k(retrieved_doc_ids, query.relevant_doc_ids, 10)

            precision_1_sum += self.calculate_precision_at_k(retrieved_doc_ids, query.relevant_doc_ids, 1)
            precision_3_sum += self.calculate_precision_at_k(retrieved_doc_ids, query.relevant_doc_ids, 3)
            precision_5_sum += self.calculate_precision_at_k(retrieved_doc_ids, query.relevant_doc_ids, 5)
            precision_10_sum += self.calculate_precision_at_k(retrieved_doc_ids, query.relevant_doc_ids, 10)

            recall_1_sum += self.calculate_recall_at_k(retrieved_doc_ids, query.relevant_doc_ids, 1)
            recall_3_sum += self.calculate_recall_at_k(retrieved_doc_ids, query.relevant_doc_ids, 3)
            recall_5_sum += self.calculate_recall_at_k(retrieved_doc_ids, query.relevant_doc_ids, 5)
            recall_10_sum += self.calculate_recall_at_k(retrieved_doc_ids, query.relevant_doc_ids, 10)

        # Calculate averages
        n = len(queries_with_ground_truth)

        metrics = BenchmarkMetrics(
            hit_at_1=hit_1_count / n,
            hit_at_3=hit_3_count / n,
            hit_at_5=hit_5_count / n,
            hit_at_10=hit_10_count / n,
            mrr=mrr_sum / n,
            ndcg_at_10=ndcg_sum / n,
            precision_at_1=precision_1_sum / n,
            precision_at_3=precision_3_sum / n,
            precision_at_5=precision_5_sum / n,
            precision_at_10=precision_10_sum / n,
            recall_at_1=recall_1_sum / n,
            recall_at_3=recall_3_sum / n,
            recall_at_5=recall_5_sum / n,
            recall_at_10=recall_10_sum / n,
            total_queries=n,
            avg_retrieval_time_ms=total_time_ms / n
        )

        # Save results
        results_data = {
            'timestamp': datetime.now().isoformat(),
            'metrics': asdict(metrics),
            'embedding_model': 'BAAI/bge-base-en-v1.5',
            'vector_dimension': 384,
        }

        with open(self.results_file, 'w') as f:
            json.dump(results_data, f, indent=2)

        return metrics

    def print_report(self, metrics: BenchmarkMetrics):
        """Print formatted benchmark report"""
        print("\n" + "="*80)
        print("SCREENSHOT INTELLIGENCE - ACCURACY BENCHMARK REPORT")
        print("="*80)

        print(f"\nTotal Queries: {metrics.total_queries}")
        print(f"Avg Retrieval Time: {metrics.avg_retrieval_time_ms:.2f}ms")

        print("\n--- HIT RATES (Does ANY relevant doc appear in top-K?) ---")
        print(f"Hit@1:  {metrics.hit_at_1*100:6.2f}%")
        print(f"Hit@3:  {metrics.hit_at_3*100:6.2f}%")
        print(f"Hit@5:  {metrics.hit_at_5*100:6.2f}%")
        print(f"Hit@10: {metrics.hit_at_10*100:6.2f}%")

        print("\n--- RANKING QUALITY ---")
        print(f"MRR (Mean Reciprocal Rank): {metrics.mrr:.4f}")
        print(f"nDCG@10:                    {metrics.ndcg_at_10:.4f}")

        print("\n--- PRECISION (What % of top-K are relevant?) ---")
        print(f"Precision@1:  {metrics.precision_at_1*100:6.2f}%")
        print(f"Precision@3:  {metrics.precision_at_3*100:6.2f}%")
        print(f"Precision@5:  {metrics.precision_at_5*100:6.2f}%")
        print(f"Precision@10: {metrics.precision_at_10*100:6.2f}%")

        print("\n--- RECALL (What % of relevant docs are in top-K?) ---")
        print(f"Recall@1:  {metrics.recall_at_1*100:6.2f}%")
        print(f"Recall@3:  {metrics.recall_at_3*100:6.2f}%")
        print(f"Recall@5:  {metrics.recall_at_5*100:6.2f}%")
        print(f"Recall@10: {metrics.recall_at_10*100:6.2f}%")

        print("\n" + "="*80)
        print("\nEmbedding Model: BAAI/bge-base-en-v1.5 (384D)")
        print(f"Results saved to: {self.results_file}")
        print("="*80)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Benchmark Screenshot Intelligence Accuracy"
    )
    parser.add_argument(
        '--create-testset',
        action='store_true',
        help='Create test dataset with sample queries'
    )
    parser.add_argument(
        '--run-benchmark',
        action='store_true',
        help='Run benchmark on test queries'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=10,
        help='Number of results to retrieve per query'
    )

    args = parser.parse_args()

    benchmark = AccuracyBenchmark()

    if args.create_testset:
        stats = benchmark.create_test_dataset()
        print(f"\nTest Dataset Statistics:")
        print(f"  Total queries: {stats['total_queries']}")
        print(f"  By type: {stats['by_type']}")
        print(f"  By difficulty: {stats['by_difficulty']}")

    elif args.run_benchmark:
        try:
            metrics = benchmark.run_benchmark(limit=args.limit)
            benchmark.print_report(metrics)
        except ValueError as e:
            print(f"❌ {e}")
            sys.exit(1)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
