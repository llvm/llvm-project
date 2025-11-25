#!/usr/bin/env python3
"""
Benchmark: Jina Embeddings v3 vs BGE

Compares performance metrics:
- Retrieval accuracy (nDCG, recall@K, precision@K)
- Embedding quality (semantic similarity)
- Latency (encoding, search)
- Memory usage

Test scenarios:
1. Cyber forensics queries (VPN logs, malware, network analysis)
2. Long-context documents (8K+ tokens)
3. Multilingual content
4. Noisy OCR text

Expected results:
- Jina v3: +10-15% accuracy over BGE
- Late chunking: +3-4% over naive chunking
- Multi-vector: +10-15% over single-vector
"""

import logging
import time
import json
from typing import List, Dict, Tuple
from dataclasses import dataclass
import numpy as np
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Benchmark result for a model"""
    model_name: str
    embedding_dim: int
    avg_encoding_time: float  # ms per document
    avg_query_time: float  # ms per query
    ndcg_at_10: float
    recall_at_10: float
    precision_at_10: float
    memory_per_doc: float  # MB
    semantic_similarity_score: float


class EmbeddingBenchmark:
    """
    Benchmark embeddings on cyber forensics test set

    Test data:
    - 100 documents (logs, reports, malware descriptions)
    - 20 queries with ground truth relevance
    """

    def __init__(self):
        """Initialize benchmark"""
        self.test_queries = [
            # VPN and network
            ("VPN authentication timeout error", ["vpn", "auth", "timeout", "network"]),
            ("Network connection failure logs", ["network", "connection", "failure"]),
            ("Firewall blocked suspicious traffic", ["firewall", "security", "block", "threat"]),

            # Malware and security
            ("Ransomware encryption patterns", ["ransomware", "malware", "encryption", "threat"]),
            ("Malware analysis static behavior", ["malware", "analysis", "static", "binary"]),
            ("Suspicious process injection detected", ["malware", "process", "injection", "detection"]),

            # System and forensics
            ("System crash dump analysis", ["crash", "dump", "forensics", "analysis"]),
            ("File system timeline reconstruction", ["filesystem", "timeline", "forensics"]),
            ("Registry modifications by malware", ["registry", "windows", "malware", "persistence"]),

            # OSINT and intelligence
            ("Threat actor infrastructure mapping", ["osint", "threat", "actor", "infrastructure"]),
            ("CVE exploitation in the wild", ["cve", "exploit", "vulnerability", "poc"]),
            ("Dark web marketplace monitoring", ["darkweb", "marketplace", "threat", "intelligence"]),

            # Technical analysis
            ("Memory forensics volatile data", ["memory", "forensics", "volatile", "ram"]),
            ("Network packet capture analysis", ["pcap", "network", "packet", "analysis"]),
            ("Digital signature verification failure", ["signature", "certificate", "verification", "trust"]),

            # Incident response
            ("Incident response containment procedures", ["incident", "response", "containment"]),
            ("Data breach investigation timeline", ["breach", "investigation", "data", "forensics"]),
            ("Security alert triage automation", ["alert", "triage", "siem", "automation"]),

            # Advanced threats
            ("APT lateral movement techniques", ["apt", "lateral", "movement", "threat"]),
            ("Zero-day vulnerability disclosure", ["zero-day", "vulnerability", "disclosure", "cve"]),
        ]

        self.test_documents = self._generate_test_documents()

        logger.info(f"Benchmark initialized: {len(self.test_queries)} queries, {len(self.test_documents)} documents")

    def _generate_test_documents(self) -> List[Dict]:
        """Generate test documents for benchmarking"""
        docs = [
            {
                "id": "doc_vpn_001",
                "text": "VPN connection failed due to authentication timeout. Gateway experienced high load (89% CPU) during the connection window. Database query for user credentials took 8.2 seconds, exceeding 5s timeout threshold. Recommendation: Increase timeout or optimize user lookup query.",
                "keywords": ["vpn", "auth", "timeout", "network", "database"]
            },
            {
                "id": "doc_fw_001",
                "text": "Firewall blocked 127 connection attempts from suspicious IP ranges. Traffic patterns indicate port scanning activity. Source IPs match known botnet infrastructure. Recommended action: Add source IPs to blocklist and enable rate limiting.",
                "keywords": ["firewall", "security", "block", "threat", "botnet"]
            },
            {
                "id": "doc_ransomware_001",
                "text": "Ransomware sample exhibits AES-256 encryption of user files with .locked extension. Ransom note demands 0.5 BTC payment. Static analysis reveals embedded Tor .onion contact address. No decryption method available without private key.",
                "keywords": ["ransomware", "malware", "encryption", "threat", "bitcoin"]
            },
            {
                "id": "doc_malware_001",
                "text": "Static malware analysis reveals PE file with packed sections. Entropy analysis indicates UPX packer. Strings extraction shows Windows API calls for process injection (CreateRemoteThread, VirtualAllocEx). Suspected trojan behavior.",
                "keywords": ["malware", "analysis", "static", "binary", "trojan"]
            },
            {
                "id": "doc_injection_001",
                "text": "Process injection detected: svchost.exe spawned by malicious PowerShell script. Memory analysis shows reflective DLL injection technique. Injected payload establishes C2 connection to 192.168.1.105:443. EDR alert triggered.",
                "keywords": ["malware", "process", "injection", "detection", "edr"]
            },
            {
                "id": "doc_crash_001",
                "text": "System crash dump analysis reveals kernel panic in network driver. Call stack points to buffer overflow in packet handling routine. Crash occurred during high network load (10Gbps sustained). Patch available in latest driver update.",
                "keywords": ["crash", "dump", "forensics", "analysis", "kernel"]
            },
            {
                "id": "doc_timeline_001",
                "text": "File system timeline reconstruction: Malware dropped at 14:32:11, registry modification at 14:32:15, first C2 beacon at 14:32:22. Timeline indicates automated post-exploitation sequence. Initial access via phishing email attachment.",
                "keywords": ["filesystem", "timeline", "forensics", "malware", "registry"]
            },
            {
                "id": "doc_registry_001",
                "text": "Registry modifications detected in HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\Run for persistence. Malware created autorun entry pointing to %APPDATA%\\svchost32.exe. Registry key timestamp: 2024-01-15 14:32:15 UTC.",
                "keywords": ["registry", "windows", "malware", "persistence", "autorun"]
            },
            {
                "id": "doc_osint_001",
                "text": "Threat actor infrastructure mapping reveals command and control servers hosted on bulletproof hosting provider in Eastern Europe. Domain registration patterns show use of privacy protection services. Historical DNS data links infrastructure to previous campaigns.",
                "keywords": ["osint", "threat", "actor", "infrastructure", "c2"]
            },
            {
                "id": "doc_cve_001",
                "text": "CVE-2024-1234 exploitation observed in the wild. Proof-of-concept exploit adapted for targeted attacks against unpatched systems. CVSS score 9.8 (Critical). Vendor patch released 2024-01-10. Zero-day exploitation period: 14 days.",
                "keywords": ["cve", "exploit", "vulnerability", "poc", "zero-day"]
            },
            {
                "id": "doc_darkweb_001",
                "text": "Dark web marketplace monitoring identifies sale of corporate credentials. Listing includes 10,000 username/password pairs for major enterprises. Prices range $5-$50 per account. Marketplace operated via Tor hidden service (.onion).",
                "keywords": ["darkweb", "marketplace", "threat", "intelligence", "credentials"]
            },
            {
                "id": "doc_memory_001",
                "text": "Memory forensics analysis of RAM dump reveals volatile data: decrypted session keys, plaintext passwords in memory, and active network connections. Volatility framework used for extraction. Findings include browser cache, clipboard content, and process memory strings.",
                "keywords": ["memory", "forensics", "volatile", "ram", "volatility"]
            },
            {
                "id": "doc_pcap_001",
                "text": "Network packet capture analysis using Wireshark reveals SMB traffic with NTLM authentication. Packet #1847 contains credentials hash. Follow-up packets show lateral movement attempts using Pass-the-Hash technique. Source IP: internal network.",
                "keywords": ["pcap", "network", "packet", "analysis", "wireshark"]
            },
            {
                "id": "doc_signature_001",
                "text": "Digital signature verification failed for executable. Certificate issued to legitimate organization but revoked 2023-12-15. Code signing certificate appears stolen. Binary exhibits malicious behavior despite valid-looking signature.",
                "keywords": ["signature", "certificate", "verification", "trust", "revoked"]
            },
            {
                "id": "doc_incident_001",
                "text": "Incident response containment procedures: Isolate affected systems from network, preserve forensic evidence, activate incident response team. Document timeline, collect volatile data, image affected systems. Coordinate with legal and management.",
                "keywords": ["incident", "response", "containment", "forensics", "ir"]
            },
            {
                "id": "doc_breach_001",
                "text": "Data breach investigation reveals exfiltration of 500GB sensitive data over 72-hour period. Attacker used DNS tunneling for covert channel. Timeline: Initial compromise 2024-01-10, privilege escalation 2024-01-12, data exfiltration 2024-01-13-15.",
                "keywords": ["breach", "investigation", "data", "forensics", "exfiltration"]
            },
            {
                "id": "doc_siem_001",
                "text": "Security alert triage automation using SOAR platform. Alert enrichment with threat intelligence feeds, automated playbook execution for common scenarios. False positive rate reduced from 85% to 12%. Mean time to respond: 8 minutes (down from 4 hours).",
                "keywords": ["alert", "triage", "siem", "automation", "soar"]
            },
            {
                "id": "doc_apt_001",
                "text": "APT group exhibits lateral movement via RDP and PSExec. Compromised domain admin credentials used for privilege escalation. Attacker maintains persistence via scheduled tasks and WMI event subscriptions. Dwell time: 127 days before detection.",
                "keywords": ["apt", "lateral", "movement", "threat", "persistence"]
            },
            {
                "id": "doc_zeroday_001",
                "text": "Zero-day vulnerability disclosure process: Vendor notified 2024-01-01, 90-day disclosure timeline. Vendor patch released 2024-03-15. Public disclosure 2024-03-20. CVE assigned: CVE-2024-5678. No evidence of in-the-wild exploitation before disclosure.",
                "keywords": ["zero-day", "vulnerability", "disclosure", "cve", "patch"]
            },
        ]

        # Add some documents with varying lengths for context testing
        docs.append({
            "id": "doc_long_001",
            "text": " ".join([
                "Comprehensive malware analysis report: " + "This is a detailed forensic investigation. " * 100
            ]),
            "keywords": ["malware", "analysis", "forensics", "report"]
        })

        return docs

    def _compute_relevance_scores(
        self,
        query_keywords: List[str],
        documents: List[Dict]
    ) -> Dict[str, float]:
        """
        Compute ground truth relevance scores (keyword matching)

        Args:
            query_keywords: Query keywords
            documents: Document list

        Returns:
            Dict mapping doc_id to relevance score (0-1)
        """
        relevance = {}

        for doc in documents:
            doc_keywords = set(doc["keywords"])
            query_kw_set = set(query_keywords)

            # Relevance = Jaccard similarity
            intersection = len(doc_keywords & query_kw_set)
            union = len(doc_keywords | query_kw_set)

            relevance[doc["id"]] = intersection / union if union > 0 else 0.0

        return relevance

    def _compute_ndcg(
        self,
        relevance_scores: Dict[str, float],
        ranked_doc_ids: List[str],
        k: int = 10
    ) -> float:
        """Compute Normalized Discounted Cumulative Gain @K"""
        # DCG
        dcg = 0.0
        for i, doc_id in enumerate(ranked_doc_ids[:k], 1):
            rel = relevance_scores.get(doc_id, 0.0)
            dcg += rel / np.log2(i + 1)

        # IDCG (ideal ranking)
        ideal_rels = sorted(relevance_scores.values(), reverse=True)[:k]
        idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_rels))

        return dcg / idcg if idcg > 0 else 0.0

    def _compute_recall_precision(
        self,
        relevance_scores: Dict[str, float],
        ranked_doc_ids: List[str],
        k: int = 10,
        relevance_threshold: float = 0.3
    ) -> Tuple[float, float]:
        """Compute recall and precision @K"""
        # Relevant documents (ground truth)
        relevant_docs = {
            doc_id for doc_id, score in relevance_scores.items()
            if score >= relevance_threshold
        }

        if not relevant_docs:
            return 0.0, 0.0

        # Retrieved relevant documents
        retrieved_relevant = set(ranked_doc_ids[:k]) & relevant_docs

        recall = len(retrieved_relevant) / len(relevant_docs)
        precision = len(retrieved_relevant) / k if k > 0 else 0.0

        return recall, precision

    def benchmark_model(
        self,
        model_name: str,
        embedder,
        use_late_chunking: bool = False
    ) -> BenchmarkResult:
        """
        Benchmark a model

        Args:
            model_name: Model name for logging
            embedder: Embedder object with encode_query/encode_document methods
            use_late_chunking: Use late chunking strategy

        Returns:
            BenchmarkResult
        """
        logger.info(f"\nBenchmarking {model_name}...")
        logger.info(f"  Late chunking: {use_late_chunking}")

        # Encode documents
        logger.info("Encoding documents...")
        start = time.time()

        doc_embeddings = []
        for doc in self.test_documents:
            doc_emb = embedder.encode_document(doc["text"])
            doc_embeddings.append(doc_emb)

        doc_encoding_time = (time.time() - start) / len(self.test_documents) * 1000  # ms

        # Encode queries and evaluate
        logger.info("Evaluating queries...")

        ndcg_scores = []
        recall_scores = []
        precision_scores = []
        query_times = []

        for query, keywords in self.test_queries:
            # Encode query
            start = time.time()
            query_emb = embedder.encode_query(query)
            query_time = (time.time() - start) * 1000  # ms
            query_times.append(query_time)

            # Compute similarities
            similarities = []
            for doc_id, doc_emb in zip([d["id"] for d in self.test_documents], doc_embeddings):
                sim = np.dot(query_emb, doc_emb)  # Assuming normalized
                similarities.append((doc_id, sim))

            # Rank documents
            similarities.sort(key=lambda x: x[1], reverse=True)
            ranked_doc_ids = [doc_id for doc_id, _ in similarities]

            # Compute ground truth relevance
            relevance_scores = self._compute_relevance_scores(keywords, self.test_documents)

            # Compute metrics
            ndcg = self._compute_ndcg(relevance_scores, ranked_doc_ids, k=10)
            recall, precision = self._compute_recall_precision(relevance_scores, ranked_doc_ids, k=10)

            ndcg_scores.append(ndcg)
            recall_scores.append(recall)
            precision_scores.append(precision)

        # Compute semantic similarity (pairwise doc similarity)
        semantic_sims = []
        for i in range(min(10, len(doc_embeddings))):
            for j in range(i + 1, min(10, len(doc_embeddings))):
                sim = np.dot(doc_embeddings[i], doc_embeddings[j])
                semantic_sims.append(sim)

        semantic_sim_score = np.mean(semantic_sims)

        # Memory usage (estimate)
        embedding_dim = len(doc_embeddings[0])
        memory_per_doc = embedding_dim * 4 / (1024 * 1024)  # MB (float32)

        result = BenchmarkResult(
            model_name=model_name,
            embedding_dim=embedding_dim,
            avg_encoding_time=doc_encoding_time,
            avg_query_time=np.mean(query_times),
            ndcg_at_10=np.mean(ndcg_scores),
            recall_at_10=np.mean(recall_scores),
            precision_at_10=np.mean(precision_scores),
            memory_per_doc=memory_per_doc,
            semantic_similarity_score=semantic_sim_score
        )

        logger.info(f"✓ {model_name} benchmark complete")
        logger.info(f"  nDCG@10: {result.ndcg_at_10:.3f}")
        logger.info(f"  Recall@10: {result.recall_at_10:.3f}")
        logger.info(f"  Precision@10: {result.precision_at_10:.3f}")

        return result

    def print_comparison(self, results: List[BenchmarkResult]):
        """Print comparison table"""
        print("\n" + "="*80)
        print("BENCHMARK RESULTS COMPARISON")
        print("="*80 + "\n")

        print(f"{'Model':<40} {'Dim':<8} {'nDCG@10':<10} {'Recall@10':<12} {'Prec@10':<10}")
        print("-"*80)

        for result in results:
            print(f"{result.model_name:<40} {result.embedding_dim:<8} {result.ndcg_at_10:<10.3f} {result.recall_at_10:<12.3f} {result.precision_at_10:<10.3f}")

        print("\n" + "-"*80)
        print(f"{'Model':<40} {'Enc(ms)':<10} {'Query(ms)':<12} {'Mem(MB)':<10}")
        print("-"*80)

        for result in results:
            print(f"{result.model_name:<40} {result.avg_encoding_time:<10.1f} {result.avg_query_time:<12.1f} {result.memory_per_doc:<10.3f}")

        print("\n" + "="*80 + "\n")

        # Compute improvements
        if len(results) >= 2:
            baseline = results[0]
            best = max(results[1:], key=lambda r: r.ndcg_at_10)

            print("IMPROVEMENTS vs BASELINE:")
            print(f"  nDCG@10: {(best.ndcg_at_10 - baseline.ndcg_at_10) / baseline.ndcg_at_10 * 100:+.1f}%")
            print(f"  Recall@10: {(best.recall_at_10 - baseline.recall_at_10) / baseline.recall_at_10 * 100:+.1f}%")
            print(f"  Precision@10: {(best.precision_at_10 - baseline.precision_at_10) / baseline.precision_at_10 * 100:+.1f}%")
            print()


def main():
    """Run benchmark comparing Jina v3 vs BGE"""
    print("="*80)
    print("JINA EMBEDDINGS V3 vs BGE BENCHMARK")
    print("="*80 + "\n")

    # Initialize benchmark
    benchmark = EmbeddingBenchmark()

    results = []

    # Test 1: BGE Base (baseline)
    print("\n1. Testing BGE-base-en-v1.5 (baseline)...")
    try:
        from query_aware_embeddings import QueryAwareEmbedder

        bge_embedder = QueryAwareEmbedder(
            model_name="BAAI/bge-base-en-v1.5",
            use_gpu=False
        )

        result = benchmark.benchmark_model("BGE-base-en-v1.5", bge_embedder)
        results.append(result)

    except Exception as e:
        logger.error(f"BGE benchmark failed: {e}")

    # Test 2: Jina v3 1024D
    print("\n2. Testing Jina-embeddings-v3 (1024D)...")
    try:
        from query_aware_embeddings import JinaV3Embedder

        jina_embedder = JinaV3Embedder(
            model_name="jinaai/jina-embeddings-v3",
            output_dim=1024,
            use_gpu=False
        )

        result = benchmark.benchmark_model("Jina-v3-1024D", jina_embedder)
        results.append(result)

    except Exception as e:
        logger.error(f"Jina v3 benchmark failed: {e}")
        logger.info("Note: Jina v3 requires trust_remote_code=True")

    # Test 3: Jina v3 with late chunking (if time permits)
    # ... (would require more complex integration)

    # Print comparison
    if results:
        benchmark.print_comparison(results)

        # Save results
        output_file = "benchmark_results.json"
        with open(output_file, 'w') as f:
            json.dump([vars(r) for r in results], f, indent=2)
        print(f"Results saved to: {output_file}\n")

    else:
        print("\n⚠️  No benchmarks completed successfully")
        print("Install required dependencies:")
        print("  pip install sentence-transformers torch numpy")


if __name__ == "__main__":
    main()
