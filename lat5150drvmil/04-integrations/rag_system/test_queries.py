#!/usr/bin/env python3
"""
Test suite for LAT5150DRVMIL RAG system
Validates accuracy and performance
"""

import time
from rag_query import LAT5150RAG


class RAGTester:
    """Test RAG system accuracy and performance"""

    def __init__(self):
        self.rag = LAT5150RAG()
        self.test_cases = self._load_test_cases()

    def _load_test_cases(self):
        """Define test cases with expected keywords"""
        return [
            {
                'query': 'What is DSMIL activation?',
                'expected_keywords': ['dsmil', 'activation', 'military', 'token'],
                'expected_files': ['MILITARY_TOKEN_ACTIVATION'],
            },
            {
                'query': 'How to enable NPU modules?',
                'expected_keywords': ['npu', 'module', 'enable', 'kernel'],
                'expected_files': ['NPU_MODULES'],
            },
            {
                'query': 'APT41 security features?',
                'expected_keywords': ['apt41', 'security', 'hardening', 'tpm'],
                'expected_files': ['APT41_SECURITY'],
            },
            {
                'query': 'Kernel build process steps?',
                'expected_keywords': ['kernel', 'build', 'make', 'compile'],
                'expected_files': ['KERNEL_BUILD'],
            },
            {
                'query': 'What is the unified platform architecture?',
                'expected_keywords': ['unified', 'platform', 'architecture', 'system'],
                'expected_files': ['UNIFIED_PLATFORM'],
            },
            {
                'query': 'ZFS upgrade procedure?',
                'expected_keywords': ['zfs', 'upgrade', 'inplace'],
                'expected_files': ['ZFS_INPLACE'],
            },
            {
                'query': 'VAULT7 defense matrix?',
                'expected_keywords': ['vault7', 'defense', 'matrix', 'security'],
                'expected_files': ['VAULT7'],
            },
            {
                'query': 'Claude local AI setup?',
                'expected_keywords': ['claude', 'local', 'ai', 'setup'],
                'expected_files': ['CLAUDE_LOCAL', 'SETUP_LOCAL_AI'],
            },
            {
                'query': 'DSMIL agent coordination?',
                'expected_keywords': ['dsmil', 'agent', 'coordination', 'team'],
                'expected_files': ['AGENT_COORDINATION', 'DSMIL-AGENT'],
            },
            {
                'query': 'What are the current system capabilities?',
                'expected_keywords': ['system', 'capabilities', 'features'],
                'expected_files': ['SYSTEM_CAPABILITIES'],
            },
        ]

    def test_relevance(self, test_case):
        """Test if retrieved results are relevant"""
        results = self.rag.retriever.search(test_case['query'], top_k=3)

        score = 0
        max_score = 0

        # Check keyword matches
        for result, relevance in results:
            text_lower = result['text'].lower()
            for keyword in test_case['expected_keywords']:
                max_score += 1
                if keyword.lower() in text_lower:
                    score += 1

        # Check file matches
        for result, relevance in results:
            filename = result['metadata']['filepath']
            for expected_file in test_case['expected_files']:
                max_score += 1
                if expected_file.lower() in filename.lower():
                    score += 1

        accuracy = (score / max_score * 100) if max_score > 0 else 0
        return accuracy, results

    def run_all_tests(self):
        """Run all test cases and report results"""
        print("="*70)
        print("LAT5150DRVMIL RAG System - Test Suite")
        print("="*70)
        print(f"\nRunning {len(self.test_cases)} test cases...\n")

        results_summary = []
        total_time = 0

        for i, test_case in enumerate(self.test_cases, 1):
            start_time = time.time()
            accuracy, results = self.test_relevance(test_case)
            elapsed = time.time() - start_time
            total_time += elapsed

            results_summary.append({
                'query': test_case['query'],
                'accuracy': accuracy,
                'time': elapsed
            })

            # Display result
            status = "✓ PASS" if accuracy >= 50 else "✗ FAIL"
            print(f"[{i}/{len(self.test_cases)}] {status} | Accuracy: {accuracy:.1f}% | Time: {elapsed:.3f}s")
            print(f"    Query: {test_case['query']}")

            if accuracy < 50:
                print(f"    Expected keywords: {test_case['expected_keywords']}")
                print(f"    Top result: {results[0][0]['metadata']['filepath']}")

            print()

        # Summary statistics
        print("="*70)
        print("Test Summary")
        print("="*70)

        accuracies = [r['accuracy'] for r in results_summary]
        times = [r['time'] for r in results_summary]

        avg_accuracy = sum(accuracies) / len(accuracies)
        avg_time = sum(times) / len(times)
        passed = sum(1 for a in accuracies if a >= 50)

        print(f"\nTotal Tests: {len(self.test_cases)}")
        print(f"Passed: {passed} ({passed/len(self.test_cases)*100:.1f}%)")
        print(f"Failed: {len(self.test_cases) - passed}")
        print(f"\nAverage Accuracy: {avg_accuracy:.1f}%")
        print(f"Average Response Time: {avg_time:.3f}s")
        print(f"Total Time: {total_time:.3f}s")

        print("\n" + "="*70)

        # Performance rating
        if avg_accuracy >= 88:
            rating = "EXCELLENT (Research target achieved!)"
        elif avg_accuracy >= 75:
            rating = "GOOD (Above baseline)"
        elif avg_accuracy >= 60:
            rating = "FAIR (Needs improvement)"
        else:
            rating = "POOR (Requires optimization)"

        print(f"Performance Rating: {rating}")
        print(f"Target: >88% (from Maharana et al. research)")
        print("="*70)

        return results_summary


if __name__ == '__main__':
    tester = RAGTester()
    tester.run_all_tests()
