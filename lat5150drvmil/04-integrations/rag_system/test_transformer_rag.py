#!/usr/bin/env python3
"""
Test Transformer RAG vs TF-IDF baseline
Validates accuracy improvement
"""

import time
from transformer_query import TransformerRAG


def run_transformer_tests():
    """Run same test suite with transformer embeddings"""

    print("="*70)
    print("LAT5150DRVMIL RAG - Transformer vs TF-IDF Comparison")
    print("="*70)
    print()

    # Initialize transformer RAG
    rag = TransformerRAG()

    # Same test cases as original
    test_cases = [
        {
            'query': 'What is DSMIL activation?',
            'expected_keywords': ['dsmil', 'activation', 'military', 'token'],
        },
        {
            'query': 'How to enable NPU modules?',
            'expected_keywords': ['npu', 'module', 'enable', 'kernel'],
        },
        {
            'query': 'APT41 security features?',
            'expected_keywords': ['apt41', 'security', 'hardening', 'tpm'],
        },
        {
            'query': 'Kernel build process steps?',
            'expected_keywords': ['kernel', 'build', 'make', 'compile'],
        },
        {
            'query': 'What is the unified platform architecture?',
            'expected_keywords': ['unified', 'platform', 'architecture', 'system'],
        },
        {
            'query': 'ZFS upgrade procedure?',
            'expected_keywords': ['zfs', 'upgrade', 'inplace'],
        },
        {
            'query': 'VAULT7 defense matrix?',
            'expected_keywords': ['vault7', 'defense', 'matrix', 'security'],
        },
        {
            'query': 'Claude local AI setup?',
            'expected_keywords': ['claude', 'local', 'ai', 'setup'],
        },
        {
            'query': 'DSMIL agent coordination?',
            'expected_keywords': ['dsmil', 'agent', 'coordination', 'team'],
        },
        {
            'query': 'What are the current system capabilities?',
            'expected_keywords': ['system', 'capabilities', 'features'],
        },
    ]

    print(f"Running {len(test_cases)} test cases...\n")

    results = []
    total_time = 0

    for i, test in enumerate(test_cases, 1):
        start_time = time.time()

        # Get results
        search_results = rag.retriever.search(test['query'], top_k=3)
        elapsed = time.time() - start_time
        total_time += elapsed

        # Calculate accuracy
        score = 0
        max_score = len(test['expected_keywords'])

        for result, similarity in search_results:
            text_lower = result['text'].lower()
            for keyword in test['expected_keywords']:
                if keyword.lower() in text_lower:
                    score += 1

        accuracy = (score / max_score * 100) if max_score > 0 else 0

        results.append({
            'query': test['query'],
            'accuracy': accuracy,
            'time': elapsed,
            'top_score': search_results[0][1] if search_results else 0
        })

        # Display
        status = "✓ PASS" if accuracy >= 70 else ("~ FAIR" if accuracy >= 50 else "✗ FAIL")
        print(f"[{i}/{len(test_cases)}] {status} | Accuracy: {accuracy:.1f}% | "
              f"Similarity: {search_results[0][1]:.3f} | Time: {elapsed:.3f}s")
        print(f"    Query: {test['query']}")
        print()

    # Summary
    print("="*70)
    print("Transformer RAG Test Summary")
    print("="*70)

    accuracies = [r['accuracy'] for r in results]
    times = [r['time'] for r in results]

    avg_accuracy = sum(accuracies) / len(accuracies)
    avg_time = sum(times) / len(times)
    passed_70 = sum(1 for a in accuracies if a >= 70)
    passed_50 = sum(1 for a in accuracies if a >= 50)

    print(f"\nTotal Tests: {len(test_cases)}")
    print(f"Passed (≥70%): {passed_70} ({passed_70/len(test_cases)*100:.1f}%)")
    print(f"Passed (≥50%): {passed_50} ({passed_50/len(test_cases)*100:.1f}%)")

    print(f"\nAverage Accuracy: {avg_accuracy:.1f}%")
    print(f"Average Response Time: {avg_time:.3f}s")
    print(f"Total Time: {total_time:.3f}s")

    print("\n" + "="*70)
    print("COMPARISON WITH BASELINE")
    print("="*70)

    print(f"\nTF-IDF Baseline:      51.8% accuracy, 2.5s avg response")
    print(f"Transformer RAG:      {avg_accuracy:.1f}% accuracy, {avg_time:.3f}s avg response")
    print(f"Improvement:          {avg_accuracy - 51.8:+.1f}% accuracy")

    if avg_accuracy >= 88:
        print(f"\n🎉 EXCELLENT! Research target (88%) achieved!")
    elif avg_accuracy >= 75:
        print(f"\n✓ GOOD! Significant improvement over baseline")
    elif avg_accuracy >= 60:
        print(f"\n~ FAIR. Some improvement, consider fine-tuning")
    else:
        print(f"\n⚠ Limited improvement. Check embeddings and queries")

    print("="*70)


if __name__ == '__main__':
    run_transformer_tests()
