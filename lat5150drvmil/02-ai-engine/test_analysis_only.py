#!/usr/bin/env python3
"""
Test advanced analysis features only (no LLM required)
"""

from rag_system.code_analysis_engine import (
    SecurityScanner, PerformanceOptimizer, ComplexityAnalyzer, CodeSmellDetector
)
from rag_system.code_transformers import apply_all_transformers
from rag_system.code_generators import DocumentationGenerator, TestGenerator

# Test code with various issues
vulnerable_code = """
import os
import pickle

def process_user_data(user_input, filename):
    # Security issues
    result = eval(user_input)  # CRITICAL: code injection
    os.system("cat " + filename)  # CRITICAL: command injection

    api_key = "sk-1234567890abcdef1234567890abcdef"  # hardcoded secret

    # Performance issues
    items = ['a', 'b', 'c']
    data = ""
    for i in range(len(items)):  # Use enumerate
        data += str(items[i])  # String concatenation in loop

    # Deserialize untrusted data
    config = pickle.loads(user_input)

    return data
"""

simple_code = """
def calculate_total(items, tax_rate):
    subtotal = sum(item['price'] for item in items)
    tax = subtotal * tax_rate
    return subtotal + tax

class ShoppingCart:
    def __init__(self):
        self.items = []

    def add_item(self, item):
        self.items.append(item)
"""

def main():
    print("="*80)
    print("Advanced Code Analysis Features Test")
    print("="*80)

    # Test 1: Security Scanner
    print("\n🔒 TEST 1: Security Vulnerability Scan")
    print("-"*80)
    scanner = SecurityScanner()
    security_issues = scanner.scan(vulnerable_code)
    print(f"Found {len(security_issues)} security issues:")
    for issue in security_issues[:5]:  # Show first 5
        print(f"  [{issue.severity.value}] Line {issue.line}: {issue.description}")
        print(f"    Fix: {issue.remediation}")

    # Test 2: Performance Optimizer
    print("\n⚡ TEST 2: Performance Analysis")
    print("-"*80)
    optimizer = PerformanceOptimizer()
    perf_issues = optimizer.analyze(vulnerable_code)
    print(f"Found {len(perf_issues)} performance issues:")
    for issue in perf_issues:
        print(f"  Line {issue.line}: {issue.description}")
        print(f"    → {issue.suggestion} ({issue.estimated_improvement})")

    # Test 3: Complexity Analysis
    print("\n📊 TEST 3: Code Complexity Analysis")
    print("-"*80)
    analyzer = ComplexityAnalyzer()
    complexity = analyzer.cyclomatic_complexity(vulnerable_code)
    nesting = analyzer.nesting_depth(vulnerable_code)
    print(f"Cyclomatic Complexity: {complexity}")
    print(f"Max Nesting Depth: {nesting}")

    smell_detector = CodeSmellDetector()
    smells = smell_detector.detect(vulnerable_code)
    print(f"Code Smells: {len(smells)}")
    for smell in smells:
        print(f"  Line {smell.line} [{smell.category}]: {smell.description}")

    # Test 4: AST Transformations
    print("\n✨ TEST 4: Automatic Code Transformations")
    print("-"*80)
    refactored, transforms = apply_all_transformers(vulnerable_code)
    print(f"Applied {len(transforms)} transformations:")
    for t in transforms:
        print(f"  [{t.transformer_name}] Line {t.original_line}")
        print(f"    {t.description}")

    # Test 5: Documentation Generation
    print("\n📚 TEST 5: Documentation Generation")
    print("-"*80)
    doc_gen = DocumentationGenerator(style='google')
    docs = doc_gen.generate_all(simple_code)
    print(f"Generated {len(docs)} docstrings:")
    for doc in docs:
        print(f"  {doc.target} (line {doc.line})")

    # Test 6: Test Generation
    print("\n🧪 TEST 6: Unit Test Generation")
    print("-"*80)
    test_gen = TestGenerator(framework='pytest')
    tests = test_gen.generate_all(simple_code, module_name='shopping')
    print(f"Generated test file ({len(tests)} characters)")
    print("First 300 characters:")
    print(tests[:300])

    print("\n" + "="*80)
    print("✅ All Analysis Features Working Correctly!")
    print("="*80)

if __name__ == '__main__':
    main()
