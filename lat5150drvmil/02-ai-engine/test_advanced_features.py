#!/usr/bin/env python3
"""
Test script for advanced code assistant features
"""

from rag_system.code_assistant import CodeAssistant

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
    data = ""
    for i in range(len(items)):  # Use enumerate
        data += str(items[i])  # String concatenation in loop

    # Deserialize untrusted data
    config = pickle.loads(user_input)

    return data
"""

def main():
    print("="*70)
    print("Testing Advanced Code Assistant Features")
    print("="*70)

    assistant = CodeAssistant(verbose=True)

    # Test 1: Security Analysis
    print("\n\n🔒 TEST 1: Security Vulnerability Scan")
    print("-"*70)
    assistant.analyze_security(vulnerable_code)

    # Test 2: Performance Analysis
    print("\n\n⚡ TEST 2: Performance Analysis")
    print("-"*70)
    assistant.optimize_performance(vulnerable_code)

    # Test 3: Complexity Analysis
    print("\n\n📊 TEST 3: Code Complexity Analysis")
    print("-"*70)
    assistant.analyze_complexity(vulnerable_code)

    # Test 4: Auto-refactor
    print("\n\n✨ TEST 4: Automatic Refactoring")
    print("-"*70)
    refactored, transforms = assistant.auto_refactor(vulnerable_code)
    print("\nRefactored code preview (first 500 chars):")
    print(refactored[:500])

    # Test 5: Full Analysis
    print("\n\n🔍 TEST 5: Full Code Analysis")
    print("-"*70)
    results = assistant.full_analysis(vulnerable_code)

    print("\n" + "="*70)
    print("✅ All tests completed!")
    print("="*70)

if __name__ == '__main__':
    main()
