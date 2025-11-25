#!/usr/bin/env python3
"""
Pydantic AI Usage Examples
Demonstrates both legacy (dict) and Pydantic (type-safe) modes

Run: python3 example_pydantic_usage.py
"""

from dsmil_ai_engine import DSMILAIEngine, PYDANTIC_AVAILABLE
from pydantic_models import DSMILQueryRequest, ModelTier

# Color output
class Colors:
    GREEN = '\033[92m'
    BLUE = '\033[94m'
    YELLOW = '\033[93m'
    CYAN = '\033[96m'
    RESET = '\033[0m'


def print_header(text):
    print(f"\n{Colors.CYAN}{'='*70}")
    print(f"{text}")
    print(f"{'='*70}{Colors.RESET}\n")


def example_1_legacy_mode():
    """Example 1: Legacy mode - Returns dict"""
    print_header("Example 1: Legacy Mode (Dict Response)")

    # Create engine in legacy mode
    engine = DSMILAIEngine(pydantic_mode=False)

    # Simple query
    result = engine.generate("What is TPM attestation?", model_selection="fast")

    print(f"{Colors.YELLOW}Query:{Colors.RESET} What is TPM attestation?")
    print(f"{Colors.YELLOW}Mode:{Colors.RESET} Legacy (dict)")
    print()

    if result.get('success'):
        print(f"{Colors.GREEN}Response:{Colors.RESET}")
        print(result['response'][:200] + "..." if len(result['response']) > 200 else result['response'])
        print()
        print(f"Model: {result['model']}")
        print(f"Latency: {result['latency_ms']:.2f}ms")
        print(f"Attestation verified: {result['attestation']['verified']}")
    else:
        print(f"{Colors.YELLOW}⚠ {result['error']}{Colors.RESET}")

    print(f"\n{Colors.BLUE}Return type:{Colors.RESET} dict - Access with: result['response']")


def example_2_pydantic_mode():
    """Example 2: Pydantic mode - Returns validated model"""
    if not PYDANTIC_AVAILABLE:
        print(f"{Colors.YELLOW}⚠ Pydantic not available - skipping{Colors.RESET}")
        return

    print_header("Example 2: Pydantic Mode (Type-Safe Response)")

    # Create engine in Pydantic mode
    engine = DSMILAIEngine(pydantic_mode=True)

    # Query with Pydantic request model
    request = DSMILQueryRequest(
        prompt="Explain kernel module compilation",
        model=ModelTier.FAST,
        temperature=0.7
    )

    print(f"{Colors.YELLOW}Query:{Colors.RESET} {request.prompt}")
    print(f"{Colors.YELLOW}Mode:{Colors.RESET} Pydantic (type-safe)")
    print()

    result = engine.generate(request)

    # Result is a Pydantic model with validation and autocomplete!
    print(f"{Colors.GREEN}Response:{Colors.RESET}")
    print(result.response[:200] + "..." if len(result.response) > 200 else result.response)
    print()
    print(f"Model: {result.model_used}")
    print(f"Latency: {result.latency_ms:.2f}ms")
    print(f"Confidence: {result.confidence:.2f}")
    if result.attestation_hash:
        print(f"Attestation: {result.attestation_hash[:16]}...")

    print(f"\n{Colors.BLUE}Return type:{Colors.RESET} DSMILQueryResult - IDE autocomplete works!")
    print(f"{Colors.BLUE}Access with:{Colors.RESET} result.response (typed property)")


def example_3_hybrid_mode():
    """Example 3: Hybrid mode - Override per-call"""
    if not PYDANTIC_AVAILABLE:
        print(f"{Colors.YELLOW}⚠ Pydantic not available - skipping{Colors.RESET}")
        return

    print_header("Example 3: Hybrid Mode (Per-Call Override)")

    # Create engine in legacy mode
    engine = DSMILAIEngine(pydantic_mode=False)

    print(f"{Colors.YELLOW}Engine mode:{Colors.RESET} Legacy (default)")
    print()

    # Call 1: Use legacy mode (default)
    print(f"{Colors.BLUE}Call 1:{Colors.RESET} Using default (dict)")
    result1 = engine.generate("Hello", model_selection="fast")
    print(f"Result type: {type(result1).__name__}")  # dict
    print()

    # Call 2: Override to use Pydantic for this call
    print(f"{Colors.BLUE}Call 2:{Colors.RESET} Override to Pydantic")
    result2 = engine.generate("Hello", model_selection="fast", return_pydantic=True)
    print(f"Result type: {type(result2).__name__}")  # DSMILQueryResult
    print()

    print(f"{Colors.GREEN}✓{Colors.RESET} Can mix dict and Pydantic responses from same engine!")


def example_4_statistics():
    """Example 4: Engine statistics"""
    print_header("Example 4: Engine Statistics")

    engine = DSMILAIEngine(pydantic_mode=True)

    # Run a few queries
    for i in range(3):
        engine.generate(f"Test query {i+1}", model_selection="fast")

    stats = engine.get_statistics()

    print(f"{Colors.YELLOW}Statistics:{Colors.RESET}")
    print(f"  Total queries: {stats['total_queries']}")
    print(f"  Pydantic mode: {stats['pydantic_mode']}")
    print(f"  Pydantic available: {stats['pydantic_available']}")
    print(f"  RAG enabled: {stats['rag_enabled']}")


def main():
    print(f"""
{Colors.CYAN}╔══════════════════════════════════════════════════════════════════════╗
║         DSMIL AI Engine - Pydantic AI Examples                        ║
╚══════════════════════════════════════════════════════════════════════╝{Colors.RESET}
""")

    print(f"{Colors.YELLOW}Pydantic Available:{Colors.RESET} {PYDANTIC_AVAILABLE}")

    if not PYDANTIC_AVAILABLE:
        print(f"\n{Colors.YELLOW}⚠  Install Pydantic to see all examples:{Colors.RESET}")
        print("   pip install pydantic pydantic-ai")
        print("\nShowing legacy mode only...\n")

    try:
        example_1_legacy_mode()
        example_2_pydantic_mode()
        example_3_hybrid_mode()
        example_4_statistics()

        print(f"\n{Colors.GREEN}✓ All examples completed!{Colors.RESET}")
        print(f"\nNext steps:")
        print(f"  • Read PYDANTIC_AI_INTEGRATION.md for full docs")
        print(f"  • Run benchmark: python3 benchmark_binary_vs_pydantic.py")
        print(f"  • Use in your code: from dsmil_ai_engine import DSMILAIEngine\n")

    except Exception as e:
        print(f"\n{Colors.YELLOW}⚠ Error: {e}{Colors.RESET}")
        print(f"Make sure Ollama is running: ollama serve")


if __name__ == "__main__":
    main()
