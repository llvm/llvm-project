#!/usr/bin/env python3
"""
Reflection Agent - Self-Assessment of Evidence Quality

Evaluates retrieved evidence and decides if more retrieval is needed.

Questions asked:
- Do I have enough evidence?
- Is the evidence relevant?
- Should I retrieve more or proceed?

Author: LAT5150DRVMIL AI Framework
Version: 1.0.0
"""

from typing import Dict, List


class ReflectionAgent:
    """
    Self-reflection on evidence quality

    Usage:
        agent = ReflectionAgent()
        reflection = agent.reflect(
            query="How to optimize SQL?",
            documents=[...],
            iteration=1
        )
        # Returns: {"decision": "sufficient"|"insufficient", "confidence": 0.8, ...}
    """

    def __init__(self, min_documents: int = 3, min_confidence: float = 0.7):
        """
        Initialize reflection agent

        Args:
            min_documents: Minimum documents needed
            min_confidence: Minimum confidence threshold
        """
        self.min_documents = min_documents
        self.min_confidence = min_confidence

    def reflect(
        self,
        query: str,
        documents: List[Dict],
        iteration: int,
        max_iterations: int = 10
    ) -> Dict:
        """
        Reflect on evidence quality

        Args:
            query: Original query
            documents: Retrieved documents
            iteration: Current iteration
            max_iterations: Maximum allowed iterations

        Returns:
            Reflection dict with decision and reasoning
        """
        # Check document quantity
        has_enough_docs = len(documents) >= self.min_documents

        # Estimate relevance (simplified - would use LLM in production)
        avg_score = sum(doc.get("score", 0) for doc in documents) / len(documents) if documents else 0
        has_good_relevance = avg_score >= self.min_confidence

        # Check iteration limit
        at_iteration_limit = iteration >= max_iterations

        # Make decision
        if at_iteration_limit:
            decision = "sufficient"  # Must proceed
            reasoning = f"Iteration limit reached ({iteration}/{max_iterations})"
            confidence = 0.5
        elif has_enough_docs and has_good_relevance:
            decision = "sufficient"
            reasoning = f"Have {len(documents)} docs with avg score {avg_score:.2f}"
            confidence = min(avg_score, 1.0)
        else:
            decision = "insufficient"
            if not has_enough_docs:
                reasoning = f"Only {len(documents)} docs (need {self.min_documents})"
            else:
                reasoning = f"Low relevance: {avg_score:.2f} (need {self.min_confidence})"
            confidence = 0.4

        return {
            "decision": decision,
            "reasoning": reasoning,
            "confidence": confidence,
            "num_documents": len(documents),
            "avg_relevance": avg_score,
            "iteration": iteration
        }


if __name__ == "__main__":
    print("="*70)
    print("Reflection Agent Demo")
    print("="*70)

    agent = ReflectionAgent()

    # Test case 1: Sufficient evidence
    docs = [{"text": f"Doc {i}", "score": 0.9 - i*0.1} for i in range(5)]
    reflection = agent.reflect("Test query", docs, iteration=1)
    print(f"\nCase 1: Sufficient evidence")
    print(f"  Decision: {reflection['decision']}")
    print(f"  Reasoning: {reflection['reasoning']}")
    print(f"  Confidence: {reflection['confidence']:.2f}")

    # Test case 2: Insufficient evidence
    docs = [{"text": "Doc 1", "score": 0.4}]
    reflection = agent.reflect("Test query", docs, iteration=1)
    print(f"\nCase 2: Insufficient evidence")
    print(f"  Decision: {reflection['decision']}")
    print(f"  Reasoning: {reflection['reasoning']}")

    # Test case 3: Iteration limit
    docs = [{"text": "Doc 1", "score": 0.5}]
    reflection = agent.reflect("Test query", docs, iteration=10, max_iterations=10)
    print(f"\nCase 3: Iteration limit")
    print(f"  Decision: {reflection['decision']}")
    print(f"  Reasoning: {reflection['reasoning']}")
