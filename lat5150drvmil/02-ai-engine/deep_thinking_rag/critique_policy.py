#!/usr/bin/env python3
"""
Critique Policy Agent - Control Flow Decisions

Makes strategic decisions about pipeline execution:
- Continue retrieval
- Revise query
- Synthesize answer
- Fail (cannot complete)

Uses policy-based decision making inspired by MDP (Markov Decision Process).

Author: LAT5150DRVMIL AI Framework
Version: 1.0.0
"""

from typing import Dict
from enum import Enum


class PolicyAction(Enum):
    """Policy agent actions"""
    CONTINUE = "continue"           # Continue pipeline
    RETRIEVE_MORE = "retrieve_more" # Need more documents
    REVISE_QUERY = "revise_query"   # Replan with different strategy
    SYNTHESIZE = "synthesize"       # Ready to generate answer
    FAIL = "fail"                   # Cannot complete task


class CritiquePolicy:
    """
    Policy-based control flow for RAG pipeline

    Usage:
        policy = CritiquePolicy()
        action = policy.decide(state)
        # Returns: PolicyAction.SYNTHESIZE
    """

    def __init__(self):
        """Initialize critique policy"""
        pass

    def decide(self, state: Dict) -> Dict:
        """
        Make policy decision based on current state

        Args:
            state: Current RAG state dict with:
                - iteration: Current iteration
                - max_iterations: Max allowed
                - reflection: Reflection assessment
                - documents: Retrieved documents
                - refined_documents: Refined documents

        Returns:
            Decision dict with action and reasoning
        """
        iteration = state.get("iteration", 0)
        max_iterations = state.get("max_iterations", 10)
        reflection = state.get("reflection", {})
        documents = state.get("documents", [])
        refined_docs = state.get("refined_documents", [])

        # Rule 1: If at max iterations, must synthesize or fail
        if iteration >= max_iterations:
            if len(refined_docs) > 0:
                return {
                    "action": PolicyAction.SYNTHESIZE.value,
                    "reasoning": "Max iterations reached, have some evidence",
                    "confidence": 0.6
                }
            else:
                return {
                    "action": PolicyAction.FAIL.value,
                    "reasoning": "Max iterations reached, no useful evidence",
                    "confidence": 0.9
                }

        # Rule 2: If reflection says sufficient, synthesize
        if reflection.get("decision") == "sufficient":
            return {
                "action": PolicyAction.SYNTHESIZE.value,
                "reasoning": "Reflection assessment: sufficient evidence",
                "confidence": reflection.get("confidence", 0.7)
            }

        # Rule 3: If no documents at all, retrieve more
        if len(documents) == 0:
            return {
                "action": PolicyAction.RETRIEVE_MORE.value,
                "reasoning": "No documents retrieved yet",
                "confidence": 0.9
            }

        # Rule 4: If low quality and early iterations, retrieve more
        avg_score = sum(d.get("score", 0) for d in refined_docs) / len(refined_docs) if refined_docs else 0

        if avg_score < 0.5 and iteration < max_iterations // 2:
            return {
                "action": PolicyAction.RETRIEVE_MORE.value,
                "reasoning": f"Low quality docs (avg={avg_score:.2f}), try different strategy",
                "confidence": 0.7
            }

        # Rule 5: If some evidence but not great, continue refining
        if len(refined_docs) > 0:
            return {
                "action": PolicyAction.CONTINUE.value,
                "reasoning": f"Have {len(refined_docs)} docs, continue refining",
                "confidence": 0.6
            }

        # Default: Continue
        return {
            "action": PolicyAction.CONTINUE.value,
            "reasoning": "Default continue action",
            "confidence": 0.5
        }

    def should_synthesize(self, state: Dict) -> bool:
        """Check if should synthesize answer"""
        decision = self.decide(state)
        return decision["action"] == PolicyAction.SYNTHESIZE.value

    def should_retrieve_more(self, state: Dict) -> bool:
        """Check if should retrieve more documents"""
        decision = self.decide(state)
        return decision["action"] == PolicyAction.RETRIEVE_MORE.value


if __name__ == "__main__":
    print("="*70)
    print("Critique Policy Agent Demo")
    print("="*70)

    policy = CritiquePolicy()

    # Test case 1: Sufficient evidence
    state = {
        "iteration": 2,
        "max_iterations": 10,
        "reflection": {"decision": "sufficient", "confidence": 0.9},
        "documents": [{"text": f"Doc {i}", "score": 0.9} for i in range(5)],
        "refined_documents": [{"text": f"Doc {i}", "score": 0.9} for i in range(3)]
    }
    decision = policy.decide(state)
    print(f"\nCase 1: Sufficient evidence")
    print(f"  Action: {decision['action']}")
    print(f"  Reasoning: {decision['reasoning']}")
    print(f"  Confidence: {decision['confidence']}")

    # Test case 2: No documents
    state = {
        "iteration": 1,
        "max_iterations": 10,
        "reflection": {"decision": "insufficient"},
        "documents": [],
        "refined_documents": []
    }
    decision = policy.decide(state)
    print(f"\nCase 2: No documents")
    print(f"  Action: {decision['action']}")
    print(f"  Reasoning: {decision['reasoning']}")

    # Test case 3: Max iterations
    state = {
        "iteration": 10,
        "max_iterations": 10,
        "reflection": {"decision": "insufficient"},
        "documents": [{"text": "Doc 1", "score": 0.5}],
        "refined_documents": [{"text": "Doc 1", "score": 0.5}]
    }
    decision = policy.decide(state)
    print(f"\nCase 3: Max iterations")
    print(f"  Action: {decision['action']}")
    print(f"  Reasoning: {decision['reasoning']}")
