#!/usr/bin/env python3
"""
Mixture of Experts Aggregator

Combines outputs from multiple expert models using various aggregation strategies.
"""

import re
from typing import List, Dict, Any
from dataclasses import dataclass
from enum import Enum

from .expert_models import ExpertResponse


class AggregationStrategy(Enum):
    """Strategies for combining expert outputs"""
    WEIGHTED_VOTE = "weighted_vote"      # Weight by confidence
    BEST_OF_N = "best_of_n"              # Select highest confidence
    CONSENSUS = "consensus"               # Majority agreement
    CONCATENATE = "concatenate"           # Combine all outputs
    STAGED = "staged"                     # Primary with fallback


@dataclass
class AggregatedResponse:
    """Combined response from multiple experts"""
    final_response: str
    contributing_experts: List[str]
    confidence: float
    strategy_used: str
    metadata: Dict[str, Any]


class MoEAggregator:
    """
    Aggregates outputs from multiple expert models.

    Implements various strategies for combining expert responses.
    """

    def __init__(self, default_strategy: AggregationStrategy = AggregationStrategy.WEIGHTED_VOTE):
        self.default_strategy = default_strategy

    def aggregate(
        self,
        responses: List[ExpertResponse],
        strategy: AggregationStrategy = None
    ) -> AggregatedResponse:
        """
        Aggregate expert responses using specified strategy.

        Args:
            responses: List of expert responses
            strategy: Aggregation strategy (uses default if None)

        Returns:
            Aggreg

ated response
        """
        if not responses:
            return AggregatedResponse(
                final_response="No expert responses available",
                contributing_experts=[],
                confidence=0.0,
                strategy_used="none",
                metadata={}
            )

        strategy = strategy or self.default_strategy

        if strategy == AggregationStrategy.BEST_OF_N:
            return self._best_of_n(responses)
        elif strategy == AggregationStrategy.WEIGHTED_VOTE:
            return self._weighted_vote(responses)
        elif strategy == AggregationStrategy.CONSENSUS:
            return self._consensus(responses)
        elif strategy == AggregationStrategy.CONCATENATE:
            return self._concatenate(responses)
        elif strategy == AggregationStrategy.STAGED:
            return self._staged(responses)
        else:
            return self._best_of_n(responses)

    def _best_of_n(self, responses: List[ExpertResponse]) -> AggregatedResponse:
        """Select the response with highest confidence."""
        best = max(responses, key=lambda r: r.confidence)

        return AggregatedResponse(
            final_response=best.response_text,
            contributing_experts=[best.expert_name],
            confidence=best.confidence,
            strategy_used="best_of_n",
            metadata={"all_confidences": [r.confidence for r in responses]}
        )

    def _weighted_vote(self, responses: List[ExpertResponse]) -> AggregatedResponse:
        """Combine responses weighted by confidence."""
        if len(responses) == 1:
            return self._best_of_n(responses)

        # For now, use best response with confidence adjustment
        # In production, would implement actual weighted combination
        best = max(responses, key=lambda r: r.confidence)
        avg_confidence = sum(r.confidence for r in responses) / len(responses)

        return AggregatedResponse(
            final_response=best.response_text,
            contributing_experts=[r.expert_name for r in responses],
            confidence=avg_confidence,
            strategy_used="weighted_vote",
            metadata={"individual_confidences": {r.expert_name: r.confidence for r in responses}}
        )

    def _consensus(self, responses: List[ExpertResponse]) -> AggregatedResponse:
        """Select response with most agreement."""
        # Placeholder: in production would implement similarity matching
        return self._best_of_n(responses)

    def _concatenate(self, responses: List[ExpertResponse]) -> AggregatedResponse:
        """Concatenate all expert responses."""
        combined = []
        for r in responses:
            combined.append(f"[{r.expert_name}]:\n{r.response_text}\n")

        final = "\n".join(combined)
        avg_confidence = sum(r.confidence for r in responses) / len(responses)

        return AggregatedResponse(
            final_response=final,
            contributing_experts=[r.expert_name for r in responses],
            confidence=avg_confidence,
            strategy_used="concatenate",
            metadata={}
        )

    def _staged(self, responses: List[ExpertResponse]) -> AggregatedResponse:
        """Use primary with fallback to secondaries."""
        # Primary is first, fallback to others if confidence low
        primary = responses[0]

        if primary.confidence > 0.7 or len(responses) == 1:
            return AggregatedResponse(
                final_response=primary.response_text,
                contributing_experts=[primary.expert_name],
                confidence=primary.confidence,
                strategy_used="staged_primary",
                metadata={}
            )

        # Use secondary
        return self._best_of_n(responses[1:])
