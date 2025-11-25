#!/usr/bin/env python3
"""
Learned Mixture-of-Experts (MoE) Gating Network

Routes queries to specialized expert models based on:
1. Query embeddings (semantic routing)
2. Task type detection
3. Learned gating network (trained to maximize performance)
4. Load balancing across experts

Architecture:
- Gating Network: Small neural network (embedding → expert weights)
- Experts: Specialized models for different domains
  - Code expert (programming questions)
  - Math expert (mathematical reasoning)
  - General expert (broad knowledge)
  - Security expert (cybersecurity)

Training:
- Supervised: Use human labels for query types
- Reinforcement Learning: Optimize for end-to-end performance
- Load balancing loss: Encourage even expert usage

Hardware Optimization:
- Run gating network on NPU (fast, low power)
- Cache expert models in UMA (44-48 GiB available)
- Top-k expert selection (activate 2-3 experts per query)
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExpertType(Enum):
    """Types of expert models"""
    CODE = "code"             # Programming, algorithms
    MATH = "math"             # Mathematics, reasoning
    SECURITY = "security"     # Cybersecurity, vulnerabilities
    GENERAL = "general"       # Broad knowledge
    CREATIVE = "creative"     # Creative writing, ideation


@dataclass
class ExpertInfo:
    """Information about an expert model"""
    expert_type: ExpertType
    model_path: str
    specialization: str
    load_priority: int        # 1 = always loaded, 2 = sometimes, 3 = rarely
    memory_gb: float          # Memory footprint


@dataclass
class GatingDecision:
    """Gating network decision"""
    query: str
    expert_weights: Dict[ExpertType, float]  # Expert → weight (0-1)
    top_k_experts: List[ExpertType]          # Selected experts
    confidence: float                          # Confidence in routing (0-1)
    reasoning: str                             # Why these experts?


class LearnedGatingNetwork(nn.Module):
    """
    Neural network that routes queries to experts

    Architecture:
        Input: Query embedding (384-dim from all-MiniLM-L6-v2)
        Hidden: 256 → 128
        Output: Expert weights (softmax over 5 experts)

    Features:
        - Load balancing loss (encourage equal expert usage)
        - Top-k gating (activate only k experts)
        - Temperature scaling (control sharpness)
    """

    def __init__(
        self,
        embedding_dim: int = 384,
        hidden_dim: int = 256,
        num_experts: int = 5,
        top_k: int = 2,
        temperature: float = 1.0,
        load_balance_weight: float = 0.01
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.temperature = temperature
        self.load_balance_weight = load_balance_weight

        # Gating network layers
        self.gate = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_experts)
        )

        # Expert names (for interpretability)
        self.expert_names = [e.value for e in ExpertType]

        # Statistics for load balancing
        self.expert_usage_count = torch.zeros(num_experts)

    def forward(
        self,
        query_embedding: torch.Tensor,
        return_all_weights: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass: compute expert weights

        Args:
            query_embedding: Query embedding [batch_size, embedding_dim]
            return_all_weights: Return all expert weights (not just top-k)

        Returns:
            - expert_weights: Top-k expert weights [batch_size, top_k]
            - expert_indices: Top-k expert indices [batch_size, top_k]
            - all_weights: All expert weights [batch_size, num_experts] (optional)
        """
        # Compute logits
        logits = self.gate(query_embedding)  # [batch_size, num_experts]

        # Apply temperature
        logits = logits / self.temperature

        # Softmax to get probabilities
        all_probs = F.softmax(logits, dim=-1)  # [batch_size, num_experts]

        # Top-k selection
        top_k_probs, top_k_indices = torch.topk(all_probs, self.top_k, dim=-1)

        # Renormalize top-k (so they sum to 1)
        top_k_weights = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

        # Update usage statistics
        with torch.no_grad():
            for idx in top_k_indices.flatten():
                self.expert_usage_count[idx] += 1

        if return_all_weights:
            return top_k_weights, top_k_indices, all_probs
        else:
            return top_k_weights, top_k_indices, None

    def compute_load_balance_loss(
        self,
        all_weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute load balancing loss

        Encourages equal expert usage to prevent "expert collapse"
        (where only 1-2 experts are used)

        Loss = coefficient of variation of expert usage

        Args:
            all_weights: All expert weights [batch_size, num_experts]

        Returns:
            Load balance loss (scalar)
        """
        # Average usage per expert across batch
        avg_usage = all_weights.mean(dim=0)  # [num_experts]

        # Coefficient of variation (std / mean)
        std = avg_usage.std()
        mean = avg_usage.mean()

        # Avoid division by zero
        cv = std / (mean + 1e-8)

        return cv * self.load_balance_weight

    def get_expert_statistics(self) -> Dict[str, float]:
        """
        Get expert usage statistics

        Returns:
            Dict mapping expert name → usage percentage
        """
        total_usage = self.expert_usage_count.sum().item()

        if total_usage == 0:
            return {name: 0.0 for name in self.expert_names}

        stats = {}
        for i, name in enumerate(self.expert_names):
            usage_pct = (self.expert_usage_count[i].item() / total_usage) * 100
            stats[name] = usage_pct

        return stats

    def reset_statistics(self):
        """Reset expert usage statistics"""
        self.expert_usage_count.zero_()


class MoERouter:
    """
    Routes queries to appropriate expert models using learned gating network
    """

    def __init__(
        self,
        gating_model_path: Optional[str] = None,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        top_k: int = 2,
        device: str = "cpu"
    ):
        self.top_k = top_k
        self.device = device

        # Initialize gating network
        self.gating_network = LearnedGatingNetwork(
            embedding_dim=384,
            num_experts=len(ExpertType),
            top_k=top_k
        ).to(device)

        # Load trained weights if available
        if gating_model_path and os.path.exists(gating_model_path):
            logger.info(f"Loading gating network from: {gating_model_path}")
            self.gating_network.load_state_dict(torch.load(gating_model_path))
            self.gating_network.eval()
        else:
            logger.warning("⚠️  Gating network not trained - using random initialization")

        # Initialize embedding model
        try:
            from transformers import AutoTokenizer, AutoModel
            self.tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
            self.embedding_model = AutoModel.from_pretrained(embedding_model_name).to(device)
            self.embedding_model.eval()
        except ImportError:
            logger.error("transformers not installed - using dummy embeddings")
            self.tokenizer = None
            self.embedding_model = None

        # Expert configurations
        self.experts = {
            ExpertType.CODE: ExpertInfo(
                expert_type=ExpertType.CODE,
                model_path="/tank/ai-engine/models/code-expert",
                specialization="Programming, algorithms, debugging, code review",
                load_priority=1,
                memory_gb=3.0
            ),
            ExpertType.MATH: ExpertInfo(
                expert_type=ExpertType.MATH,
                model_path="/tank/ai-engine/models/math-expert",
                specialization="Mathematics, logic, reasoning, calculations",
                load_priority=2,
                memory_gb=3.0
            ),
            ExpertType.SECURITY: ExpertInfo(
                expert_type=ExpertType.SECURITY,
                model_path="/tank/ai-engine/models/security-expert",
                specialization="Cybersecurity, vulnerabilities, DSMIL enumeration",
                load_priority=1,
                memory_gb=3.0
            ),
            ExpertType.GENERAL: ExpertInfo(
                expert_type=ExpertType.GENERAL,
                model_path="/tank/ai-engine/models/general-expert",
                specialization="Broad knowledge, factual questions, explanations",
                load_priority=1,
                memory_gb=3.0
            ),
            ExpertType.CREATIVE: ExpertInfo(
                expert_type=ExpertType.CREATIVE,
                model_path="/tank/ai-engine/models/creative-expert",
                specialization="Creative writing, brainstorming, ideation",
                load_priority=3,
                memory_gb=3.0
            ),
        }

        logger.info(f"MoE Router initialized with {len(self.experts)} experts")

    def embed_query(self, query: str) -> torch.Tensor:
        """
        Embed query using sentence transformer

        Args:
            query: Query text

        Returns:
            Embedding tensor [1, 384]
        """
        if self.embedding_model is None:
            # Dummy embedding for testing
            return torch.randn(1, 384).to(self.device)

        # Tokenize
        inputs = self.tokenizer(
            query,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)

        # Generate embedding
        with torch.no_grad():
            outputs = self.embedding_model(**inputs)
            # Mean pooling
            embedding = outputs.last_hidden_state.mean(dim=1)

        return embedding

    def route_query(self, query: str) -> GatingDecision:
        """
        Route query to appropriate experts

        Args:
            query: Query text

        Returns:
            GatingDecision with expert weights and reasoning
        """
        # Embed query
        query_embedding = self.embed_query(query)

        # Get gating decision
        with torch.no_grad():
            top_k_weights, top_k_indices, all_weights = self.gating_network(
                query_embedding,
                return_all_weights=True
            )

        # Convert to expert types
        top_k_experts = [
            list(ExpertType)[idx.item()]
            for idx in top_k_indices[0]
        ]

        # Create weights dict
        expert_weights = {}
        for expert_type, weight in zip(top_k_experts, top_k_weights[0]):
            expert_weights[expert_type] = weight.item()

        # Compute confidence (entropy-based)
        all_probs = all_weights[0].cpu().numpy()
        entropy = -np.sum(all_probs * np.log(all_probs + 1e-10))
        max_entropy = np.log(len(ExpertType))
        confidence = 1.0 - (entropy / max_entropy)  # High confidence = low entropy

        # Generate reasoning
        reasoning = self._generate_reasoning(query, top_k_experts, expert_weights)

        return GatingDecision(
            query=query,
            expert_weights=expert_weights,
            top_k_experts=top_k_experts,
            confidence=confidence,
            reasoning=reasoning
        )

    def _generate_reasoning(
        self,
        query: str,
        experts: List[ExpertType],
        weights: Dict[ExpertType, float]
    ) -> str:
        """Generate human-readable reasoning for expert selection"""

        parts = []

        # Primary expert
        primary = experts[0]
        primary_weight = weights[primary]
        parts.append(f"{primary.value} expert ({primary_weight:.1%})")

        # Secondary experts
        if len(experts) > 1:
            secondary = experts[1]
            secondary_weight = weights[secondary]
            parts.append(f"{secondary.value} expert ({secondary_weight:.1%})")

        reasoning = f"Routing to: {', '.join(parts)}"

        # Add query analysis
        query_lower = query.lower()

        if "code" in query_lower or "function" in query_lower or "python" in query_lower:
            reasoning += " - detected programming-related query"
        elif "math" in query_lower or "calculate" in query_lower or "prove" in query_lower:
            reasoning += " - detected mathematical query"
        elif "security" in query_lower or "vulnerability" in query_lower or "dsmil" in query_lower:
            reasoning += " - detected security-related query"

        return reasoning

    def get_expert_usage_stats(self) -> Dict[str, float]:
        """Get expert usage statistics"""
        return self.gating_network.get_expert_statistics()


class MoETrainer:
    """
    Train the gating network using supervised learning or RL

    Training approaches:
    1. Supervised: Use labeled data (query → expert_id)
    2. Reward-based: Optimize for end-to-end performance
    3. Combined: Supervised pre-training + RL fine-tuning
    """

    def __init__(
        self,
        gating_network: LearnedGatingNetwork,
        learning_rate: float = 1e-4,
        device: str = "cpu"
    ):
        self.gating_network = gating_network
        self.device = device

        # Optimizer
        self.optimizer = torch.optim.Adam(
            gating_network.parameters(),
            lr=learning_rate
        )

        # Loss function (cross-entropy for supervised)
        self.criterion = nn.CrossEntropyLoss()

    def train_supervised(
        self,
        queries: List[str],
        expert_labels: List[int],  # Expert indices (0-4)
        query_embeddings: torch.Tensor,
        num_epochs: int = 10,
        batch_size: int = 32
    ):
        """
        Train gating network with supervised learning

        Args:
            queries: List of query texts
            expert_labels: Ground truth expert indices
            query_embeddings: Pre-computed embeddings [num_queries, 384]
            num_epochs: Number of training epochs
            batch_size: Batch size
        """
        logger.info("=" * 80)
        logger.info("  Training Gating Network (Supervised)")
        logger.info("=" * 80)

        num_samples = len(queries)
        expert_labels_tensor = torch.tensor(expert_labels, dtype=torch.long).to(self.device)

        for epoch in range(num_epochs):
            total_loss = 0.0
            num_batches = 0

            # Shuffle data
            indices = torch.randperm(num_samples)

            for i in range(0, num_samples, batch_size):
                batch_indices = indices[i:i + batch_size]

                # Get batch
                batch_embeddings = query_embeddings[batch_indices].to(self.device)
                batch_labels = expert_labels_tensor[batch_indices]

                # Forward pass
                self.optimizer.zero_grad()

                _, _, all_weights = self.gating_network(
                    batch_embeddings,
                    return_all_weights=True
                )

                # Compute logits (un-normalized)
                logits = torch.log(all_weights + 1e-10)

                # Classification loss
                classification_loss = self.criterion(logits, batch_labels)

                # Load balancing loss
                load_balance_loss = self.gating_network.compute_load_balance_loss(all_weights)

                # Total loss
                loss = classification_loss + load_balance_loss

                # Backward pass
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            avg_loss = total_loss / num_batches
            logger.info(f"Epoch {epoch + 1}/{num_epochs} - Loss: {avg_loss:.4f}")

        logger.info("✓ Training complete")

    def save_model(self, path: str):
        """Save trained gating network"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.gating_network.state_dict(), path)
        logger.info(f"✓ Model saved to: {path}")


def demo():
    """Demo of MoE routing"""
    print("=" * 80)
    print("  Mixture-of-Experts (MoE) Routing Demo")
    print("=" * 80)

    # Initialize router
    router = MoERouter(device="cpu")

    # Test queries
    test_queries = [
        "How do I write a Python function to sort a list?",
        "What is the derivative of x^2?",
        "Explain DSMIL enumeration techniques",
        "What are the best practices for secure coding?",
        "Write a creative story about a robot",
        "How does gradient descent work?",
    ]

    print("\nRouting decisions:\n")

    for query in test_queries:
        decision = router.route_query(query)

        print(f"Query: {query}")
        print(f"  Experts: {[e.value for e in decision.top_k_experts]}")
        print(f"  Weights: {[f'{w:.2%}' for w in decision.expert_weights.values()]}")
        print(f"  Confidence: {decision.confidence:.2%}")
        print(f"  Reasoning: {decision.reasoning}")
        print()

    # Usage statistics
    print("\nExpert usage statistics:")
    stats = router.get_expert_usage_stats()
    for expert, usage_pct in stats.items():
        print(f"  {expert}: {usage_pct:.1f}%")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    demo()
