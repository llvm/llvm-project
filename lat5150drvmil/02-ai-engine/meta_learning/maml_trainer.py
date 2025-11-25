#!/usr/bin/env python3
"""
Model-Agnostic Meta-Learning (MAML) Trainer

Enables few-shot learning by training models to adapt quickly to new tasks

MAML Algorithm:
1. Sample batch of tasks
2. For each task:
   a. Adapt model with K-shot training (inner loop)
   b. Evaluate on query set
3. Update meta-parameters to minimize query loss (outer loop)

Applications:
- Few-shot domain adaptation (new cybersecurity domains with limited data)
- Rapid fine-tuning for new DSMIL device types
- Quick personalization for specific use cases

Hardware Optimization:
- Intel Arc GPU for meta-training (UMA 44-48 GiB)
- BF16 precision for memory efficiency
- Gradient checkpointing for large models

References:
- Finn et al., "Model-Agnostic Meta-Learning for Fast Adaptation" (ICML 2017)
- Nichol et al., "On First-Order Meta-Learning Algorithms" (2018)
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass
import numpy as np
from copy import deepcopy
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MAMLConfig:
    """MAML training configuration"""
    # Task configuration
    n_way: int = 5               # Number of classes per task
    k_shot: int = 5              # Number of examples per class (support set)
    q_query: int = 15            # Number of query examples per class

    # Meta-learning hyperparameters
    inner_lr: float = 0.01       # Learning rate for inner loop (task adaptation)
    outer_lr: float = 0.001      # Learning rate for outer loop (meta-update)
    num_inner_steps: int = 5     # Number of gradient steps in inner loop
    first_order: bool = False    # Use first-order approximation (faster, less accurate)

    # Training settings
    meta_batch_size: int = 4     # Number of tasks per meta-batch
    num_meta_iterations: int = 10000  # Total meta-training iterations
    eval_interval: int = 100     # Evaluate every N iterations

    # Hardware settings
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    precision: str = "bf16"      # "fp32", "fp16", "bf16"
    gradient_checkpointing: bool = True


@dataclass
class Task:
    """A single meta-learning task"""
    support_x: torch.Tensor  # Support set inputs [n_way * k_shot, ...]
    support_y: torch.Tensor  # Support set labels [n_way * k_shot]
    query_x: torch.Tensor    # Query set inputs [n_way * q_query, ...]
    query_y: torch.Tensor    # Query set labels [n_way * q_query]
    task_id: int
    task_description: str


class MAMLModel(nn.Module):
    """
    Meta-learnable model wrapper

    Wraps any PyTorch model to enable MAML training
    """

    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base_model = base_model

        # Store parameter names for easier manipulation
        self.param_names = [name for name, _ in base_model.named_parameters()]

    def forward(self, x: torch.Tensor, params: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        """
        Forward pass with optional parameter override

        Args:
            x: Input tensor
            params: Optional parameter dict (for adapted parameters)

        Returns:
            Model output
        """
        if params is None:
            return self.base_model(x)
        else:
            # Use functional API with custom parameters
            return self._forward_with_params(x, params)

    def _forward_with_params(self, x: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass using custom parameters

        This allows us to use adapted parameters without modifying the original model
        """
        # For now, temporarily replace parameters
        # In production, would use functional API or manual forward pass

        original_params = {}
        for name, param in self.base_model.named_parameters():
            if name in params:
                original_params[name] = param.data.clone()
                param.data = params[name]

        # Forward pass
        output = self.base_model(x)

        # Restore original parameters
        for name, original_data in original_params.items():
            self.base_model.get_parameter(name).data = original_data

        return output

    def get_parameters(self) -> Dict[str, torch.Tensor]:
        """Get model parameters as dict"""
        return {name: param.clone() for name, param in self.base_model.named_parameters()}

    def set_parameters(self, params: Dict[str, torch.Tensor]):
        """Set model parameters from dict"""
        for name, param in self.base_model.named_parameters():
            if name in params:
                param.data = params[name].data


class MAMLTrainer:
    """
    MAML meta-trainer

    Trains models to adapt quickly to new tasks with few examples
    """

    def __init__(
        self,
        model: nn.Module,
        config: MAMLConfig
    ):
        self.model = MAMLModel(model)
        self.config = config

        # Move to device
        self.model = self.model.to(config.device)

        # Meta-optimizer (updates meta-parameters)
        self.meta_optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.outer_lr
        )

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Statistics
        self.meta_train_losses = []
        self.meta_val_accuracies = []

        # Mixed precision training
        self.use_amp = (config.precision in ["fp16", "bf16"])
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()

        logger.info("=" * 80)
        logger.info("  MAML Meta-Learning Trainer")
        logger.info("=" * 80)
        logger.info(f"Configuration:")
        logger.info(f"  {config.n_way}-way {config.k_shot}-shot learning")
        logger.info(f"  Inner steps: {config.inner_steps}")
        logger.info(f"  Inner LR: {config.inner_lr}")
        logger.info(f"  Outer LR: {config.outer_lr}")
        logger.info(f"  Device: {config.device}")
        logger.info(f"  Precision: {config.precision}")

    def inner_loop(
        self,
        task: Task,
        params: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[Dict[str, torch.Tensor], float]:
        """
        Inner loop: Adapt model to task using support set

        Args:
            task: Task with support and query sets
            params: Starting parameters (default: current model parameters)

        Returns:
            - adapted_params: Parameters after adaptation
            - support_loss: Final loss on support set
        """
        if params is None:
            params = self.model.get_parameters()

        # Clone parameters for adaptation
        adapted_params = {name: param.clone() for name, param in params.items()}

        # Adapt for K inner steps
        for step in range(self.config.num_inner_steps):
            # Forward pass on support set
            support_logits = self.model(task.support_x, adapted_params)
            support_loss = self.criterion(support_logits, task.support_y)

            # Compute gradients
            grads = torch.autograd.grad(
                support_loss,
                adapted_params.values(),
                create_graph=not self.config.first_order
            )

            # Update adapted parameters
            adapted_params = {
                name: param - self.config.inner_lr * grad
                for (name, param), grad in zip(adapted_params.items(), grads)
            }

        # Compute final support loss
        support_logits = self.model(task.support_x, adapted_params)
        final_support_loss = self.criterion(support_logits, task.support_y)

        return adapted_params, final_support_loss.item()

    def outer_loop(
        self,
        tasks: List[Task]
    ) -> Tuple[float, float]:
        """
        Outer loop: Update meta-parameters based on query set performance

        Args:
            tasks: Batch of tasks

        Returns:
            - meta_loss: Average loss across tasks
            - meta_accuracy: Average accuracy across tasks
        """
        self.meta_optimizer.zero_grad()

        meta_losses = []
        meta_accuracies = []

        for task in tasks:
            # Inner loop: adapt to task
            adapted_params, support_loss = self.inner_loop(task)

            # Evaluate on query set
            query_logits = self.model(task.query_x, adapted_params)
            query_loss = self.criterion(query_logits, task.query_y)

            meta_losses.append(query_loss)

            # Compute accuracy
            query_preds = torch.argmax(query_logits, dim=-1)
            accuracy = (query_preds == task.query_y).float().mean()
            meta_accuracies.append(accuracy.item())

        # Average loss across tasks
        meta_loss = torch.stack(meta_losses).mean()

        # Backward pass
        if self.use_amp:
            self.scaler.scale(meta_loss).backward()
            self.scaler.step(self.meta_optimizer)
            self.scaler.update()
        else:
            meta_loss.backward()
            self.meta_optimizer.step()

        avg_loss = meta_loss.item()
        avg_accuracy = np.mean(meta_accuracies)

        return avg_loss, avg_accuracy

    def train(
        self,
        task_sampler: Callable[[], List[Task]],
        num_iterations: Optional[int] = None,
        val_task_sampler: Optional[Callable[[], List[Task]]] = None
    ):
        """
        Meta-training loop

        Args:
            task_sampler: Function that returns a batch of tasks
            num_iterations: Number of meta-iterations (default: from config)
            val_task_sampler: Optional validation task sampler
        """
        num_iterations = num_iterations or self.config.num_meta_iterations

        logger.info("\n" + "=" * 80)
        logger.info("  Starting Meta-Training")
        logger.info("=" * 80)

        for iteration in range(num_iterations):
            # Sample batch of tasks
            tasks = task_sampler()

            # Meta-update
            meta_loss, meta_accuracy = self.outer_loop(tasks)

            # Log
            self.meta_train_losses.append(meta_loss)

            if iteration % 10 == 0:
                logger.info(
                    f"Iteration {iteration}/{num_iterations} - "
                    f"Loss: {meta_loss:.4f}, Accuracy: {meta_accuracy:.2%}"
                )

            # Evaluate on validation tasks
            if val_task_sampler and iteration % self.config.eval_interval == 0:
                val_accuracy = self.evaluate(val_task_sampler)
                self.meta_val_accuracies.append(val_accuracy)
                logger.info(f"  Validation Accuracy: {val_accuracy:.2%}")

        logger.info("\n" + "=" * 80)
        logger.info("✅ Meta-Training Complete")
        logger.info("=" * 80)

    def evaluate(
        self,
        task_sampler: Callable[[], List[Task]],
        num_eval_tasks: int = 100
    ) -> float:
        """
        Evaluate meta-learned model on new tasks

        Args:
            task_sampler: Function that returns evaluation tasks
            num_eval_tasks: Number of tasks to evaluate on

        Returns:
            Average accuracy across tasks
        """
        self.model.eval()

        accuracies = []

        with torch.no_grad():
            for _ in range(num_eval_tasks):
                tasks = task_sampler()

                for task in tasks:
                    # Adapt to task
                    adapted_params, _ = self.inner_loop(task)

                    # Evaluate on query set
                    query_logits = self.model(task.query_x, adapted_params)
                    query_preds = torch.argmax(query_logits, dim=-1)
                    accuracy = (query_preds == task.query_y).float().mean()

                    accuracies.append(accuracy.item())

        self.model.train()

        return np.mean(accuracies)

    def adapt_to_new_task(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        num_steps: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Adapt meta-learned model to a new task

        This is the key benefit of MAML: rapid adaptation with few examples

        Args:
            support_x: Support set inputs
            support_y: Support set labels
            num_steps: Number of adaptation steps (default: from config)

        Returns:
            Adapted model parameters
        """
        num_steps = num_steps or self.config.num_inner_steps

        # Create dummy task
        task = Task(
            support_x=support_x,
            support_y=support_y,
            query_x=support_x,  # Not used in inner loop
            query_y=support_y,
            task_id=0,
            task_description="New task"
        )

        # Adapt
        adapted_params, _ = self.inner_loop(task)

        return adapted_params

    def save_checkpoint(self, path: str):
        """Save meta-learned model checkpoint"""
        os.makedirs(os.path.dirname(path), exist_ok=True)

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "meta_optimizer_state_dict": self.meta_optimizer.state_dict(),
            "config": self.config,
            "meta_train_losses": self.meta_train_losses,
            "meta_val_accuracies": self.meta_val_accuracies,
        }

        torch.save(checkpoint, path)
        logger.info(f"✓ Checkpoint saved to: {path}")

    def load_checkpoint(self, path: str):
        """Load meta-learned model checkpoint"""
        checkpoint = torch.load(path, map_location=self.config.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.meta_optimizer.load_state_dict(checkpoint["meta_optimizer_state_dict"])
        self.meta_train_losses = checkpoint["meta_train_losses"]
        self.meta_val_accuracies = checkpoint["meta_val_accuracies"]

        logger.info(f"✓ Checkpoint loaded from: {path}")


# =============================================================================
# Example Task Samplers
# =============================================================================

class SyntheticTaskSampler:
    """
    Generate synthetic tasks for testing MAML

    Useful for debugging and initial experiments
    """

    def __init__(
        self,
        n_way: int = 5,
        k_shot: int = 5,
        q_query: int = 15,
        input_dim: int = 84,
        device: str = "cpu"
    ):
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query
        self.input_dim = input_dim
        self.device = device

    def sample_tasks(self, batch_size: int = 4) -> List[Task]:
        """Sample a batch of random tasks"""
        tasks = []

        for task_id in range(batch_size):
            # Generate random support set
            support_x = torch.randn(
                self.n_way * self.k_shot,
                self.input_dim
            ).to(self.device)

            support_y = torch.arange(self.n_way).repeat_interleave(
                self.k_shot
            ).to(self.device)

            # Generate random query set
            query_x = torch.randn(
                self.n_way * self.q_query,
                self.input_dim
            ).to(self.device)

            query_y = torch.arange(self.n_way).repeat_interleave(
                self.q_query
            ).to(self.device)

            tasks.append(Task(
                support_x=support_x,
                support_y=support_y,
                query_x=query_x,
                query_y=query_y,
                task_id=task_id,
                task_description=f"Synthetic task {task_id}"
            ))

        return tasks


def demo():
    """Demo MAML training"""
    print("=" * 80)
    print("  MAML Meta-Learning Demo")
    print("=" * 80)

    # Simple model for testing
    model = nn.Sequential(
        nn.Linear(84, 128),
        nn.ReLU(),
        nn.Linear(128, 5)  # 5-way classification
    )

    # Config
    config = MAMLConfig(
        n_way=5,
        k_shot=5,
        q_query=15,
        num_meta_iterations=100,
        eval_interval=20,
        device="cpu"
    )

    # Trainer
    trainer = MAMLTrainer(model, config)

    # Task sampler
    task_sampler = SyntheticTaskSampler(
        n_way=config.n_way,
        k_shot=config.k_shot,
        q_query=config.q_query,
        device=config.device
    )

    # Meta-train
    trainer.train(
        task_sampler=lambda: task_sampler.sample_tasks(config.meta_batch_size),
        val_task_sampler=lambda: task_sampler.sample_tasks(10)
    )

    # Evaluate
    print("\n" + "=" * 80)
    print("  Final Evaluation")
    print("=" * 80)

    final_accuracy = trainer.evaluate(
        task_sampler=lambda: task_sampler.sample_tasks(10),
        num_eval_tasks=20
    )

    print(f"Final Accuracy: {final_accuracy:.2%}")

    # Test adaptation to new task
    print("\n" + "=" * 80)
    print("  Testing Rapid Adaptation")
    print("=" * 80)

    new_task = task_sampler.sample_tasks(1)[0]
    adapted_params = trainer.adapt_to_new_task(
        new_task.support_x,
        new_task.support_y,
        num_steps=10
    )

    print(f"✓ Adapted to new task with {config.k_shot} examples per class")


if __name__ == "__main__":
    demo()
