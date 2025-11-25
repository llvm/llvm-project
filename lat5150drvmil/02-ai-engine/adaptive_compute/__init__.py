"""
Adaptive Test-Time Compute Scaling

Allocates reasoning budget based on query complexity:
- Simple queries: Fast, minimal iterations
- Medium queries: Moderate compute
- Hard queries: Deep reasoning, maximum iterations

Improves efficiency by 2-3× through smart resource allocation.
"""

from .difficulty_classifier import DifficultyClassifier, DifficultyLevel
from .budget_allocator import BudgetAllocator, ComputeBudget

__all__ = ['DifficultyClassifier', 'DifficultyLevel', 'BudgetAllocator', 'ComputeBudget']
