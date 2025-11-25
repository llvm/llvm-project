"""
Distributed Training Infrastructure

Implements multi-GPU training strategies:
- FSDP (Fully Sharded Data Parallel)
- Mixed precision training (FP16/BF16/FP8)
- Gradient checkpointing

Based on MegaDLMs framework
"""

from .fsdp_trainer import FSDPTrainer

__all__ = ['FSDPTrainer']
