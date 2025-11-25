#!/usr/bin/env python3
"""
Heretic Unsloth Integration - Fast Fine-Tuning for Abliteration

Integrates Unsloth (https://github.com/unslothai/unsloth) optimizations:
- 2x faster training
- 70% less VRAM usage
- 0% accuracy loss
- Support for 4-bit/8-bit quantization

This enables rapid experimentation with abliteration on consumer GPUs.

Key Features:
- Fast LoRA/QLoRA training for post-abliteration fine-tuning
- Efficient gradient computation for refusal direction calculation
- Memory-optimized batch processing
- Checkpoint-based resumption for long training runs
"""

import torch
import logging
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class UnslothConfig:
    """Configuration for Unsloth-optimized training"""
    # Model loading
    load_in_4bit: bool = True  # 4-bit quantization for memory efficiency
    load_in_8bit: bool = False
    use_gradient_checkpointing: bool = True

    # LoRA configuration
    lora_r: int = 16  # LoRA rank
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    target_modules: List[str] = None  # Auto-detected if None

    # Training optimization
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    max_seq_length: int = 2048

    # Performance
    use_triton_kernels: bool = True  # Unsloth's custom Triton kernels
    use_flash_attention: bool = True

    def __post_init__(self):
        if self.target_modules is None:
            # Default LoRA targets (common across models)
            self.target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                                   "gate_proj", "up_proj", "down_proj"]


class UnslothOptimizer:
    """
    Unsloth-optimized model wrapper for fast abliteration workflows

    Provides:
    - Fast model loading with quantization
    - Optimized gradient computation for refusal direction calculation
    - Memory-efficient batch processing
    - LoRA/QLoRA fine-tuning post-abliteration
    """

    def __init__(
        self,
        model_name: str,
        config: Optional[UnslothConfig] = None,
        device: str = "cuda"
    ):
        """
        Initialize Unsloth-optimized wrapper

        Args:
            model_name: HuggingFace model identifier
            config: Unsloth configuration
            device: Device for computation
        """
        self.model_name = model_name
        self.config = config or UnslothConfig()
        self.device = device
        self.model = None
        self.tokenizer = None

        # Check for Unsloth availability
        try:
            import unsloth
            self.unsloth_available = True
            logger.info("✅ Unsloth available - optimizations enabled")
        except ImportError:
            self.unsloth_available = False
            logger.warning("⚠️  Unsloth not available - falling back to standard methods")

    def load_model(self) -> Tuple[Any, Any]:
        """
        Load model with Unsloth optimizations

        Returns:
            Tuple of (model, tokenizer)
        """
        if self.unsloth_available:
            return self._load_with_unsloth()
        else:
            return self._load_standard()

    def _load_with_unsloth(self) -> Tuple[Any, Any]:
        """Load model using Unsloth FastLanguageModel"""
        try:
            from unsloth import FastLanguageModel

            logger.info(f"Loading {self.model_name} with Unsloth optimizations...")

            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.model_name,
                max_seq_length=self.config.max_seq_length,
                dtype=None,  # Auto-detect
                load_in_4bit=self.config.load_in_4bit,
                load_in_8bit=self.config.load_in_8bit,
            )

            # Enable gradient checkpointing for memory efficiency
            if self.config.use_gradient_checkpointing:
                model.gradient_checkpointing_enable()

            self.model = model
            self.tokenizer = tokenizer

            logger.info(f"✅ Model loaded with Unsloth optimizations")
            logger.info(f"   4-bit: {self.config.load_in_4bit}, "
                       f"Gradient checkpointing: {self.config.use_gradient_checkpointing}")

            return model, tokenizer

        except Exception as e:
            logger.error(f"Failed to load with Unsloth: {e}")
            return self._load_standard()

    def _load_standard(self) -> Tuple[Any, Any]:
        """Fallback to standard HuggingFace loading"""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info(f"Loading {self.model_name} with standard methods...")

        load_kwargs = {}
        if self.config.load_in_4bit:
            load_kwargs["load_in_4bit"] = True
        elif self.config.load_in_8bit:
            load_kwargs["load_in_8bit"] = True

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            **load_kwargs
        )

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        if self.config.use_gradient_checkpointing:
            model.gradient_checkpointing_enable()

        self.model = model
        self.tokenizer = tokenizer

        return model, tokenizer

    def prepare_for_lora(self) -> Any:
        """
        Prepare model for LoRA fine-tuning (post-abliteration)

        Returns:
            LoRA-wrapped model
        """
        if not self.unsloth_available:
            logger.warning("LoRA preparation requires Unsloth - skipping")
            return self.model

        try:
            from unsloth import FastLanguageModel

            logger.info("Preparing model for LoRA fine-tuning...")

            model = FastLanguageModel.get_peft_model(
                self.model,
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=self.config.target_modules,
                bias="none",
                use_gradient_checkpointing=self.config.use_gradient_checkpointing,
            )

            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in model.parameters())

            logger.info(f"✅ LoRA prepared: {trainable_params:,} / {total_params:,} "
                       f"params trainable ({trainable_params/total_params*100:.2f}%)")

            return model

        except Exception as e:
            logger.error(f"Failed to prepare LoRA: {e}")
            return self.model

    def compute_refusal_directions_fast(
        self,
        good_prompts: List[str],
        bad_prompts: List[str],
        batch_size: int = 4  # Smaller batch for memory efficiency
    ) -> torch.Tensor:
        """
        Fast refusal direction computation using Unsloth optimizations

        Uses:
        - Memory-efficient batching
        - Gradient checkpointing
        - Mixed precision

        Args:
            good_prompts: Harmless prompts
            bad_prompts: Harmful prompts
            batch_size: Batch size (smaller for memory efficiency)

        Returns:
            Refusal direction tensor [n_layers, hidden_size]
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        logger.info(f"Computing refusal directions with Unsloth optimizations...")
        logger.info(f"  Good prompts: {len(good_prompts)}, Bad prompts: {len(bad_prompts)}")
        logger.info(f"  Batch size: {batch_size}")

        # Use existing RefusalDirectionCalculator but with optimized model
        from heretic_abliteration import RefusalDirectionCalculator

        calculator = RefusalDirectionCalculator(
            self.model,
            self.tokenizer,
            device=self.device
        )

        # Calculate directions
        refusal_dirs = calculator.calculate_refusal_directions(
            good_prompts,
            bad_prompts,
            batch_size=batch_size
        )

        logger.info(f"✅ Refusal directions computed: shape {refusal_dirs.shape}")

        return refusal_dirs

    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get current memory usage statistics

        Returns:
            Dictionary with memory stats
        """
        if not torch.cuda.is_available():
            return {"device": "cpu", "available": False}

        stats = {
            "device": "cuda",
            "allocated_gb": torch.cuda.memory_allocated() / 1e9,
            "reserved_gb": torch.cuda.memory_reserved() / 1e9,
            "max_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
        }

        # Calculate VRAM savings vs standard loading
        if self.config.load_in_4bit:
            stats["estimated_savings"] = "~70%"
        elif self.config.load_in_8bit:
            stats["estimated_savings"] = "~50%"
        else:
            stats["estimated_savings"] = "0%"

        return stats


class UnslothTrainer:
    """
    Fast training for post-abliteration fine-tuning

    Use cases:
    - Fine-tune abliterated model on specific tasks
    - Restore helpful capabilities while maintaining uncensored behavior
    - Domain adaptation post-abliteration
    """

    def __init__(
        self,
        model,
        tokenizer,
        config: Optional[UnslothConfig] = None
    ):
        """
        Initialize Unsloth trainer

        Args:
            model: Model to train (can be LoRA-wrapped)
            tokenizer: Tokenizer
            config: Training configuration
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or UnslothConfig()

    def train(
        self,
        train_dataset,
        output_dir: str,
        num_epochs: int = 1,
        learning_rate: float = 2e-4,
        **kwargs
    ) -> Any:
        """
        Fast training with Unsloth optimizations

        Args:
            train_dataset: Training dataset
            output_dir: Directory to save checkpoints
            num_epochs: Number of training epochs
            learning_rate: Learning rate
            **kwargs: Additional training arguments

        Returns:
            Training result
        """
        try:
            from transformers import TrainingArguments
            from trl import SFTTrainer

            logger.info(f"Starting fast training with Unsloth optimizations...")

            training_args = TrainingArguments(
                output_dir=output_dir,
                per_device_train_batch_size=self.config.per_device_train_batch_size,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                num_train_epochs=num_epochs,
                learning_rate=learning_rate,
                fp16=True if torch.cuda.is_available() else False,
                logging_steps=10,
                save_strategy="epoch",
                **kwargs
            )

            trainer = SFTTrainer(
                model=self.model,
                tokenizer=self.tokenizer,
                train_dataset=train_dataset,
                args=training_args,
                max_seq_length=self.config.max_seq_length,
            )

            result = trainer.train()

            logger.info(f"✅ Training complete")

            return result

        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise


def demo():
    """Demo of Unsloth integration"""
    print("=== Heretic Unsloth Integration Demo ===\n")

    # 1. Configuration
    print("1. Unsloth Configuration")
    config = UnslothConfig(
        load_in_4bit=True,
        lora_r=16,
        max_seq_length=2048,
    )
    print(f"   4-bit loading: {config.load_in_4bit}")
    print(f"   LoRA rank: {config.lora_r}")
    print(f"   Max sequence: {config.max_seq_length}")

    # 2. Check availability
    print("\n2. Checking Unsloth Availability")
    optimizer = UnslothOptimizer(
        model_name="Qwen/Qwen2-7B-Instruct",
        config=config
    )
    print(f"   Unsloth available: {optimizer.unsloth_available}")

    # 3. Memory estimation
    print("\n3. Memory Savings Estimation")
    print(f"   Standard 7B model: ~28GB VRAM")
    print(f"   With 4-bit quantization: ~8.4GB VRAM (70% savings)")
    print(f"   Enables: Consumer GPUs (RTX 3090, 4090)")

    print("\n✅ Unsloth integration ready for fast abliteration workflows")


if __name__ == "__main__":
    demo()
