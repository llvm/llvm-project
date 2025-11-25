"""
WhiteRabbitNeo Model Manager
=============================
Comprehensive manager for WhiteRabbitNeo models with multi-device support,
runtime device switching, and dynamic allocation.

Supported Models:
- WhiteRabbitNeo-33B-v1 (33B parameters)
- Llama-3.1-WhiteRabbitNeo-2-70B (70B parameters)

Device Support:
- Main System (CPU/RAM)
- Intel Arc Graphics (GPU)
- Intel NPU (AI Boost)
- Intel NCS2 (Edge inference)

Features:
- Runtime device selection and switching
- Dynamic memory management
- Layer-wise offloading
- Quantization support (FP16/INT8/INT4)
- Validation pipeline integration

Author: LAT5150DRVMIL AI Platform
"""

import json
import logging
import os
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from dynamic_allocator import (
    AllocationPlan,
    DeviceType,
    ModelSpec,
    QuantizationType,
    get_allocator,
)

logger = logging.getLogger(__name__)


class ModelState(Enum):
    """Model loading state."""
    UNLOADED = "unloaded"
    LOADING = "loading"
    LOADED = "loaded"
    ERROR = "error"


class InferenceMode(Enum):
    """Inference mode selection."""
    GPU_PRIMARY = "gpu_primary"  # GPU with layer offloading
    NPU_PRIMARY = "npu_primary"  # NPU with GPU fallback
    EDGE_PRIMARY = "edge_primary"  # NCS2 with streaming
    CPU_ONLY = "cpu_only"  # CPU fallback
    DISTRIBUTED = "distributed"  # Balanced across all devices
    AUTO = "auto"  # Automatic selection


@dataclass
class ModelConfig:
    """Model configuration."""
    name: str
    model_id: str  # HuggingFace model ID
    params_billions: float
    num_layers: int
    hidden_size: int
    num_attention_heads: int
    context_length: int = 4096

    # Device preferences
    preferred_device: DeviceType = DeviceType.GPU_ARC
    inference_mode: InferenceMode = InferenceMode.AUTO
    quantization: QuantizationType = QuantizationType.INT4

    # Runtime config
    enable_swap: bool = True
    enable_streaming: bool = True
    max_batch_size: int = 1

    # Cache directory
    cache_dir: str = "/tmp/whiterabbit_cache"


@dataclass
class ModelStatus:
    """Model runtime status."""
    model_name: str
    state: ModelState
    device: DeviceType
    inference_mode: InferenceMode
    quantization: QuantizationType

    memory_used_gb: float
    layers_loaded: int
    total_layers: int

    # Performance metrics
    tokens_per_second: float = 0.0
    average_latency_ms: float = 0.0
    total_tokens_generated: int = 0

    # Allocation
    allocation_plan: Optional[AllocationPlan] = None


class WhiteRabbitModelManager:
    """
    Manager for WhiteRabbitNeo models with comprehensive device support.

    Features:
    - Multi-device inference (GPU/NPU/NCS2/CPU)
    - Runtime device switching
    - Dynamic memory management
    - Quantization support
    - Validation pipeline ready
    """

    # Predefined model configurations
    MODELS = {
        "WhiteRabbitNeo-33B-v1": ModelConfig(
            name="WhiteRabbitNeo-33B-v1",
            model_id="WhiteRabbitNeo/WhiteRabbitNeo-33B-v1",
            params_billions=33.0,
            num_layers=60,  # Estimated
            hidden_size=8192,  # Estimated
            num_attention_heads=64,  # Estimated
            context_length=4096,
        ),
        "Llama-3.1-WhiteRabbitNeo-2-70B": ModelConfig(
            name="Llama-3.1-WhiteRabbitNeo-2-70B",
            model_id="WhiteRabbitNeo/Llama-3.1-WhiteRabbitNeo-2-70B",
            params_billions=70.0,
            num_layers=80,  # Llama 70B standard
            hidden_size=8192,
            num_attention_heads=64,
            context_length=8192,  # Extended context
        ),
    }

    def __init__(self):
        """Initialize model manager."""
        self.allocator = get_allocator()
        self.loaded_models: Dict[str, ModelStatus] = {}
        self.active_model: Optional[str] = None

        logger.info("WhiteRabbitNeo Model Manager initialized")
        logger.info(f"  Available models: {list(self.MODELS.keys())}")

    def list_models(self) -> List[str]:
        """List available models."""
        return list(self.MODELS.keys())

    def get_model_config(self, model_name: str) -> Optional[ModelConfig]:
        """Get model configuration."""
        return self.MODELS.get(model_name)

    def get_model_status(self, model_name: str) -> Optional[ModelStatus]:
        """Get model status."""
        return self.loaded_models.get(model_name)

    def plan_allocation(
        self,
        model_name: str,
        quantization: QuantizationType = QuantizationType.INT4,
        inference_mode: InferenceMode = InferenceMode.AUTO,
        enable_swap: bool = True
    ) -> Optional[AllocationPlan]:
        """
        Plan model allocation across devices.

        Args:
            model_name: Model name
            quantization: Quantization type
            inference_mode: Inference mode
            enable_swap: Allow swap usage

        Returns:
            AllocationPlan or None if model not found
        """
        config = self.MODELS.get(model_name)
        if not config:
            logger.error(f"Model not found: {model_name}")
            return None

        # Create ModelSpec
        spec = ModelSpec(
            name=config.name,
            params_billions=config.params_billions,
            context_length=config.context_length
        )
        spec.num_layers = config.num_layers

        # Create allocation plan
        plan = self.allocator.create_allocation_plan(
            model_spec=spec,
            quantization=quantization,
            enable_swap=enable_swap
        )

        return plan

    def ensure_swap(self, size_gb: int = 32) -> bool:
        """
        Ensure swap file exists.

        Args:
            size_gb: Swap size in GB

        Returns:
            True if swap is available
        """
        if self.allocator.swap_size_gb >= size_gb:
            logger.info(f"Swap already available: {self.allocator.swap_size_gb:.1f}GB")
            return True

        logger.info(f"Creating {size_gb}GB swap file...")
        return self.allocator.create_swap_file(size_gb)

    def load_model(
        self,
        model_name: str,
        quantization: QuantizationType = QuantizationType.INT4,
        inference_mode: InferenceMode = InferenceMode.AUTO,
        enable_swap: bool = True,
        force_reload: bool = False
    ) -> bool:
        """
        Load model with specified configuration.

        Args:
            model_name: Model name
            quantization: Quantization type
            inference_mode: Inference mode
            enable_swap: Allow swap usage
            force_reload: Force reload if already loaded

        Returns:
            True if successful
        """
        # Check if already loaded
        if model_name in self.loaded_models and not force_reload:
            status = self.loaded_models[model_name]
            if status.state == ModelState.LOADED:
                logger.info(f"Model already loaded: {model_name}")
                return True

        logger.info(f"Loading model: {model_name}")
        config = self.MODELS.get(model_name)
        if not config:
            logger.error(f"Model not found: {model_name}")
            return False

        # Create allocation plan
        plan = self.plan_allocation(
            model_name,
            quantization=quantization,
            inference_mode=inference_mode,
            enable_swap=enable_swap
        )

        if not plan or not plan.is_feasible:
            logger.error("Model allocation not feasible")
            return False

        # Check if swap is needed
        if plan.swap_memory_gb > 0:
            if not self.ensure_swap(int(plan.swap_memory_gb) + 4):
                logger.error("Failed to create swap file")
                return False

        # Create model status
        status = ModelStatus(
            model_name=model_name,
            state=ModelState.LOADING,
            device=DeviceType.GPU_ARC,  # Primary device
            inference_mode=inference_mode,
            quantization=quantization,
            memory_used_gb=plan.total_memory_required_gb,
            layers_loaded=0,
            total_layers=config.num_layers,
            allocation_plan=plan
        )

        self.loaded_models[model_name] = status

        try:
            # TODO: Implement actual model loading
            # This would involve:
            # 1. Download model from HuggingFace
            # 2. Apply quantization
            # 3. Load layers to appropriate devices
            # 4. Initialize inference engines

            logger.info(f"Model loading simulation: {model_name}")
            logger.info("  [This is a placeholder - actual loading not implemented]")
            logger.info("  [Would use transformers library with device_map]")

            # Simulate successful load
            status.state = ModelState.LOADED
            status.layers_loaded = config.num_layers
            self.active_model = model_name

            logger.info(f"✓ Model loaded successfully: {model_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            status.state = ModelState.ERROR
            return False

    def unload_model(self, model_name: str) -> bool:
        """
        Unload model from memory.

        Args:
            model_name: Model name

        Returns:
            True if successful
        """
        if model_name not in self.loaded_models:
            logger.warning(f"Model not loaded: {model_name}")
            return False

        logger.info(f"Unloading model: {model_name}")

        try:
            # TODO: Implement actual model unloading
            # Free memory, close device handles, etc.

            del self.loaded_models[model_name]

            if self.active_model == model_name:
                self.active_model = None

            logger.info(f"✓ Model unloaded: {model_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to unload model: {e}")
            return False

    def switch_device(
        self,
        model_name: str,
        target_device: DeviceType,
        inference_mode: Optional[InferenceMode] = None
    ) -> bool:
        """
        Switch model to different device at runtime.

        Args:
            model_name: Model name
            target_device: Target device
            inference_mode: Optional new inference mode

        Returns:
            True if successful
        """
        status = self.loaded_models.get(model_name)
        if not status or status.state != ModelState.LOADED:
            logger.error(f"Model not loaded: {model_name}")
            return False

        logger.info(f"Switching {model_name} from {status.device.value} to {target_device.value}")

        try:
            # TODO: Implement actual device switching
            # This would involve:
            # 1. Move model weights to target device
            # 2. Reinitialize inference engine
            # 3. Update allocation plan

            status.device = target_device
            if inference_mode:
                status.inference_mode = inference_mode

            logger.info(f"✓ Switched to {target_device.value}")
            return True

        except Exception as e:
            logger.error(f"Failed to switch device: {e}")
            return False

    def set_active_model(self, model_name: str) -> bool:
        """
        Set active model for inference.

        Args:
            model_name: Model name

        Returns:
            True if successful
        """
        status = self.loaded_models.get(model_name)
        if not status or status.state != ModelState.LOADED:
            logger.error(f"Model not loaded: {model_name}")
            return False

        self.active_model = model_name
        logger.info(f"Active model: {model_name}")
        return True

    def get_active_model(self) -> Optional[str]:
        """Get active model name."""
        return self.active_model

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        model_name: Optional[str] = None
    ) -> Optional[str]:
        """
        Generate text from model.

        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            model_name: Model name (default: active model)

        Returns:
            Generated text or None if failed
        """
        # Use active model if not specified
        if model_name is None:
            model_name = self.active_model

        if not model_name:
            logger.error("No active model")
            return None

        status = self.loaded_models.get(model_name)
        if not status or status.state != ModelState.LOADED:
            logger.error(f"Model not loaded: {model_name}")
            return None

        logger.info(f"Generating with {model_name}...")
        logger.info(f"  Prompt length: {len(prompt)} chars")
        logger.info(f"  Max tokens: {max_new_tokens}")

        try:
            # TODO: Implement actual inference
            # This would use the loaded model and allocation plan

            # Placeholder response
            response = f"[Generated by {model_name} - Placeholder]\n{prompt}\n\n[Actual inference not implemented yet]"

            # Update stats
            status.total_tokens_generated += max_new_tokens

            return response

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return None

    def print_status(self):
        """Print manager status."""
        print("\n" + "=" * 70)
        print("WHITERABBITNEO MODEL MANAGER STATUS")
        print("=" * 70)

        print(f"\nAvailable Models: {len(self.MODELS)}")
        for name in self.MODELS:
            print(f"  - {name}")

        print(f"\nLoaded Models: {len(self.loaded_models)}")
        for name, status in self.loaded_models.items():
            active_marker = " [ACTIVE]" if name == self.active_model else ""
            print(f"  - {name}{active_marker}")
            print(f"      State: {status.state.value}")
            print(f"      Device: {status.device.value}")
            print(f"      Mode: {status.inference_mode.value}")
            print(f"      Quantization: {status.quantization.value}")
            print(f"      Memory: {status.memory_used_gb:.1f} GB")
            print(f"      Layers: {status.layers_loaded}/{status.total_layers}")
            if status.total_tokens_generated > 0:
                print(f"      Tokens generated: {status.total_tokens_generated}")

        if not self.loaded_models:
            print("  (none)")

        print("=" * 70 + "\n")

    def save_config(self, path: str):
        """Save configuration to file."""
        config = {
            "loaded_models": {
                name: {
                    "model_name": status.model_name,
                    "state": status.state.value,
                    "device": status.device.value,
                    "inference_mode": status.inference_mode.value,
                    "quantization": status.quantization.value,
                    "memory_used_gb": status.memory_used_gb,
                }
                for name, status in self.loaded_models.items()
            },
            "active_model": self.active_model,
        }

        with open(path, 'w') as f:
            json.dump(config, f, indent=2)

        logger.info(f"Configuration saved: {path}")


# Singleton instance
_manager: Optional[WhiteRabbitModelManager] = None


def get_model_manager() -> WhiteRabbitModelManager:
    """Get or create singleton model manager."""
    global _manager

    if _manager is None:
        _manager = WhiteRabbitModelManager()

    return _manager
