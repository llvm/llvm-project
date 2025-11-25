#!/usr/bin/env python3
"""
Mixture of Experts - Expert Models

Defines expert model interfaces and implementations for specialized domains.
Supports local models, OpenAI-compatible APIs, and fine-tuned LoRA adapters.
"""

import os
import json
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from pathlib import Path
from enum import Enum

# Optional imports
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    from peft import PeftModel, PeftConfig
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False


class ModelBackend(Enum):
    """Available model backends"""
    LOCAL_TRANSFORMERS = "local_transformers"  # Local HuggingFace models
    OPENAI_COMPATIBLE = "openai_compatible"    # OpenAI API-compatible endpoints
    LORA_ADAPTER = "lora_adapter"              # PEFT LoRA fine-tuned adapters
    LLAMACPP = "llamacpp"                      # llama.cpp for quantized models
    VLLM = "vllm"                              # vLLM for high-throughput serving


@dataclass
class ExpertModelConfig:
    """Configuration for an expert model"""
    name: str
    domain: str  # code, database, security, etc.
    backend: ModelBackend
    model_path: str  # Path or identifier
    context_length: int = 4096
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 2048
    system_prompt: str = ""
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    device: str = "auto"
    lora_adapter_path: Optional[str] = None
    api_base: Optional[str] = None
    api_key: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExpertResponse:
    """Response from an expert model"""
    expert_name: str
    domain: str
    response_text: str
    confidence: float  # 0.0 to 1.0
    tokens_used: int
    inference_time: float  # seconds
    metadata: Dict[str, Any] = field(default_factory=dict)


class ExpertModel(ABC):
    """
    Abstract base class for expert models.

    All expert models must implement this interface to be compatible
    with the MoE system.
    """

    def __init__(self, config: ExpertModelConfig):
        self.config = config
        self.name = config.name
        self.domain = config.domain
        self.loaded = False
        self.inference_count = 0
        self.total_tokens = 0
        self.total_time = 0.0

    @abstractmethod
    def load(self):
        """Load the model into memory."""
        pass

    @abstractmethod
    def unload(self):
        """Unload the model from memory."""
        pass

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> ExpertResponse:
        """
        Generate a response for the given prompt.

        Args:
            prompt: Input prompt
            **kwargs: Additional generation parameters

        Returns:
            ExpertResponse with the model output
        """
        pass

    def get_statistics(self) -> Dict[str, Any]:
        """Get usage statistics for this expert."""
        avg_time = self.total_time / self.inference_count if self.inference_count > 0 else 0

        return {
            "name": self.name,
            "domain": self.domain,
            "backend": self.config.backend.value,
            "loaded": self.loaded,
            "inference_count": self.inference_count,
            "total_tokens": self.total_tokens,
            "total_time": self.total_time,
            "average_time": avg_time,
            "tokens_per_second": self.total_tokens / self.total_time if self.total_time > 0 else 0
        }


class TransformersExpert(ExpertModel):
    """
    Expert model using HuggingFace Transformers.

    Supports local models loaded with transformers library.
    """

    def __init__(self, config: ExpertModelConfig):
        super().__init__(config)
        self.model = None
        self.tokenizer = None

    def load(self):
        """Load model and tokenizer from HuggingFace."""
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers not available. Install with: pip install transformers")

        print(f"[{self.name}] Loading model from {self.config.model_path}...")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)

        # Configure quantization
        load_kwargs = {"torch_dtype": torch.float16}

        if self.config.load_in_8bit:
            load_kwargs["load_in_8bit"] = True
        elif self.config.load_in_4bit:
            load_kwargs["load_in_4bit"] = True

        if self.config.device != "auto":
            load_kwargs["device_map"] = self.config.device

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_path,
            **load_kwargs
        )

        # Load LoRA adapter if specified
        if self.config.lora_adapter_path and PEFT_AVAILABLE:
            print(f"[{self.name}] Loading LoRA adapter from {self.config.lora_adapter_path}...")
            self.model = PeftModel.from_pretrained(self.model, self.config.lora_adapter_path)

        self.loaded = True
        print(f"[{self.name}] Model loaded successfully")

    def unload(self):
        """Unload model from memory."""
        if self.model is not None:
            del self.model
            del self.tokenizer
            self.model = None
            self.tokenizer = None

            if TRANSFORMERS_AVAILABLE and torch.cuda.is_available():
                torch.cuda.empty_cache()

        self.loaded = False
        print(f"[{self.name}] Model unloaded")

    def generate(self, prompt: str, **kwargs) -> ExpertResponse:
        """Generate response using transformers."""
        if not self.loaded:
            raise RuntimeError(f"Model {self.name} not loaded")

        start_time = time.time()

        # Add system prompt if configured
        if self.config.system_prompt:
            full_prompt = f"{self.config.system_prompt}\n\n{prompt}"
        else:
            full_prompt = prompt

        # Tokenize input
        inputs = self.tokenizer(full_prompt, return_tensors="pt")
        input_tokens = inputs.input_ids.shape[1]

        # Move to device
        if torch.cuda.is_available() and self.config.device != "cpu":
            inputs = {k: v.cuda() for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=kwargs.get("max_tokens", self.config.max_tokens),
                temperature=kwargs.get("temperature", self.config.temperature),
                top_p=kwargs.get("top_p", self.config.top_p),
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # Decode output
        response_text = self.tokenizer.decode(outputs[0][input_tokens:], skip_special_tokens=True)

        # Calculate metrics
        output_tokens = outputs.shape[1] - input_tokens
        inference_time = time.time() - start_time
        total_tokens = input_tokens + output_tokens

        # Update statistics
        self.inference_count += 1
        self.total_tokens += total_tokens
        self.total_time += inference_time

        # Estimate confidence (placeholder - would need proper implementation)
        confidence = 0.8  # Default confidence

        return ExpertResponse(
            expert_name=self.name,
            domain=self.domain,
            response_text=response_text,
            confidence=confidence,
            tokens_used=total_tokens,
            inference_time=inference_time,
            metadata={
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "backend": "transformers"
            }
        )


class OpenAICompatibleExpert(ExpertModel):
    """
    Expert model using OpenAI-compatible API.

    Works with OpenAI API, local OpenAI-compatible servers (llama.cpp, vLLM, etc.)
    """

    def __init__(self, config: ExpertModelConfig):
        super().__init__(config)

        try:
            import openai
            self.openai = openai
            self.api_available = True
        except ImportError:
            self.api_available = False
            print(f"[{self.name}] Warning: openai package not available")

    def load(self):
        """Configure API client."""
        if not self.api_available:
            raise ImportError("openai not available. Install with: pip install openai")

        # Configure API
        if self.config.api_base:
            self.openai.api_base = self.config.api_base

        if self.config.api_key:
            self.openai.api_key = self.config.api_key
        elif "OPENAI_API_KEY" in os.environ:
            self.openai.api_key = os.environ["OPENAI_API_KEY"]

        self.loaded = True
        print(f"[{self.name}] API client configured")

    def unload(self):
        """Nothing to unload for API-based models."""
        self.loaded = False

    def generate(self, prompt: str, **kwargs) -> ExpertResponse:
        """Generate response using OpenAI API."""
        if not self.loaded:
            raise RuntimeError(f"Model {self.name} not configured")

        start_time = time.time()

        # Prepare messages
        messages = []
        if self.config.system_prompt:
            messages.append({"role": "system", "content": self.config.system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Call API
        try:
            response = self.openai.ChatCompletion.create(
                model=self.config.model_path,  # model name/ID
                messages=messages,
                temperature=kwargs.get("temperature", self.config.temperature),
                top_p=kwargs.get("top_p", self.config.top_p),
                max_tokens=kwargs.get("max_tokens", self.config.max_tokens)
            )

            response_text = response.choices[0].message.content
            tokens_used = response.usage.total_tokens if hasattr(response.usage, "total_tokens") else 0

            # Update statistics
            inference_time = time.time() - start_time
            self.inference_count += 1
            self.total_tokens += tokens_used
            self.total_time += inference_time

            return ExpertResponse(
                expert_name=self.name,
                domain=self.domain,
                response_text=response_text,
                confidence=0.85,  # API models generally more reliable
                tokens_used=tokens_used,
                inference_time=inference_time,
                metadata={
                    "backend": "openai_compatible",
                    "model": self.config.model_path
                }
            )

        except Exception as e:
            print(f"[{self.name}] API call failed: {e}")
            return ExpertResponse(
                expert_name=self.name,
                domain=self.domain,
                response_text=f"Error: {str(e)}",
                confidence=0.0,
                tokens_used=0,
                inference_time=time.time() - start_time,
                metadata={"error": str(e)}
            )


class ExpertModelRegistry:
    """
    Registry for managing expert models.

    Handles model loading, caching, and lifecycle management.
    """

    def __init__(self, cache_size: int = 3):
        """
        Initialize model registry.

        Args:
            cache_size: Maximum number of models to keep loaded simultaneously
        """
        self.cache_size = cache_size
        self.experts: Dict[str, ExpertModel] = {}
        self.loaded_experts: List[str] = []  # LRU cache

    def register_expert(self, config: ExpertModelConfig) -> ExpertModel:
        """
        Register a new expert model.

        Args:
            config: Expert model configuration

        Returns:
            ExpertModel instance
        """
        # Create expert based on backend
        if config.backend == ModelBackend.LOCAL_TRANSFORMERS:
            expert = TransformersExpert(config)
        elif config.backend == ModelBackend.OPENAI_COMPATIBLE:
            expert = OpenAICompatibleExpert(config)
        elif config.backend == ModelBackend.LORA_ADAPTER:
            expert = TransformersExpert(config)  # LoRA uses transformers backend
        else:
            raise ValueError(f"Unsupported backend: {config.backend}")

        self.experts[config.name] = expert
        print(f"[Registry] Registered expert: {config.name} ({config.domain})")

        return expert

    def get_expert(self, name: str) -> Optional[ExpertModel]:
        """Get an expert by name."""
        return self.experts.get(name)

    def load_expert(self, name: str):
        """
        Load an expert model.

        Uses LRU caching to manage memory.
        """
        expert = self.experts.get(name)
        if not expert:
            raise ValueError(f"Expert not found: {name}")

        # Already loaded
        if expert.loaded:
            # Move to end (most recently used)
            if name in self.loaded_experts:
                self.loaded_experts.remove(name)
            self.loaded_experts.append(name)
            return

        # Check if cache is full
        while len(self.loaded_experts) >= self.cache_size:
            # Unload least recently used
            lru_name = self.loaded_experts.pop(0)
            lru_expert = self.experts[lru_name]
            lru_expert.unload()
            print(f"[Registry] Unloaded LRU expert: {lru_name}")

        # Load expert
        expert.load()
        self.loaded_experts.append(name)

    def unload_expert(self, name: str):
        """Explicitly unload an expert model."""
        expert = self.experts.get(name)
        if not expert:
            return

        expert.unload()
        if name in self.loaded_experts:
            self.loaded_experts.remove(name)

    def get_all_statistics(self) -> Dict[str, Dict]:
        """Get statistics for all experts."""
        return {name: expert.get_statistics() for name, expert in self.experts.items()}

    def save_registry(self, path: str):
        """Save registry configuration to file."""
        registry_data = {
            "experts": [
                {
                    "name": expert.name,
                    "domain": expert.domain,
                    "backend": expert.config.backend.value,
                    "model_path": expert.config.model_path,
                    "statistics": expert.get_statistics()
                }
                for expert in self.experts.values()
            ]
        }

        with open(path, "w") as f:
            json.dump(registry_data, f, indent=2)

        print(f"[Registry] Saved to {path}")


if __name__ == "__main__":
    # Test the expert model system
    print("=" * 80)
    print("Expert Model Registry Test")
    print("=" * 80)

    # Create registry
    registry = ExpertModelRegistry(cache_size=2)

    # Register a few test experts (using OpenAI-compatible API for testing)
    configs = [
        ExpertModelConfig(
            name="code-expert",
            domain="code",
            backend=ModelBackend.OPENAI_COMPATIBLE,
            model_path="deepseek-coder-6.7b",
            system_prompt="You are an expert in code generation and optimization."
        ),
        ExpertModelConfig(
            name="database-expert",
            domain="database",
            backend=ModelBackend.OPENAI_COMPATIBLE,
            model_path="codellama-7b",
            system_prompt="You are an expert in database design and SQL optimization."
        )
    ]

    for config in configs:
        registry.register_expert(config)

    print("\nRegistered experts:")
    for name in registry.experts.keys():
        print(f"  - {name}")

    print("\nRegistry statistics:")
    print(json.dumps(registry.get_all_statistics(), indent=2))
