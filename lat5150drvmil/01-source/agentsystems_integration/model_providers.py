#!/usr/bin/env python3
"""
LAT5150 DRVMIL - Multi-Model Provider Abstraction
Support for multiple AI model providers with unified interface

Based on: AgentSystems multi-provider architecture
Providers: Anthropic Claude, OpenAI, Ollama (local), AWS Bedrock, Custom
"""

import os
import json
import asyncio
import httpx
from typing import Dict, List, Optional, Any, AsyncIterator
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
from enum import Enum
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] ModelProvider: %(message)s'
)
logger = logging.getLogger(__name__)


class ModelProvider(str, Enum):
    """Supported model providers"""
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    OLLAMA = "ollama"
    BEDROCK = "bedrock"
    CUSTOM = "custom"


@dataclass
class ModelInfo:
    """Model information"""
    name: str
    provider: str
    context_window: int
    max_tokens: int
    supports_streaming: bool
    supports_tools: bool
    cost_per_1k_tokens: Optional[float] = None

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class CompletionResponse:
    """Unified completion response"""
    text: str
    model: str
    provider: str
    usage: Dict[str, int]
    finish_reason: str
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict:
        return asdict(self)


class BaseModelProvider(ABC):
    """Abstract base class for model providers"""

    def __init__(self, api_key: Optional[str] = None, **kwargs):
        self.api_key = api_key
        self.config = kwargs

    @abstractmethod
    async def complete(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs
    ) -> CompletionResponse:
        """Generate completion"""
        pass

    @abstractmethod
    async def stream_complete(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs
    ) -> AsyncIterator[str]:
        """Generate streaming completion"""
        pass

    @abstractmethod
    def list_models(self) -> List[ModelInfo]:
        """List available models"""
        pass

    @abstractmethod
    def get_default_model(self) -> str:
        """Get default model for this provider"""
        pass


class AnthropicProvider(BaseModelProvider):
    """Anthropic Claude provider"""

    MODELS = {
        "claude-opus-4-5-20250929": ModelInfo(
            name="claude-opus-4-5-20250929",
            provider="anthropic",
            context_window=200000,
            max_tokens=8192,
            supports_streaming=True,
            supports_tools=True,
            cost_per_1k_tokens=15.0
        ),
        "claude-sonnet-4-5-20250929": ModelInfo(
            name="claude-sonnet-4-5-20250929",
            provider="anthropic",
            context_window=200000,
            max_tokens=8192,
            supports_streaming=True,
            supports_tools=True,
            cost_per_1k_tokens=3.0
        ),
        "claude-haiku-3-5-20250305": ModelInfo(
            name="claude-haiku-3-5-20250305",
            provider="anthropic",
            context_window=200000,
            max_tokens=8192,
            supports_streaming=True,
            supports_tools=True,
            cost_per_1k_tokens=0.8
        ),
    }

    def __init__(self, api_key: str, **kwargs):
        super().__init__(api_key, **kwargs)
        self.base_url = "https://api.anthropic.com/v1"

    async def complete(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs
    ) -> CompletionResponse:
        """Generate completion using Claude API"""

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }

        payload = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/messages",
                headers=headers,
                json=payload,
                timeout=60.0
            )

            if response.status_code != 200:
                raise Exception(f"Anthropic API error: {response.status_code} - {response.text}")

            data = response.json()

            return CompletionResponse(
                text=data["content"][0]["text"],
                model=model,
                provider="anthropic",
                usage={
                    "input_tokens": data["usage"]["input_tokens"],
                    "output_tokens": data["usage"]["output_tokens"]
                },
                finish_reason=data["stop_reason"],
                metadata={"id": data["id"]}
            )

    async def stream_complete(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs
    ) -> AsyncIterator[str]:
        """Generate streaming completion"""

        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }

        payload = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True
        }

        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/messages",
                headers=headers,
                json=payload,
                timeout=60.0
            ) as response:
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str == "[DONE]":
                            break

                        try:
                            data = json.loads(data_str)
                            if data["type"] == "content_block_delta":
                                yield data["delta"]["text"]
                        except:
                            continue

    def list_models(self) -> List[ModelInfo]:
        """List available Claude models"""
        return list(self.MODELS.values())

    def get_default_model(self) -> str:
        """Get default Claude model"""
        return "claude-sonnet-4-5-20250929"


class OpenAIProvider(BaseModelProvider):
    """OpenAI GPT provider"""

    MODELS = {
        "gpt-4-turbo": ModelInfo(
            name="gpt-4-turbo",
            provider="openai",
            context_window=128000,
            max_tokens=4096,
            supports_streaming=True,
            supports_tools=True,
            cost_per_1k_tokens=10.0
        ),
        "gpt-4": ModelInfo(
            name="gpt-4",
            provider="openai",
            context_window=8192,
            max_tokens=4096,
            supports_streaming=True,
            supports_tools=True,
            cost_per_1k_tokens=30.0
        ),
        "gpt-3.5-turbo": ModelInfo(
            name="gpt-3.5-turbo",
            provider="openai",
            context_window=16385,
            max_tokens=4096,
            supports_streaming=True,
            supports_tools=True,
            cost_per_1k_tokens=1.5
        ),
    }

    def __init__(self, api_key: str, **kwargs):
        super().__init__(api_key, **kwargs)
        self.base_url = "https://api.openai.com/v1"

    async def complete(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs
    ) -> CompletionResponse:
        """Generate completion using OpenAI API"""

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=60.0
            )

            if response.status_code != 200:
                raise Exception(f"OpenAI API error: {response.status_code} - {response.text}")

            data = response.json()

            return CompletionResponse(
                text=data["choices"][0]["message"]["content"],
                model=model,
                provider="openai",
                usage={
                    "input_tokens": data["usage"]["prompt_tokens"],
                    "output_tokens": data["usage"]["completion_tokens"]
                },
                finish_reason=data["choices"][0]["finish_reason"],
                metadata={"id": data["id"]}
            )

    async def stream_complete(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs
    ) -> AsyncIterator[str]:
        """Generate streaming completion"""

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True
        }

        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=60.0
            ) as response:
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str == "[DONE]":
                            break

                        try:
                            data = json.loads(data_str)
                            if "choices" in data and len(data["choices"]) > 0:
                                delta = data["choices"][0].get("delta", {})
                                if "content" in delta:
                                    yield delta["content"]
                        except:
                            continue

    def list_models(self) -> List[ModelInfo]:
        """List available OpenAI models"""
        return list(self.MODELS.values())

    def get_default_model(self) -> str:
        """Get default OpenAI model"""
        return "gpt-4-turbo"


class OllamaProvider(BaseModelProvider):
    """Ollama local LLM provider"""

    def __init__(self, endpoint: str = "http://localhost:11434", **kwargs):
        super().__init__(api_key=None, **kwargs)
        self.endpoint = endpoint

    async def complete(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs
    ) -> CompletionResponse:
        """Generate completion using Ollama"""

        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.endpoint}/api/generate",
                json=payload,
                timeout=120.0
            )

            if response.status_code != 200:
                raise Exception(f"Ollama API error: {response.status_code} - {response.text}")

            data = response.json()

            return CompletionResponse(
                text=data["response"],
                model=model,
                provider="ollama",
                usage={
                    "input_tokens": data.get("prompt_eval_count", 0),
                    "output_tokens": data.get("eval_count", 0)
                },
                finish_reason="stop",
                metadata={
                    "total_duration": data.get("total_duration", 0),
                    "load_duration": data.get("load_duration", 0)
                }
            )

    async def stream_complete(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs
    ) -> AsyncIterator[str]:
        """Generate streaming completion"""

        payload = {
            "model": model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }

        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                f"{self.endpoint}/api/generate",
                json=payload,
                timeout=120.0
            ) as response:
                async for line in response.aiter_lines():
                    try:
                        data = json.loads(line)
                        if "response" in data:
                            yield data["response"]
                    except:
                        continue

    def list_models(self) -> List[ModelInfo]:
        """List available Ollama models"""
        # In real implementation, would query Ollama API
        return [
            ModelInfo(
                name="llama3.2:latest",
                provider="ollama",
                context_window=8192,
                max_tokens=4096,
                supports_streaming=True,
                supports_tools=False,
                cost_per_1k_tokens=0.0  # Local, free
            ),
            ModelInfo(
                name="codellama:latest",
                provider="ollama",
                context_window=16384,
                max_tokens=4096,
                supports_streaming=True,
                supports_tools=False,
                cost_per_1k_tokens=0.0
            ),
            ModelInfo(
                name="mixtral:latest",
                provider="ollama",
                context_window=32768,
                max_tokens=4096,
                supports_streaming=True,
                supports_tools=False,
                cost_per_1k_tokens=0.0
            ),
        ]

    def get_default_model(self) -> str:
        """Get default Ollama model"""
        return "llama3.2:latest"


class ModelProviderManager:
    """
    Manager for multiple model providers

    Allows switching between providers and models dynamically
    """

    def __init__(self):
        self.providers: Dict[str, BaseModelProvider] = {}
        self.default_provider: Optional[str] = None

    def register_provider(
        self,
        name: str,
        provider: BaseModelProvider,
        set_as_default: bool = False
    ):
        """Register a model provider"""
        self.providers[name] = provider
        logger.info(f"Registered provider: {name}")

        if set_as_default or self.default_provider is None:
            self.default_provider = name
            logger.info(f"Set default provider: {name}")

    def get_provider(self, name: Optional[str] = None) -> BaseModelProvider:
        """Get a provider by name"""
        provider_name = name or self.default_provider

        if not provider_name or provider_name not in self.providers:
            raise ValueError(f"Provider '{provider_name}' not found")

        return self.providers[provider_name]

    async def complete(
        self,
        prompt: str,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        **kwargs
    ) -> CompletionResponse:
        """Generate completion using specified or default provider"""

        provider_obj = self.get_provider(provider)

        # Use default model if not specified
        if not model:
            model = provider_obj.get_default_model()

        return await provider_obj.complete(prompt, model, **kwargs)

    async def stream_complete(
        self,
        prompt: str,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        **kwargs
    ) -> AsyncIterator[str]:
        """Generate streaming completion"""

        provider_obj = self.get_provider(provider)

        if not model:
            model = provider_obj.get_default_model()

        async for chunk in provider_obj.stream_complete(prompt, model, **kwargs):
            yield chunk

    def list_all_models(self) -> Dict[str, List[ModelInfo]]:
        """List all models from all providers"""
        all_models = {}

        for name, provider in self.providers.items():
            all_models[name] = provider.list_models()

        return all_models

    def list_providers(self) -> List[str]:
        """List registered providers"""
        return list(self.providers.keys())


# Example usage
async def main():
    """Test multi-model provider system"""

    # Initialize manager
    manager = ModelProviderManager()

    # Register Anthropic (if API key available)
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    if anthropic_key:
        manager.register_provider(
            "anthropic",
            AnthropicProvider(api_key=anthropic_key),
            set_as_default=True
        )

    # Register OpenAI (if API key available)
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        manager.register_provider(
            "openai",
            OpenAIProvider(api_key=openai_key)
        )

    # Register Ollama (always available if running)
    manager.register_provider(
        "ollama",
        OllamaProvider(endpoint="http://localhost:11434")
    )

    # List all models
    print("\n=== Available Models ===")
    all_models = manager.list_all_models()
    for provider_name, models in all_models.items():
        print(f"\n{provider_name.upper()}:")
        for model in models:
            print(f"  - {model.name} (context: {model.context_window}, cost: ${model.cost_per_1k_tokens or 0}/1k tokens)")

    # Example completion (requires valid API key or running Ollama)
    # response = await manager.complete(
    #     prompt="Explain quantum computing in one sentence",
    #     provider="ollama",  # or "anthropic", "openai"
    #     temperature=0.7
    # )
    # print(f"\nResponse: {response.text}")


if __name__ == "__main__":
    asyncio.run(main())
