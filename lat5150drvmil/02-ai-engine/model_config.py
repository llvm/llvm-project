#!/usr/bin/env python3
"""
Centralized Model Configuration

Single source of truth for all model configurations.
Loads from models.json and provides easy access.
"""

import json
import os
from typing import Dict, Optional, List


class ModelConfig:
    """Centralized model configuration manager"""

    def __init__(self, config_path: str = None):
        """
        Initialize model config

        Args:
            config_path: Path to models.json (default: same directory as this file)
        """
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(__file__),
                "models.json"
            )

        self.config_path = config_path
        self._config = self._load_config()

    def _load_config(self) -> Dict:
        """Load configuration from JSON"""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            # Fallback to hardcoded defaults
            return self._get_default_config()
        except json.JSONDecodeError as e:
            print(f"⚠️  Error loading models.json: {e}")
            return self._get_default_config()

    def _get_default_config(self) -> Dict:
        """Get default configuration if JSON not available"""
        return {
            "models": {
                "fast": {
                    "name": "deepseek-r1:1.5b",
                    "description": "Fast general queries",
                    "expected_time_sec": 5,
                    "context_window": 8192,
                    "use_cases": ["quick_answers"]
                },
                "code": {
                    "name": "deepseek-coder:6.7b-instruct",
                    "description": "Code generation",
                    "expected_time_sec": 10,
                    "context_window": 8192,
                    "use_cases": ["code_generation"]
                },
                "quality_code": {
                    "name": "qwen2.5-coder:7b",
                    "description": "High-quality code",
                    "expected_time_sec": 15,
                    "context_window": 8192,
                    "use_cases": ["complex_code"]
                },
                "uncensored_code": {
                    "name": "wizardlm-uncensored-codellama:34b-q4_K_M",
                    "description": "Uncensored (DEFAULT)",
                    "expected_time_sec": 25,
                    "context_window": 8192,
                    "use_cases": ["unrestricted_coding"],
                    "default": True
                },
                "large": {
                    "name": "codellama:70b-q4_K_M",
                    "description": "Large model for review",
                    "expected_time_sec": 60,
                    "context_window": 8192,
                    "use_cases": ["code_review"]
                }
            },
            "model_aliases": {
                "f": "fast",
                "c": "code",
                "q": "quality_code",
                "u": "uncensored_code",
                "l": "large",
                "default": "uncensored_code"
            }
        }

    def get_model_name(self, key: str) -> Optional[str]:
        """
        Get model name by key

        Args:
            key: Model key (e.g., 'fast', 'code', 'f', 'c')

        Returns:
            Model name string (e.g., 'deepseek-r1:1.5b') or None
        """
        # Check if it's an alias
        aliases = self._config.get("model_aliases", {})
        if key in aliases:
            key = aliases[key]

        # Get model
        models = self._config.get("models", {})
        model_info = models.get(key, {})
        return model_info.get("name")

    def get_model_info(self, key: str) -> Optional[Dict]:
        """Get full model information"""
        aliases = self._config.get("model_aliases", {})
        if key in aliases:
            key = aliases[key]

        models = self._config.get("models", {})
        return models.get(key)

    def get_default_model(self) -> str:
        """Get default model key"""
        aliases = self._config.get("model_aliases", {})
        return aliases.get("default", "uncensored_code")

    def get_default_model_name(self) -> str:
        """Get default model name"""
        default_key = self.get_default_model()
        return self.get_model_name(default_key)

    def get_all_models(self) -> Dict:
        """Get all model configurations"""
        return self._config.get("models", {})

    def get_all_model_names(self) -> Dict[str, str]:
        """Get mapping of keys to model names"""
        models = self.get_all_models()
        return {key: info["name"] for key, info in models.items()}

    def get_aliases(self) -> Dict[str, str]:
        """Get all aliases"""
        return self._config.get("model_aliases", {})

    def get_routing_keywords(self) -> Dict[str, List[str]]:
        """Get routing keywords for smart model selection"""
        return self._config.get("routing_keywords", {})

    def resolve_model(self, model_selection: str) -> str:
        """
        Resolve model selection to model name

        Args:
            model_selection: Key, alias, or model name

        Returns:
            Resolved model name
        """
        # Try as key/alias first
        model_name = self.get_model_name(model_selection)
        if model_name:
            return model_name

        # Check if it's already a full model name
        all_names = list(self.get_all_model_names().values())
        if model_selection in all_names:
            return model_selection

        # Default fallback
        return self.get_default_model_name()

    def get_stats(self) -> Dict:
        """Get configuration statistics"""
        models = self.get_all_models()
        aliases = self.get_aliases()

        return {
            "total_models": len(models),
            "total_aliases": len(aliases),
            "default_model": self.get_default_model(),
            "config_path": self.config_path,
            "config_loaded": bool(self._config)
        }


# Global instance (singleton pattern)
_global_config = None


def get_model_config() -> ModelConfig:
    """Get global model configuration instance"""
    global _global_config
    if _global_config is None:
        _global_config = ModelConfig()
    return _global_config


# Convenience functions
def get_model_name(key: str) -> Optional[str]:
    """Get model name by key"""
    return get_model_config().get_model_name(key)


def get_default_model() -> str:
    """Get default model name"""
    return get_model_config().get_default_model_name()


def get_all_models() -> Dict:
    """Get all models"""
    return get_model_config().get_all_models()


# Example usage
if __name__ == "__main__":
    print("Centralized Model Configuration")
    print("=" * 60)

    config = ModelConfig()

    print("\nModel Configuration:")
    stats = config.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print("\nAvailable Models:")
    for key, name in config.get_all_model_names().items():
        info = config.get_model_info(key)
        default_tag = " (DEFAULT)" if info.get("default") else ""
        print(f"  {key}: {name}{default_tag}")
        print(f"      {info['description']}")

    print("\nAliases:")
    for alias, key in config.get_aliases().items():
        print(f"  {alias} → {key}")

    print("\nResolving Examples:")
    examples = ["f", "fast", "uncensored_code", "u", "default"]
    for example in examples:
        resolved = config.resolve_model(example)
        print(f"  '{example}' → {resolved}")
