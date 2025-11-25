#!/usr/bin/env python3
"""
Agent Configuration System with Pydantic Validation

Centralized configuration for all AI agents in the DSMIL framework.
Provides type-safe configuration with validation, defaults, and documentation.

Usage:
    from agent_config import AgentRegistry, get_agent_config

    # Get validated config
    config = get_agent_config("gemini")
    print(config.api_key)  # Type-safe access

    # List all agents
    registry = AgentRegistry()
    for agent in registry.list_agents():
        print(f"{agent.name}: {agent.description}")
"""

from pydantic import BaseModel, Field, SecretStr, field_validator
from typing import Optional, List, Dict, Any, Literal
from enum import Enum
from pathlib import Path
import os
import json


# ============================================================================
# Agent Configuration Models
# ============================================================================

class AgentStatus(str, Enum):
    """Agent availability status"""
    AVAILABLE = "available"
    CONFIGURED = "configured"
    UNAVAILABLE = "unavailable"
    ERROR = "error"


class AgentPriority(str, Enum):
    """Agent routing priority"""
    PRIMARY = "primary"  # Default for all queries
    SECONDARY = "secondary"  # Fallback
    MULTIMODAL_ONLY = "multimodal_only"  # Images/video only
    EXPLICIT_ONLY = "explicit_only"  # Only when explicitly requested
    AUTO_ROUTED = "auto_routed"  # Based on query detection


class LocalAIConfig(BaseModel):
    """Local Ollama configuration"""
    ollama_url: str = Field(
        default="http://localhost:11434",
        description="Ollama API endpoint"
    )
    models: Dict[str, str] = Field(
        default={
            "fast": "whiterabbit-neo-33b",
            "code": "whiterabbit-neo-33b",
            "quality_code": "qwen2.5-coder:7b",
            "uncensored_code": "wizardlm-uncensored-codellama:34b",
            "large": "codellama:70b",
            # Legacy models
            "deepseek_fast": "deepseek-r1:1.5b",
            "deepseek_code": "deepseek-coder:6.7b-instruct",
        },
        description="Available local models"
    )
    enable_gpu: bool = Field(
        default=True,
        description="Enable GPU acceleration if available"
    )
    max_concurrent_requests: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum concurrent inference requests"
    )
    timeout_seconds: int = Field(
        default=120,
        ge=10,
        le=600,
        description="Request timeout in seconds"
    )


class GeminiConfig(BaseModel):
    """Google Gemini configuration"""
    api_key: Optional[SecretStr] = Field(
        default=None,
        description="Gemini API key from https://aistudio.google.com/apikey"
    )
    model: str = Field(
        default="gemini-2.0-flash-exp",
        description="Gemini model to use"
    )
    priority: AgentPriority = AgentPriority.MULTIMODAL_ONLY
    max_retries: int = Field(default=3, ge=1, le=10)
    timeout_seconds: int = Field(default=30, ge=5, le=300)

    @field_validator('api_key', mode='before')
    @classmethod
    def get_api_key_from_env(cls, v):
        """Get API key from environment if not provided"""
        if v is None:
            env_key = os.environ.get('GEMINI_API_KEY')
            if env_key:
                return SecretStr(env_key)
        return v


class OpenAIConfig(BaseModel):
    """OpenAI configuration"""
    api_key: Optional[SecretStr] = Field(
        default=None,
        description="OpenAI API key from https://platform.openai.com/api-keys"
    )
    models: List[str] = Field(
        default=["gpt-4-turbo", "gpt-3.5-turbo", "gpt-4"],
        description="Available OpenAI models"
    )
    default_model: str = Field(default="gpt-4-turbo")
    priority: AgentPriority = AgentPriority.EXPLICIT_ONLY
    max_tokens: int = Field(default=4096, ge=1, le=32000)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    enable_structured_outputs: bool = Field(
        default=True,
        description="Enable Pydantic structured outputs (beta)"
    )

    @field_validator('api_key', mode='before')
    @classmethod
    def get_api_key_from_env(cls, v):
        if v is None:
            env_key = os.environ.get('OPENAI_API_KEY')
            if env_key:
                return SecretStr(env_key)
        return v


class WebSearchConfig(BaseModel):
    """Web search configuration"""
    enabled: bool = Field(default=True)
    provider: Literal["duckduckgo", "searx"] = Field(default="duckduckgo")
    max_results: int = Field(default=5, ge=1, le=20)
    timeout_seconds: int = Field(default=10, ge=1, le=60)
    cache_ttl_seconds: int = Field(
        default=3600,
        ge=0,
        description="Cache time-to-live (0 = no cache)"
    )


class ShodanConfig(BaseModel):
    """Shodan threat intelligence configuration"""
    api_key: Optional[SecretStr] = Field(
        default=None,
        description="Shodan API key from https://account.shodan.io/"
    )
    enabled: bool = Field(default=False)
    max_results: int = Field(default=10, ge=1, le=100)
    default_facets: List[str] = Field(
        default=["country", "org", "port"],
        description="Default facets to retrieve"
    )

    @field_validator('api_key', mode='before')
    @classmethod
    def get_api_key_from_env(cls, v):
        if v is None:
            env_key = os.environ.get('SHODAN_API_KEY')
            if env_key:
                return SecretStr(env_key)
                # Enable if API key is set
                cls.enabled = True
        return v


class RAGConfig(BaseModel):
    """RAG system configuration"""
    enabled: bool = Field(default=True)
    index_path: Path = Field(
        default=Path.home() / ".rag_index",
        description="Path to RAG index storage"
    )
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Sentence transformer model for embeddings"
    )
    chunk_size: int = Field(default=512, ge=128, le=2048)
    chunk_overlap: int = Field(default=50, ge=0, le=512)
    top_k_results: int = Field(default=5, ge=1, le=20)
    min_relevance_score: float = Field(default=0.0, ge=0.0, le=1.0)


class ACEConfig(BaseModel):
    """ACE-FCA (Advanced Context Engineering) configuration"""
    enabled: bool = Field(default=True)
    max_context_tokens: int = Field(default=8192, ge=2048, le=32000)
    target_utilization: float = Field(
        default=0.6,
        ge=0.3,
        le=0.9,
        description="Target context window utilization"
    )
    enable_human_review: bool = Field(default=True)
    compaction_threshold: float = Field(
        default=0.8,
        ge=0.5,
        le=0.95,
        description="Trigger compaction at this utilization"
    )


class ParallelExecutionConfig(BaseModel):
    """Parallel agent execution configuration"""
    enabled: bool = Field(default=True)
    max_concurrent_agents: int = Field(default=3, ge=1, le=10)
    enable_worktrees: bool = Field(
        default=True,
        description="Enable git worktree management for parallel development"
    )
    task_distribution: Literal["round_robin", "load_balanced", "priority"] = Field(
        default="load_balanced"
    )


# ============================================================================
# Master Configuration
# ============================================================================

class DSMILAIConfig(BaseModel):
    """Master DSMIL AI configuration"""
    # Core components
    local_ai: LocalAIConfig = Field(default_factory=LocalAIConfig)
    gemini: GeminiConfig = Field(default_factory=GeminiConfig)
    openai: OpenAIConfig = Field(default_factory=OpenAIConfig)

    # Search & intelligence
    web_search: WebSearchConfig = Field(default_factory=WebSearchConfig)
    shodan: ShodanConfig = Field(default_factory=ShodanConfig)

    # Knowledge & context
    rag: RAGConfig = Field(default_factory=RAGConfig)
    ace: ACEConfig = Field(default_factory=ACEConfig)

    # Execution
    parallel_execution: ParallelExecutionConfig = Field(
        default_factory=ParallelExecutionConfig
    )

    # Global settings
    default_backend: Literal["local", "gemini", "openai"] = Field(default="local")
    enable_pydantic_mode: bool = Field(
        default=True,
        description="Enable type-safe Pydantic responses by default"
    )
    enable_attestation: bool = Field(
        default=True,
        description="Enable TPM hardware attestation"
    )
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO")

    def save(self, path: Path):
        """Save configuration to JSON file"""
        with open(path, 'w') as f:
            # Convert to dict, handling SecretStr
            config_dict = json.loads(self.model_dump_json())
            json.dump(config_dict, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> 'DSMILAIConfig':
        """Load configuration from JSON file"""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls.model_validate(config_dict)


# ============================================================================
# Agent Registry
# ============================================================================

class AgentInfo(BaseModel):
    """Information about a registered agent"""
    name: str
    description: str
    priority: AgentPriority
    status: AgentStatus
    capabilities: List[str]
    config_class: Optional[str] = None


class AgentRegistry:
    """Central registry of all AI agents"""

    AGENTS = [
        AgentInfo(
            name="local",
            description="Local Ollama models (WhiteRabbitNeo, Qwen)",
            priority=AgentPriority.PRIMARY,
            status=AgentStatus.AVAILABLE,
            capabilities=[
                "General queries",
                "Code generation",
                "Code review",
                "Fast inference",
                "Privacy-preserving (local)",
                "Zero cost",
                "Hardware attestation",
                "Multi-device (NPU, GPU, NCS2)",
                "Dual-model validation",
            ],
            config_class="LocalAIConfig"
        ),
        AgentInfo(
            name="gemini",
            description="Google Gemini Pro",
            priority=AgentPriority.MULTIMODAL_ONLY,
            status=AgentStatus.CONFIGURED,
            capabilities=[
                "Multimodal (images, video)",
                "Document analysis",
                "Visual question answering",
                "OCR and text extraction",
            ],
            config_class="GeminiConfig"
        ),
        AgentInfo(
            name="openai",
            description="OpenAI GPT-4/GPT-3.5",
            priority=AgentPriority.EXPLICIT_ONLY,
            status=AgentStatus.CONFIGURED,
            capabilities=[
                "Advanced reasoning",
                "Structured outputs (beta)",
                "Code generation with validation",
                "JSON schema compliance",
            ],
            config_class="OpenAIConfig"
        ),
        AgentInfo(
            name="geospatial",
            description="Geospatial analytics and OSINT",
            priority=AgentPriority.AUTO_ROUTED,
            status=AgentStatus.AVAILABLE,
            capabilities=[
                "Threat-intel mapping",
                "Infrastructure visualization",
                "KML/GeoJSON/Shapefile processing",
                "OSINT geolocation",
            ]
        ),
        AgentInfo(
            name="rdkit",
            description="Cheminformatics and drug discovery",
            priority=AgentPriority.AUTO_ROUTED,
            status=AgentStatus.AVAILABLE,
            capabilities=[
                "Molecule parsing (SMILES, InChI)",
                "Descriptor calculation",
                "Fingerprint generation",
                "Similarity search",
                "Drug-likeness assessment",
            ]
        ),
        AgentInfo(
            name="nmda",
            description="NMDA agonist analysis",
            priority=AgentPriority.AUTO_ROUTED,
            status=AgentStatus.AVAILABLE,
            capabilities=[
                "NMDA receptor binding prediction",
                "Antidepressant efficacy scoring",
                "Safety profile analysis",
                "Novel compound assessment",
            ]
        ),
        AgentInfo(
            name="nps",
            description="Novel psychoactive substance analysis",
            priority=AgentPriority.AUTO_ROUTED,
            status=AgentStatus.AVAILABLE,
            capabilities=[
                "NPS classification",
                "Abuse potential prediction (0-10 scale)",
                "Receptor binding (6 systems)",
                "Neurotoxicity assessment",
                "DEA scheduling recommendations",
                "Dark web proliferation prediction",
            ]
        ),
    ]

    def list_agents(self, priority: Optional[AgentPriority] = None) -> List[AgentInfo]:
        """List all agents, optionally filtered by priority"""
        if priority:
            return [a for a in self.AGENTS if a.priority == priority]
        return self.AGENTS

    def get_agent(self, name: str) -> Optional[AgentInfo]:
        """Get agent info by name"""
        for agent in self.AGENTS:
            if agent.name == name:
                return agent
        return None


# ============================================================================
# Convenience Functions
# ============================================================================

def get_default_config() -> DSMILAIConfig:
    """Get default configuration with environment variables loaded"""
    return DSMILAIConfig()


def load_config(path: Path = None) -> DSMILAIConfig:
    """Load configuration from file or environment"""
    if path and path.exists():
        return DSMILAIConfig.load(path)
    return get_default_config()


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "list":
        # List all agents
        registry = AgentRegistry()
        print("="*70)
        print("DSMIL AI Agent Registry")
        print("="*70)
        for agent in registry.list_agents():
            print(f"\n{agent.name} ({agent.priority.value})")
            print(f"  Status: {agent.status.value}")
            print(f"  Description: {agent.description}")
            print(f"  Capabilities:")
            for cap in agent.capabilities:
                print(f"    - {cap}")

    elif len(sys.argv) > 1 and sys.argv[1] == "save":
        # Save default config
        config = get_default_config()
        output_path = Path(sys.argv[2] if len(sys.argv) > 2 else "dsmil_ai_config.json")
        config.save(output_path)
        print(f"✓ Configuration saved to: {output_path}")

    else:
        # Show current config
        config = get_default_config()
        print("="*70)
        print("DSMIL AI Configuration (Current)")
        print("="*70)
        print(config.model_dump_json(indent=2))
        print("\nUsage:")
        print("  python3 agent_config.py list     # List all agents")
        print("  python3 agent_config.py save     # Save default config")
