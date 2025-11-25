#!/usr/bin/env python3
"""
Configuration Management - Production-ready configuration system

Handles all configuration for the Enhanced AI Engine:
- Environment variables
- Database connections (PostgreSQL, Redis, ChromaDB)
- API keys and secrets
- Component enable/disable flags
- Performance tuning parameters
- Logging and monitoring configuration

Usage:
    from config import Config
    config = Config()

    # Access configuration
    db_url = config.database.postgres_url
    redis_host = config.cache.redis_host

    # Or load from environment
    config = Config.from_env()
"""

import os
import json
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field, asdict
from pathlib import Path
from enum import Enum


class Environment(Enum):
    """Deployment environment"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


@dataclass
class DatabaseConfig:
    """Database configuration"""
    # PostgreSQL (conversation history, long-term memory)
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "lat5150_ai"
    postgres_user: str = "lat5150"
    postgres_password: str = ""
    postgres_url: Optional[str] = None  # Full connection URL

    # SQLite (event store)
    sqlite_path: str = "./agent_events.db"

    # ChromaDB (vector embeddings)
    chroma_path: str = "./chroma_db"
    chroma_host: Optional[str] = None  # If using client/server mode
    chroma_port: Optional[int] = None

    def __post_init__(self):
        """Build full URLs if not provided"""
        if not self.postgres_url:
            self.postgres_url = (
                f"postgresql://{self.postgres_user}:{self.postgres_password}"
                f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
            )


@dataclass
class CacheConfig:
    """Cache configuration"""
    # Redis
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: str = ""
    redis_url: Optional[str] = None

    # Cache settings
    enable_response_cache: bool = True
    cache_ttl_seconds: int = 3600  # 1 hour
    max_cache_size_mb: int = 1024  # 1GB

    def __post_init__(self):
        """Build full URLs if not provided"""
        if not self.redis_url:
            auth = f":{self.redis_password}@" if self.redis_password else ""
            self.redis_url = f"redis://{auth}{self.redis_host}:{self.redis_port}/{self.redis_db}"


@dataclass
class APIKeysConfig:
    """API keys and secrets"""
    # DIRECTEYE Intelligence Platform
    directeye_api_key: str = ""
    directeye_api_url: str = "https://api.directeye.io"

    # OpenAI/Anthropic (if using external LLMs)
    openai_api_key: str = ""
    anthropic_api_key: str = ""

    # Other services
    github_token: str = ""
    slack_webhook: str = ""  # For notifications
    pagerduty_key: str = ""  # For alerting


@dataclass
class ComponentsConfig:
    """Component enable/disable flags"""
    # Core components
    enable_self_improvement: bool = True
    enable_dsmil_integration: bool = True
    enable_ram_context: bool = True
    enable_forensics_knowledge: bool = True

    # Phase 1-3 (ai-that-works)
    enable_multi_model_eval: bool = True
    enable_decaying_memory: bool = True
    enable_event_driven: bool = True
    enable_entity_resolution: bool = True
    enable_dynamic_schemas: bool = True
    enable_agentic_rag: bool = True
    enable_human_in_loop: bool = True

    # Phase 4 (Production extensions)
    enable_mcp_selector: bool = True
    enable_threat_intel: bool = True
    enable_blockchain_tools: bool = True
    enable_osint_workflows: bool = True
    enable_advanced_analytics: bool = True


@dataclass
class PerformanceConfig:
    """Performance tuning parameters"""
    # Context windows
    max_context_window: int = 131072  # 131K tokens
    optimal_context_window: int = 65536  # 50% of max

    # Memory
    max_working_memory_tokens: int = 8192
    max_short_term_memory_tokens: int = 32768
    max_long_term_memory_tokens: int = 131072
    ram_context_size_mb: int = 512

    # Concurrency
    max_concurrent_queries: int = 10
    max_parallel_model_eval: int = 4

    # Timeouts
    query_timeout_seconds: int = 120
    mcp_tool_timeout_seconds: int = 60
    hilp_approval_timeout_seconds: int = 300

    # RAG
    rag_top_k_results: int = 10
    rag_similarity_threshold: float = 0.7

    # MCP Tool Selector
    mcp_optimize_for: str = "balanced"  # quality, speed, cost, balanced


@dataclass
class LoggingConfig:
    """Logging configuration"""
    # Log levels
    log_level: str = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    log_format: str = "json"  # json, text

    # Log destinations
    log_to_console: bool = True
    log_to_file: bool = True
    log_file_path: str = "./logs/ai_engine.log"
    log_file_max_mb: int = 100
    log_file_backup_count: int = 5

    # Structured logging
    enable_structured_logging: bool = True
    log_correlation_id: bool = True
    log_performance_metrics: bool = True


@dataclass
class MonitoringConfig:
    """Monitoring and observability configuration"""
    # Metrics
    enable_metrics: bool = True
    metrics_port: int = 9090

    # Tracing
    enable_tracing: bool = False
    jaeger_endpoint: str = ""

    # Health checks
    health_check_enabled: bool = True
    health_check_port: int = 8080

    # Alerting
    enable_alerting: bool = False
    alert_on_errors: bool = True
    alert_on_high_latency: bool = True
    alert_latency_threshold_ms: int = 5000


@dataclass
class SecurityConfig:
    """Security configuration"""
    # Authentication
    require_authentication: bool = False
    api_key_header: str = "X-API-Key"
    allowed_api_keys: List[str] = field(default_factory=list)

    # Rate limiting
    enable_rate_limiting: bool = True
    rate_limit_per_minute: int = 60
    rate_limit_per_hour: int = 1000

    # Data protection
    encrypt_at_rest: bool = False
    encrypt_in_transit: bool = True
    pii_detection: bool = True
    pii_redaction: bool = False

    # Audit
    audit_all_queries: bool = True
    audit_log_path: str = "./logs/audit.log"


class Config:
    """Main configuration class"""

    def __init__(
        self,
        environment: Environment = Environment.DEVELOPMENT,
        database: Optional[DatabaseConfig] = None,
        cache: Optional[CacheConfig] = None,
        api_keys: Optional[APIKeysConfig] = None,
        components: Optional[ComponentsConfig] = None,
        performance: Optional[PerformanceConfig] = None,
        logging: Optional[LoggingConfig] = None,
        monitoring: Optional[MonitoringConfig] = None,
        security: Optional[SecurityConfig] = None,
    ):
        """
        Initialize configuration

        Args:
            environment: Deployment environment
            database: Database configuration
            cache: Cache configuration
            api_keys: API keys and secrets
            components: Component enable/disable flags
            performance: Performance tuning parameters
            logging: Logging configuration
            monitoring: Monitoring configuration
            security: Security configuration
        """
        self.environment = environment
        self.database = database or DatabaseConfig()
        self.cache = cache or CacheConfig()
        self.api_keys = api_keys or APIKeysConfig()
        self.components = components or ComponentsConfig()
        self.performance = performance or PerformanceConfig()
        self.logging = logging or LoggingConfig()
        self.monitoring = monitoring or MonitoringConfig()
        self.security = security or SecurityConfig()

    @classmethod
    def from_env(cls, env_file: str = ".env") -> "Config":
        """
        Load configuration from environment variables

        Looks for .env file first, then falls back to system environment.

        Args:
            env_file: Path to .env file

        Returns:
            Config instance loaded from environment
        """
        # Load .env file if it exists
        if os.path.exists(env_file):
            with open(env_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        os.environ[key.strip()] = value.strip()

        # Determine environment
        env_name = os.getenv("ENVIRONMENT", "development")
        environment = Environment(env_name.lower())

        # Database config
        database = DatabaseConfig(
            postgres_host=os.getenv("POSTGRES_HOST", "localhost"),
            postgres_port=int(os.getenv("POSTGRES_PORT", "5432")),
            postgres_db=os.getenv("POSTGRES_DB", "lat5150_ai"),
            postgres_user=os.getenv("POSTGRES_USER", "lat5150"),
            postgres_password=os.getenv("POSTGRES_PASSWORD", ""),
            postgres_url=os.getenv("POSTGRES_URL"),
            sqlite_path=os.getenv("SQLITE_PATH", "./agent_events.db"),
            chroma_path=os.getenv("CHROMA_PATH", "./chroma_db"),
            chroma_host=os.getenv("CHROMA_HOST"),
            chroma_port=int(os.getenv("CHROMA_PORT", "0")) if os.getenv("CHROMA_PORT") else None,
        )

        # Cache config
        cache = CacheConfig(
            redis_host=os.getenv("REDIS_HOST", "localhost"),
            redis_port=int(os.getenv("REDIS_PORT", "6379")),
            redis_db=int(os.getenv("REDIS_DB", "0")),
            redis_password=os.getenv("REDIS_PASSWORD", ""),
            redis_url=os.getenv("REDIS_URL"),
            enable_response_cache=os.getenv("ENABLE_RESPONSE_CACHE", "true").lower() == "true",
            cache_ttl_seconds=int(os.getenv("CACHE_TTL_SECONDS", "3600")),
        )

        # API keys
        api_keys = APIKeysConfig(
            directeye_api_key=os.getenv("DIRECTEYE_API_KEY", ""),
            directeye_api_url=os.getenv("DIRECTEYE_API_URL", "https://api.directeye.io"),
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
            github_token=os.getenv("GITHUB_TOKEN", ""),
            slack_webhook=os.getenv("SLACK_WEBHOOK", ""),
            pagerduty_key=os.getenv("PAGERDUTY_KEY", ""),
        )

        # Components
        components = ComponentsConfig(
            enable_self_improvement=os.getenv("ENABLE_SELF_IMPROVEMENT", "true").lower() == "true",
            enable_dsmil_integration=os.getenv("ENABLE_DSMIL_INTEGRATION", "true").lower() == "true",
            enable_ram_context=os.getenv("ENABLE_RAM_CONTEXT", "true").lower() == "true",
            enable_forensics_knowledge=os.getenv("ENABLE_FORENSICS_KNOWLEDGE", "true").lower() == "true",
            enable_multi_model_eval=os.getenv("ENABLE_MULTI_MODEL_EVAL", "true").lower() == "true",
            enable_decaying_memory=os.getenv("ENABLE_DECAYING_MEMORY", "true").lower() == "true",
            enable_event_driven=os.getenv("ENABLE_EVENT_DRIVEN", "true").lower() == "true",
            enable_entity_resolution=os.getenv("ENABLE_ENTITY_RESOLUTION", "true").lower() == "true",
            enable_dynamic_schemas=os.getenv("ENABLE_DYNAMIC_SCHEMAS", "true").lower() == "true",
            enable_agentic_rag=os.getenv("ENABLE_AGENTIC_RAG", "true").lower() == "true",
            enable_human_in_loop=os.getenv("ENABLE_HUMAN_IN_LOOP", "true").lower() == "true",
            enable_mcp_selector=os.getenv("ENABLE_MCP_SELECTOR", "true").lower() == "true",
            enable_threat_intel=os.getenv("ENABLE_THREAT_INTEL", "true").lower() == "true",
            enable_blockchain_tools=os.getenv("ENABLE_BLOCKCHAIN_TOOLS", "true").lower() == "true",
            enable_osint_workflows=os.getenv("ENABLE_OSINT_WORKFLOWS", "true").lower() == "true",
            enable_advanced_analytics=os.getenv("ENABLE_ADVANCED_ANALYTICS", "true").lower() == "true",
        )

        # Performance
        performance = PerformanceConfig(
            max_context_window=int(os.getenv("MAX_CONTEXT_WINDOW", "131072")),
            optimal_context_window=int(os.getenv("OPTIMAL_CONTEXT_WINDOW", "65536")),
            max_working_memory_tokens=int(os.getenv("MAX_WORKING_MEMORY_TOKENS", "8192")),
            ram_context_size_mb=int(os.getenv("RAM_CONTEXT_SIZE_MB", "512")),
            max_concurrent_queries=int(os.getenv("MAX_CONCURRENT_QUERIES", "10")),
            query_timeout_seconds=int(os.getenv("QUERY_TIMEOUT_SECONDS", "120")),
            rag_top_k_results=int(os.getenv("RAG_TOP_K_RESULTS", "10")),
            mcp_optimize_for=os.getenv("MCP_OPTIMIZE_FOR", "balanced"),
        )

        # Logging
        logging = LoggingConfig(
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            log_format=os.getenv("LOG_FORMAT", "json"),
            log_to_console=os.getenv("LOG_TO_CONSOLE", "true").lower() == "true",
            log_to_file=os.getenv("LOG_TO_FILE", "true").lower() == "true",
            log_file_path=os.getenv("LOG_FILE_PATH", "./logs/ai_engine.log"),
        )

        # Monitoring
        monitoring = MonitoringConfig(
            enable_metrics=os.getenv("ENABLE_METRICS", "true").lower() == "true",
            metrics_port=int(os.getenv("METRICS_PORT", "9090")),
            health_check_enabled=os.getenv("HEALTH_CHECK_ENABLED", "true").lower() == "true",
            health_check_port=int(os.getenv("HEALTH_CHECK_PORT", "8080")),
        )

        # Security
        security = SecurityConfig(
            require_authentication=os.getenv("REQUIRE_AUTHENTICATION", "false").lower() == "true",
            enable_rate_limiting=os.getenv("ENABLE_RATE_LIMITING", "true").lower() == "true",
            rate_limit_per_minute=int(os.getenv("RATE_LIMIT_PER_MINUTE", "60")),
            audit_all_queries=os.getenv("AUDIT_ALL_QUERIES", "true").lower() == "true",
            audit_log_path=os.getenv("AUDIT_LOG_PATH", "./logs/audit.log"),
        )

        return cls(
            environment=environment,
            database=database,
            cache=cache,
            api_keys=api_keys,
            components=components,
            performance=performance,
            logging=logging,
            monitoring=monitoring,
            security=security,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "environment": self.environment.value,
            "database": asdict(self.database),
            "cache": asdict(self.cache),
            "api_keys": {**asdict(self.api_keys), "directeye_api_key": "***", "openai_api_key": "***"},  # Redact secrets
            "components": asdict(self.components),
            "performance": asdict(self.performance),
            "logging": asdict(self.logging),
            "monitoring": asdict(self.monitoring),
            "security": asdict(self.security),
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert configuration to JSON string"""
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, path: str):
        """Save configuration to JSON file"""
        with open(path, 'w') as f:
            f.write(self.to_json())

    @classmethod
    def load(cls, path: str) -> "Config":
        """Load configuration from JSON file"""
        with open(path, 'r') as f:
            data = json.load(f)

        return cls(
            environment=Environment(data["environment"]),
            database=DatabaseConfig(**data["database"]),
            cache=CacheConfig(**data["cache"]),
            api_keys=APIKeysConfig(**data["api_keys"]),
            components=ComponentsConfig(**data["components"]),
            performance=PerformanceConfig(**data["performance"]),
            logging=LoggingConfig(**data["logging"]),
            monitoring=MonitoringConfig(**data["monitoring"]),
            security=SecurityConfig(**data["security"]),
        )


def create_example_env_file(path: str = ".env.example"):
    """Create an example .env file with all configuration options"""
    content = """# Enhanced AI Engine Configuration
# Copy this file to .env and fill in your values

# Environment
ENVIRONMENT=development  # development, staging, production, testing

# Database Configuration
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=lat5150_ai
POSTGRES_USER=lat5150
POSTGRES_PASSWORD=your_password_here
# POSTGRES_URL=postgresql://user:pass@host:port/db  # Alternative: full URL

SQLITE_PATH=./agent_events.db

CHROMA_PATH=./chroma_db
# CHROMA_HOST=localhost  # If using client/server mode
# CHROMA_PORT=8000

# Cache Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=
# REDIS_URL=redis://:password@host:port/db  # Alternative: full URL

ENABLE_RESPONSE_CACHE=true
CACHE_TTL_SECONDS=3600

# API Keys
DIRECTEYE_API_KEY=your_directeye_key_here
DIRECTEYE_API_URL=https://api.directeye.io

OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here

GITHUB_TOKEN=your_github_token_here
SLACK_WEBHOOK=your_slack_webhook_here
PAGERDUTY_KEY=your_pagerduty_key_here

# Component Flags (true/false)
ENABLE_SELF_IMPROVEMENT=true
ENABLE_DSMIL_INTEGRATION=true
ENABLE_RAM_CONTEXT=true
ENABLE_FORENSICS_KNOWLEDGE=true
ENABLE_MULTI_MODEL_EVAL=true
ENABLE_DECAYING_MEMORY=true
ENABLE_EVENT_DRIVEN=true
ENABLE_ENTITY_RESOLUTION=true
ENABLE_DYNAMIC_SCHEMAS=true
ENABLE_AGENTIC_RAG=true
ENABLE_HUMAN_IN_LOOP=true
ENABLE_MCP_SELECTOR=true
ENABLE_THREAT_INTEL=true
ENABLE_BLOCKCHAIN_TOOLS=true
ENABLE_OSINT_WORKFLOWS=true
ENABLE_ADVANCED_ANALYTICS=true

# Performance Tuning
MAX_CONTEXT_WINDOW=131072
OPTIMAL_CONTEXT_WINDOW=65536
MAX_WORKING_MEMORY_TOKENS=8192
RAM_CONTEXT_SIZE_MB=512
MAX_CONCURRENT_QUERIES=10
QUERY_TIMEOUT_SECONDS=120
RAG_TOP_K_RESULTS=10
MCP_OPTIMIZE_FOR=balanced  # quality, speed, cost, balanced

# Logging
LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FORMAT=json  # json, text
LOG_TO_CONSOLE=true
LOG_TO_FILE=true
LOG_FILE_PATH=./logs/ai_engine.log

# Monitoring
ENABLE_METRICS=true
METRICS_PORT=9090
HEALTH_CHECK_ENABLED=true
HEALTH_CHECK_PORT=8080

# Security
REQUIRE_AUTHENTICATION=false
ENABLE_RATE_LIMITING=true
RATE_LIMIT_PER_MINUTE=60
AUDIT_ALL_QUERIES=true
AUDIT_LOG_PATH=./logs/audit.log
"""

    with open(path, 'w') as f:
        f.write(content)

    print(f"✅ Created example environment file: {path}")
    print(f"   Copy to .env and fill in your values")


if __name__ == "__main__":
    # Demo: Create example .env file
    create_example_env_file()

    # Demo: Load config and show it
    config = Config.from_env()
    print("\n📋 Current Configuration:")
    print(config.to_json())

    # Save example config
    config.save("config.example.json")
    print("\n✅ Saved example configuration to config.example.json")
