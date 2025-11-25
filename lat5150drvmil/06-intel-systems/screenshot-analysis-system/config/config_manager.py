"""
Configuration Manager for Screenshot Analysis System
Handles loading, validation, and access to system configuration
"""

import os
import yaml
import secrets
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class SystemConfig:
    """System-level configuration"""
    name: str = "ScreenshotIntel"
    version: str = "1.0.0"
    environment: str = "production"
    log_level: str = "INFO"


@dataclass
class StorageConfig:
    """Storage configuration"""
    zfs_dataset: str = "/mnt/screenshots"
    screenshots_dir: str = ""
    chat_logs_dir: str = ""
    metadata_dir: str = ""
    vector_db_dir: str = ""
    sqlite_db: str = ""
    snapshots_enabled: bool = True
    snapshot_interval: str = "6h"
    retention_days: int = 90

    def __post_init__(self):
        """Expand paths"""
        if not self.screenshots_dir:
            self.screenshots_dir = f"{self.zfs_dataset}/screenshots"
        if not self.chat_logs_dir:
            self.chat_logs_dir = f"{self.zfs_dataset}/chat_logs"
        if not self.metadata_dir:
            self.metadata_dir = f"{self.zfs_dataset}/metadata"
        if not self.vector_db_dir:
            self.vector_db_dir = f"{self.zfs_dataset}/vector_db"
        if not self.sqlite_db:
            self.sqlite_db = f"{self.zfs_dataset}/metadata/screenshot_intel.db"


@dataclass
class TelegramConfig:
    """Telegram API configuration"""
    enabled: bool = False
    api_id: str = ""
    api_hash: str = ""
    phone_number: str = ""
    session_name: str = "screenshot_intel_session"
    monitored_chats: list = field(default_factory=list)
    sync_interval: int = 300
    max_messages_per_sync: int = 100


@dataclass
class SignalConfig:
    """Signal CLI configuration"""
    enabled: bool = False
    signal_cli_path: str = "/usr/local/bin/signal-cli"
    phone_number: str = ""
    sync_interval: int = 300
    receive_timeout: int = 5


@dataclass
class VectorDBConfig:
    """Vector database configuration"""
    type: str = "qdrant"
    host: str = "localhost"
    port: int = 6333
    collections: Dict[str, str] = field(default_factory=dict)
    hnsw_config: Dict[str, int] = field(default_factory=dict)
    search: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.collections:
            self.collections = {
                "screenshots": "screenshots_collection",
                "chat_messages": "chat_messages_collection"
            }
        if not self.hnsw_config:
            self.hnsw_config = {"m": 16, "ef_construct": 100}
        if not self.search:
            self.search = {"limit": 10, "score_threshold": 0.7}


@dataclass
class APIConfig:
    """API server configuration"""
    enabled: bool = True
    host: str = "0.0.0.0"
    port: int = 5051
    auth_enabled: bool = True
    api_key: str = ""
    rate_limit: int = 100
    cors_enabled: bool = True
    cors_origins: list = field(default_factory=lambda: ["http://localhost:*"])

    def __post_init__(self):
        """Generate API key if not set"""
        if not self.api_key:
            self.api_key = secrets.token_urlsafe(32)


class ConfigManager:
    """
    Central configuration manager for the Screenshot Analysis System
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager

        Args:
            config_path: Path to config.yaml file. If None, uses default location.
        """
        if config_path is None:
            # Default to config.yaml in the same directory as this file
            config_dir = Path(__file__).parent
            config_path = config_dir / "config.yaml"

        self.config_path = Path(config_path)
        self.raw_config: Dict[str, Any] = {}

        # Configuration objects
        self.system: SystemConfig = SystemConfig()
        self.storage: StorageConfig = StorageConfig()
        self.telegram: TelegramConfig = TelegramConfig()
        self.signal: SignalConfig = SignalConfig()
        self.vector_db: VectorDBConfig = VectorDBConfig()
        self.api: APIConfig = APIConfig()

        # Load configuration
        self.load()

    def load(self):
        """Load configuration from YAML file"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        with open(self.config_path, 'r') as f:
            self.raw_config = yaml.safe_load(f)

        # Expand environment variables in config
        self.raw_config = self._expand_env_vars(self.raw_config)

        # Parse sections
        self._parse_system()
        self._parse_storage()
        self._parse_telegram()
        self._parse_signal()
        self._parse_vector_db()
        self._parse_api()

    def _expand_env_vars(self, config: Any) -> Any:
        """Recursively expand environment variables in config values"""
        if isinstance(config, dict):
            return {k: self._expand_env_vars(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [self._expand_env_vars(item) for item in config]
        elif isinstance(config, str):
            # Expand ${VAR} style variables
            return os.path.expandvars(config)
        return config

    def _parse_system(self):
        """Parse system configuration"""
        system_config = self.raw_config.get('system', {})
        self.system = SystemConfig(**system_config)

    def _parse_storage(self):
        """Parse storage configuration"""
        storage_config = self.raw_config.get('storage', {})
        self.storage = StorageConfig(**storage_config)

    def _parse_telegram(self):
        """Parse Telegram configuration"""
        telegram_config = self.raw_config.get('telegram', {})
        self.telegram = TelegramConfig(**telegram_config)

    def _parse_signal(self):
        """Parse Signal configuration"""
        signal_config = self.raw_config.get('signal', {})
        self.signal = SignalConfig(**signal_config)

    def _parse_vector_db(self):
        """Parse vector database configuration"""
        vdb_config = self.raw_config.get('vector_db', {})
        self.vector_db = VectorDBConfig(**vdb_config)

    def _parse_api(self):
        """Parse API configuration"""
        api_config = self.raw_config.get('api', {})
        self.api = APIConfig(**api_config)

    def save(self):
        """Save current configuration back to YAML file"""
        # Update raw config with current values
        self.raw_config['system'] = self.system.__dict__
        self.raw_config['storage'] = {k: v for k, v in self.storage.__dict__.items()
                                       if not k.startswith('_')}
        self.raw_config['telegram'] = {k: v for k, v in self.telegram.__dict__.items()
                                        if not k.startswith('_')}
        self.raw_config['signal'] = {k: v for k, v in self.signal.__dict__.items()
                                      if not k.startswith('_')}
        self.raw_config['vector_db'] = {k: v for k, v in self.vector_db.__dict__.items()
                                         if not k.startswith('_')}
        self.raw_config['api'] = {k: v for k, v in self.api.__dict__.items()
                                   if not k.startswith('_')}

        with open(self.config_path, 'w') as f:
            yaml.dump(self.raw_config, f, default_flow_style=False)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value by dot-notation key

        Args:
            key: Configuration key in dot notation (e.g., 'storage.zfs_dataset')
            default: Default value if key not found

        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.raw_config

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default

        return value

    def set(self, key: str, value: Any):
        """
        Set a configuration value by dot-notation key

        Args:
            key: Configuration key in dot notation
            value: Value to set
        """
        keys = key.split('.')
        config = self.raw_config

        # Navigate to the parent of the target key
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        # Set the value
        config[keys[-1]] = value

    def validate(self) -> bool:
        """
        Validate configuration

        Returns:
            True if valid, False otherwise
        """
        errors = []

        # Check storage paths
        if not self.storage.zfs_dataset:
            errors.append("storage.zfs_dataset is required")

        # Check Telegram config if enabled
        if self.telegram.enabled:
            if not self.telegram.api_id:
                errors.append("telegram.api_id is required when Telegram is enabled")
            if not self.telegram.api_hash:
                errors.append("telegram.api_hash is required when Telegram is enabled")
            if not self.telegram.phone_number:
                errors.append("telegram.phone_number is required when Telegram is enabled")

        # Check Signal config if enabled
        if self.signal.enabled:
            if not self.signal.phone_number:
                errors.append("signal.phone_number is required when Signal is enabled")
            if not Path(self.signal.signal_cli_path).exists():
                errors.append(f"signal-cli not found at {self.signal.signal_cli_path}")

        if errors:
            for error in errors:
                print(f"❌ Configuration error: {error}")
            return False

        return True

    def ensure_directories(self):
        """Create required directories if they don't exist"""
        dirs_to_create = [
            self.storage.screenshots_dir,
            self.storage.chat_logs_dir,
            self.storage.metadata_dir,
            self.storage.vector_db_dir,
            Path(self.storage.sqlite_db).parent,
        ]

        # Add device-specific directories
        devices = self.raw_config.get('ingestion', {}).get('devices', {})
        for device_config in devices.values():
            if 'path' in device_config:
                dirs_to_create.append(device_config['path'])

        for directory in dirs_to_create:
            dir_path = Path(directory)
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"✓ Ensured directory: {dir_path}")

    def __repr__(self) -> str:
        """String representation of configuration"""
        return f"<ConfigManager: {self.config_path}>"


# Global configuration instance
_config: Optional[ConfigManager] = None


def get_config(config_path: Optional[str] = None) -> ConfigManager:
    """
    Get or create global configuration instance

    Args:
        config_path: Path to config file (only used on first call)

    Returns:
        ConfigManager instance
    """
    global _config
    if _config is None:
        _config = ConfigManager(config_path)
    return _config


def reload_config():
    """Reload configuration from disk"""
    global _config
    if _config is not None:
        _config.load()


if __name__ == "__main__":
    # Test configuration loading
    config = ConfigManager()
    print(f"System: {config.system.name} v{config.system.version}")
    print(f"Storage: {config.storage.zfs_dataset}")
    print(f"Telegram enabled: {config.telegram.enabled}")
    print(f"Signal enabled: {config.signal.enabled}")
    print(f"API Key: {config.api.api_key[:16]}...")
    print(f"Validation: {'✓ PASS' if config.validate() else '✗ FAIL'}")
