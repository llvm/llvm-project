#!/usr/bin/env python3
"""
DSMIL AI MCP Server Security Module
Implements authentication, rate limiting, input validation, and audit logging

Classification: UNCLASSIFIED // FOR OFFICIAL USE ONLY
Version: 1.0.0
"""

import hashlib
import hmac
import json
import os
import time
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
from collections import defaultdict

# Security configuration file location
SECURITY_CONFIG_PATH = Path.home() / ".dsmil" / "mcp_security.json"
AUDIT_LOG_PATH = Path.home() / ".dsmil" / "mcp_audit.log"


class MCPSecurityManager:
    """
    Security manager for MCP server

    Features:
    - Token-based authentication
    - Rate limiting (per-client, per-tool)
    - Input validation and sanitization
    - Audit logging
    - Path traversal prevention
    """

    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or SECURITY_CONFIG_PATH
        self.audit_log_path = AUDIT_LOG_PATH

        # Ensure config directory exists
        self.config_path.parent.mkdir(parents=True, exist_ok=True)

        # Load or create security configuration
        self.config = self._load_config()

        # Rate limiting state
        self.rate_limits: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "count": 0,
            "reset_time": time.time() + 60
        })

        # Setup audit logging
        self._setup_audit_logging()

    def _load_config(self) -> Dict[str, Any]:
        """Load security configuration"""
        default_config = {
            "authentication": {
                "enabled": True,
                "token_hash": None,  # SHA-256 hash of auth token
                "require_token": True
            },
            "rate_limiting": {
                "enabled": True,
                "requests_per_minute": 60,
                "burst_requests": 10
            },
            "input_validation": {
                "max_query_length": 10000,
                "max_filepath_length": 4096,
                "allowed_file_extensions": [
                    ".pdf", ".txt", ".md", ".log",
                    ".c", ".h", ".cpp", ".hpp",
                    ".py", ".sh", ".json", ".yaml"
                ],
                "blocked_paths": [
                    "/etc/shadow", "/etc/passwd",
                    "/root", "/boot", "/sys", "/proc"
                ]
            },
            "audit": {
                "enabled": True,
                "log_all_requests": True,
                "log_failed_auth": True,
                "log_rate_limit": True
            },
            "sandboxing": {
                "restrict_file_access": True,
                "allowed_directories": [
                    str(Path.home()),
                    "/tmp",
                    "/var/tmp"
                ],
                "deny_dotfiles": True
            }
        }

        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    loaded_config = json.load(f)
                    # Merge with defaults
                    default_config.update(loaded_config)
            except Exception as e:
                print(f"Warning: Failed to load security config: {e}")
        else:
            # Create default config
            self._save_config(default_config)

        return default_config

    def _save_config(self, config: Dict[str, Any]):
        """Save security configuration"""
        try:
            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=2)
            os.chmod(self.config_path, 0o600)  # Read/write for owner only
        except Exception as e:
            print(f"Warning: Failed to save security config: {e}")

    def _setup_audit_logging(self):
        """Setup audit logging"""
        self.audit_log_path.parent.mkdir(parents=True, exist_ok=True)

        logging.basicConfig(
            filename=str(self.audit_log_path),
            level=logging.INFO,
            format='%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        self.audit_logger = logging.getLogger('dsmil_mcp_audit')

        # Also log to dedicated audit file
        audit_handler = logging.FileHandler(self.audit_log_path)
        audit_handler.setFormatter(logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        self.audit_logger.addHandler(audit_handler)

    def generate_token(self) -> str:
        """Generate a new authentication token"""
        token = os.urandom(32).hex()
        token_hash = hashlib.sha256(token.encode()).hexdigest()

        self.config["authentication"]["token_hash"] = token_hash
        self._save_config(self.config)

        self.audit_logger.warning(f"New auth token generated (hash: {token_hash[:16]}...)")

        return token

    def verify_token(self, token: str) -> bool:
        """Verify authentication token"""
        if not self.config["authentication"]["enabled"]:
            return True

        if not self.config["authentication"]["require_token"]:
            return True

        stored_hash = self.config["authentication"].get("token_hash")
        if not stored_hash:
            self.audit_logger.error("No auth token configured")
            return False

        token_hash = hashlib.sha256(token.encode()).hexdigest()

        # Constant-time comparison to prevent timing attacks
        is_valid = hmac.compare_digest(token_hash, stored_hash)

        if not is_valid:
            self.audit_logger.warning(f"Failed authentication attempt (hash: {token_hash[:16]}...)")

        return is_valid

    def check_rate_limit(self, client_id: str, tool_name: str) -> bool:
        """Check if request is within rate limits"""
        if not self.config["rate_limiting"]["enabled"]:
            return True

        key = f"{client_id}:{tool_name}"
        now = time.time()

        # Reset if time window expired
        if now >= self.rate_limits[key]["reset_time"]:
            self.rate_limits[key] = {
                "count": 0,
                "reset_time": now + 60
            }

        # Check limit
        limit = self.config["rate_limiting"]["requests_per_minute"]
        count = self.rate_limits[key]["count"]

        if count >= limit:
            self.audit_logger.warning(
                f"Rate limit exceeded: client={client_id}, tool={tool_name}, count={count}"
            )
            return False

        # Increment counter
        self.rate_limits[key]["count"] += 1
        return True

    def validate_query(self, query: str) -> tuple[bool, Optional[str]]:
        """Validate AI query input"""
        max_length = self.config["input_validation"]["max_query_length"]

        if len(query) > max_length:
            return False, f"Query too long (max {max_length} chars)"

        # Check for injection attempts
        suspicious_patterns = [
            "'; DROP TABLE",
            "' OR '1'='1",
            "<script>",
            "javascript:",
            "../../",
            "/etc/passwd",
            "/etc/shadow"
        ]

        query_lower = query.lower()
        for pattern in suspicious_patterns:
            if pattern.lower() in query_lower:
                self.audit_logger.warning(f"Suspicious pattern detected in query: {pattern}")
                return False, f"Query contains suspicious pattern"

        return True, None

    def validate_filepath(self, filepath: str) -> tuple[bool, Optional[str]]:
        """
        Validate file path for security

        Checks:
        - Path length
        - Path traversal attempts
        - Blocked paths
        - File extension
        - Sandboxing restrictions
        """
        max_length = self.config["input_validation"]["max_filepath_length"]

        if len(filepath) > max_length:
            return False, f"Path too long (max {max_length} chars)"

        try:
            # Resolve to absolute path
            path = Path(filepath).resolve()

            # Check for blocked paths
            blocked = self.config["input_validation"]["blocked_paths"]
            for blocked_path in blocked:
                if str(path).startswith(blocked_path):
                    self.audit_logger.warning(f"Blocked path access attempt: {path}")
                    return False, f"Access to {blocked_path} is blocked"

            # Check sandboxing
            if self.config["sandboxing"]["restrict_file_access"]:
                allowed_dirs = self.config["sandboxing"]["allowed_directories"]
                is_allowed = any(str(path).startswith(allowed_dir) for allowed_dir in allowed_dirs)

                if not is_allowed:
                    self.audit_logger.warning(f"Sandboxing violation: {path}")
                    return False, f"Path must be within: {', '.join(allowed_dirs)}"

            # Check for dotfiles
            if self.config["sandboxing"]["deny_dotfiles"]:
                if any(part.startswith('.') and part not in ['.', '..'] for part in path.parts):
                    return False, "Access to dotfiles/hidden files is blocked"

            # Check file extension (if it exists)
            if path.exists() and path.is_file():
                extension = path.suffix.lower()
                allowed_extensions = self.config["input_validation"]["allowed_file_extensions"]

                if extension and extension not in allowed_extensions:
                    return False, f"File type {extension} not allowed"

            return True, None

        except Exception as e:
            self.audit_logger.error(f"Path validation error: {e}")
            return False, f"Invalid path: {str(e)}"

    def audit_request(self, tool_name: str, arguments: Dict[str, Any],
                     client_id: str, success: bool, error: Optional[str] = None):
        """Log audit entry for request"""
        if not self.config["audit"]["enabled"]:
            return

        # Sanitize arguments (remove sensitive data)
        safe_args = self._sanitize_audit_data(arguments)

        log_data = {
            "timestamp": datetime.now().isoformat(),
            "client_id": client_id,
            "tool": tool_name,
            "arguments": safe_args,
            "success": success,
            "error": error
        }

        if success:
            self.audit_logger.info(json.dumps(log_data))
        else:
            self.audit_logger.warning(json.dumps(log_data))

    def _sanitize_audit_data(self, data: Any) -> Any:
        """Remove sensitive information from audit logs"""
        if isinstance(data, dict):
            sanitized = {}
            for key, value in data.items():
                # Redact sensitive keys
                if key.lower() in ['password', 'token', 'secret', 'key', 'auth']:
                    sanitized[key] = "[REDACTED]"
                else:
                    sanitized[key] = self._sanitize_audit_data(value)
            return sanitized
        elif isinstance(data, list):
            return [self._sanitize_audit_data(item) for item in data]
        elif isinstance(data, str) and len(data) > 1000:
            return data[:1000] + "... [TRUNCATED]"
        else:
            return data

    def get_security_status(self) -> Dict[str, Any]:
        """Get current security configuration status"""
        return {
            "authentication": {
                "enabled": self.config["authentication"]["enabled"],
                "token_configured": self.config["authentication"]["token_hash"] is not None
            },
            "rate_limiting": {
                "enabled": self.config["rate_limiting"]["enabled"],
                "limit": self.config["rate_limiting"]["requests_per_minute"]
            },
            "audit": {
                "enabled": self.config["audit"]["enabled"],
                "log_file": str(self.audit_log_path)
            },
            "sandboxing": {
                "enabled": self.config["sandboxing"]["restrict_file_access"],
                "allowed_directories": self.config["sandboxing"]["allowed_directories"]
            }
        }


# Singleton instance
_security_manager: Optional[MCPSecurityManager] = None


def get_security_manager() -> MCPSecurityManager:
    """Get global security manager instance"""
    global _security_manager
    if _security_manager is None:
        _security_manager = MCPSecurityManager()
    return _security_manager
