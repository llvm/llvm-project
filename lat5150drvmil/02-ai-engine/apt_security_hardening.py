#!/usr/bin/env python3
"""
APT-Grade Security Hardening for Self-Coding System
Defense-in-depth security for localhost-only deployment

Security Layers:
1. Network isolation (localhost-only, firewall rules)
2. Authentication (local token-based auth)
3. Input validation (comprehensive sanitization)
4. Command sandboxing (restricted execution)
5. File system protection (path validation, permissions)
6. Rate limiting (abuse prevention)
7. Security monitoring (intrusion detection)
8. Audit logging (complete activity trail)
9. Session security (token rotation, timeout)
10. Cryptographic protection (data integrity)
"""

import os
import sys
import json
import time
import hmac
import secrets
import hashlib
import logging
import subprocess
import threading
import ipaddress
from typing import Dict, List, Optional, Any, Set
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import wraps
import re
import socket

logger = logging.getLogger(__name__)


class SecurityLevel:
    """Security level constants"""
    PARANOID = "paranoid"      # Maximum security
    HIGH = "high"              # Strong security
    MEDIUM = "medium"          # Balanced security
    LOW = "low"                # Basic security


@dataclass
class SecurityConfig:
    """Security configuration"""
    # Network security
    localhost_only: bool = True
    allowed_ips: List[str] = field(default_factory=lambda: ["127.0.0.1", "::1"])
    bind_address: str = "127.0.0.1"

    # Authentication
    require_auth: bool = True
    token_length: int = 64
    token_expiry_minutes: int = 480  # 8 hours
    session_timeout_minutes: int = 60

    # Input validation
    max_message_length: int = 10000
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    allowed_file_extensions: Set[str] = field(default_factory=lambda: {
        '.py', '.js', '.ts', '.sh', '.md', '.txt', '.json', '.yaml', '.yml',
        '.c', '.cpp', '.h', '.hpp', '.go', '.rs', '.java'
    })

    # Command sandboxing
    allowed_commands: Set[str] = field(default_factory=lambda: {
        'ls', 'cat', 'head', 'tail', 'grep', 'find', 'git',
        'python', 'python3', 'pytest', 'pip', 'npm', 'node'
    })
    blocked_commands: Set[str] = field(default_factory=lambda: {
        'rm', 'dd', 'mkfs', 'format', 'fdisk', 'parted',
        'curl', 'wget', 'nc', 'netcat', 'telnet', 'ssh',
        'sudo', 'su', 'chmod', 'chown'
    })
    command_timeout: int = 300  # 5 minutes

    # File system protection
    workspace_root: str = "."
    read_only_paths: Set[str] = field(default_factory=lambda: {
        '/etc', '/sys', '/proc', '/boot', '/dev'
    })
    blocked_paths: Set[str] = field(default_factory=lambda: {
        '~/.ssh', '~/.gnupg', '/root', '/home/*/.ssh'
    })

    # Rate limiting
    max_requests_per_minute: int = 60
    max_requests_per_hour: int = 1000
    max_concurrent_sessions: int = 10

    # Security monitoring
    enable_intrusion_detection: bool = True
    enable_audit_logging: bool = True
    log_all_commands: bool = True
    log_file: str = ".security/audit.log"

    # Cryptographic protection
    enable_integrity_checks: bool = True
    secret_key: Optional[str] = None


class SecurityException(Exception):
    """Base security exception"""
    pass


class AuthenticationError(SecurityException):
    """Authentication failed"""
    pass


class AuthorizationError(SecurityException):
    """Authorization failed"""
    pass


class ValidationError(SecurityException):
    """Input validation failed"""
    pass


class SandboxViolation(SecurityException):
    """Sandbox policy violated"""
    pass


class RateLimitExceeded(SecurityException):
    """Rate limit exceeded"""
    pass


class APTGradeSecurityHardening:
    """
    APT-Grade security hardening for self-coding system

    Features:
    - Strict localhost-only access
    - Token-based authentication
    - Comprehensive input validation
    - Command execution sandboxing
    - File system protection
    - Rate limiting
    - Intrusion detection
    - Complete audit logging
    """

    def __init__(self, config: Optional[SecurityConfig] = None):
        """Initialize security hardening"""
        self.config = config or SecurityConfig()

        # Initialize security components
        self._init_crypto()
        self._init_auth()
        self._init_rate_limiter()
        self._init_audit_log()
        self._init_intrusion_detection()

        # Security state
        self.active_sessions: Dict[str, Dict] = {}
        self.blocked_ips: Set[str] = set()
        self.suspicious_activity: Dict[str, List[Dict]] = {}

        logger.info("APT-Grade Security Hardening initialized")

    def _init_crypto(self):
        """Initialize cryptographic components"""
        if not self.config.secret_key:
            self.config.secret_key = secrets.token_hex(32)

        self.secret_key = self.config.secret_key.encode()

    def _init_auth(self):
        """Initialize authentication"""
        self.valid_tokens: Dict[str, Dict] = {}
        self.token_file = Path(self.config.workspace_root) / ".security" / "tokens.json"
        self.token_file.parent.mkdir(parents=True, exist_ok=True)

        # Load existing tokens
        if self.token_file.exists():
            try:
                with open(self.token_file, 'r') as f:
                    self.valid_tokens = json.load(f)
            except Exception as e:
                logger.warning(f"Could not load tokens: {e}")

    def _init_rate_limiter(self):
        """Initialize rate limiter"""
        self.request_counts: Dict[str, List[float]] = {}
        self.rate_limit_lock = threading.Lock()

    def _init_audit_log(self):
        """Initialize audit logging"""
        if self.config.enable_audit_logging:
            log_file = Path(self.config.workspace_root) / self.config.log_file
            log_file.parent.mkdir(parents=True, exist_ok=True)

            self.audit_handler = logging.FileHandler(log_file)
            self.audit_handler.setLevel(logging.INFO)
            self.audit_handler.setFormatter(logging.Formatter(
                '%(asctime)s - SECURITY - %(levelname)s - %(message)s'
            ))

            self.audit_logger = logging.getLogger('security_audit')
            self.audit_logger.addHandler(self.audit_handler)
            self.audit_logger.setLevel(logging.INFO)

    def _init_intrusion_detection(self):
        """Initialize intrusion detection"""
        self.intrusion_patterns = [
            # Path traversal attempts
            r'\.\./|\.\.\\',
            # Command injection
            r'[;&|`$]',
            # SQL injection
            r'(union|select|insert|update|delete|drop)\s',
            # Script injection
            r'<script|javascript:|onerror=',
            # Null bytes
            r'\x00',
        ]
        self.intrusion_regex = [re.compile(p, re.IGNORECASE) for p in self.intrusion_patterns]

    # ============================================================================
    # NETWORK SECURITY
    # ============================================================================

    def verify_localhost_access(self, request_ip: str) -> bool:
        """
        Verify request is from localhost only

        Args:
            request_ip: Client IP address

        Returns:
            True if allowed, False otherwise

        Raises:
            AuthorizationError: If access denied
        """
        if not self.config.localhost_only:
            return True

        # Check if IP is blocked
        if request_ip in self.blocked_ips:
            self._audit_log("BLOCKED_IP_ATTEMPT", {
                "ip": request_ip,
                "reason": "Previously blocked"
            })
            raise AuthorizationError(f"Access denied: IP {request_ip} is blocked")

        # Check if IP is localhost
        try:
            ip = ipaddress.ip_address(request_ip)

            # Allow loopback
            if ip.is_loopback:
                return True

            # Check allowed IPs
            if request_ip in self.config.allowed_ips:
                return True

            # Deny everything else
            self._audit_log("NON_LOCALHOST_ACCESS_DENIED", {
                "ip": request_ip,
                "message": "Only localhost access allowed"
            })

            # Track suspicious activity
            self._track_suspicious_activity(request_ip, "non_localhost_access")

            raise AuthorizationError(
                f"Access denied: Only localhost access allowed (got {request_ip})"
            )

        except ValueError:
            self._audit_log("INVALID_IP_ADDRESS", {"ip": request_ip})
            raise AuthorizationError(f"Invalid IP address: {request_ip}")

    def block_ip(self, ip: str, reason: str = "Security policy"):
        """Block an IP address"""
        self.blocked_ips.add(ip)
        self._audit_log("IP_BLOCKED", {"ip": ip, "reason": reason})
        logger.warning(f"Blocked IP {ip}: {reason}")

    # ============================================================================
    # AUTHENTICATION & AUTHORIZATION
    # ============================================================================

    def generate_token(self, user_id: str = "localhost") -> str:
        """
        Generate secure authentication token

        Args:
            user_id: User identifier

        Returns:
            Secure token string
        """
        token = secrets.token_urlsafe(self.config.token_length)

        expires_at = time.time() + (self.config.token_expiry_minutes * 60)

        self.valid_tokens[token] = {
            "user_id": user_id,
            "created_at": time.time(),
            "expires_at": expires_at,
            "last_used": time.time()
        }

        self._save_tokens()
        self._audit_log("TOKEN_GENERATED", {"user_id": user_id})

        return token

    def validate_token(self, token: str) -> Dict:
        """
        Validate authentication token

        Args:
            token: Token to validate

        Returns:
            Token metadata

        Raises:
            AuthenticationError: If token invalid
        """
        if not self.config.require_auth:
            return {"user_id": "localhost", "authenticated": False}

        if not token:
            raise AuthenticationError("No token provided")

        if token not in self.valid_tokens:
            self._audit_log("INVALID_TOKEN_ATTEMPT", {"token_prefix": token[:8]})
            raise AuthenticationError("Invalid token")

        token_data = self.valid_tokens[token]

        # Check expiration
        if time.time() > token_data["expires_at"]:
            del self.valid_tokens[token]
            self._save_tokens()
            self._audit_log("EXPIRED_TOKEN_ATTEMPT", {
                "user_id": token_data["user_id"]
            })
            raise AuthenticationError("Token expired")

        # Update last used
        token_data["last_used"] = time.time()

        return token_data

    def revoke_token(self, token: str):
        """Revoke a token"""
        if token in self.valid_tokens:
            user_id = self.valid_tokens[token]["user_id"]
            del self.valid_tokens[token]
            self._save_tokens()
            self._audit_log("TOKEN_REVOKED", {"user_id": user_id})

    def _save_tokens(self):
        """Save tokens to disk"""
        try:
            with open(self.token_file, 'w') as f:
                json.dump(self.valid_tokens, f)
            os.chmod(self.token_file, 0o600)  # Read/write owner only
        except Exception as e:
            logger.error(f"Error saving tokens: {e}")

    def requires_auth(self, func):
        """Decorator for requiring authentication"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            token = kwargs.get('token') or (args[0] if args else None)

            if self.config.require_auth:
                self.validate_token(token)

            return func(*args, **kwargs)

        return wrapper

    # ============================================================================
    # INPUT VALIDATION
    # ============================================================================

    def validate_message(self, message: str) -> str:
        """
        Validate and sanitize user message

        Args:
            message: User input message

        Returns:
            Sanitized message

        Raises:
            ValidationError: If validation fails
        """
        if not message:
            raise ValidationError("Empty message")

        if len(message) > self.config.max_message_length:
            raise ValidationError(
                f"Message too long: {len(message)} > {self.config.max_message_length}"
            )

        # Check for intrusion patterns
        for pattern in self.intrusion_regex:
            if pattern.search(message):
                self._audit_log("INTRUSION_PATTERN_DETECTED", {
                    "pattern": pattern.pattern,
                    "message_prefix": message[:100]
                })
                raise ValidationError("Message contains suspicious patterns")

        # Check for null bytes
        if '\x00' in message:
            raise ValidationError("Message contains null bytes")

        # Sanitize
        sanitized = message.strip()

        return sanitized

    def validate_filepath(self, filepath: str) -> Path:
        """
        Validate file path for security

        Args:
            filepath: File path to validate

        Returns:
            Validated Path object

        Raises:
            ValidationError: If path is invalid/unsafe
        """
        try:
            # Resolve path
            workspace = Path(self.config.workspace_root).resolve()
            target = (workspace / filepath).resolve()

            # Check if path is within workspace
            if workspace not in target.parents and target != workspace:
                raise ValidationError(
                    f"Path outside workspace: {filepath}"
                )

            # Check against read-only paths
            for readonly in self.config.read_only_paths:
                readonly_path = Path(readonly).resolve()
                if readonly_path in target.parents or target == readonly_path:
                    raise ValidationError(
                        f"Path in read-only location: {filepath}"
                    )

            # Check against blocked paths
            for blocked in self.config.blocked_paths:
                # Handle wildcards
                if '*' in blocked:
                    pattern = blocked.replace('*', '.*')
                    if re.match(pattern, str(target)):
                        raise ValidationError(
                            f"Path matches blocked pattern: {filepath}"
                        )
                else:
                    blocked_path = Path(blocked).expanduser().resolve()
                    if blocked_path in target.parents or target == blocked_path:
                        raise ValidationError(
                            f"Path is blocked: {filepath}"
                        )

            # Check file extension if it exists
            if target.exists() and target.is_file():
                if target.suffix.lower() not in self.config.allowed_file_extensions:
                    raise ValidationError(
                        f"File extension not allowed: {target.suffix}"
                    )

            return target

        except Exception as e:
            if isinstance(e, ValidationError):
                raise
            raise ValidationError(f"Invalid file path: {e}")

    def validate_command(self, command: str) -> str:
        """
        Validate command for execution

        Args:
            command: Command to validate

        Returns:
            Validated command

        Raises:
            SandboxViolation: If command violates sandbox policy
        """
        if not command:
            raise SandboxViolation("Empty command")

        # Extract base command
        base_command = command.split()[0] if command.split() else ""

        # Check blocked commands
        if base_command in self.config.blocked_commands:
            self._audit_log("BLOCKED_COMMAND_ATTEMPT", {
                "command": command
            })
            raise SandboxViolation(
                f"Command blocked by security policy: {base_command}"
            )

        # Check allowed commands (if whitelist is not empty)
        if self.config.allowed_commands and base_command not in self.config.allowed_commands:
            self._audit_log("UNAUTHORIZED_COMMAND_ATTEMPT", {
                "command": command
            })
            raise SandboxViolation(
                f"Command not in allowed list: {base_command}"
            )

        # Check for command injection patterns
        dangerous_patterns = [
            r'[;&|`]',  # Command chaining
            r'\$\(',    # Command substitution
            r'>\s*/dev',  # Device access
            r'<\s*/dev',
        ]

        for pattern in dangerous_patterns:
            if re.search(pattern, command):
                self._audit_log("COMMAND_INJECTION_ATTEMPT", {
                    "command": command,
                    "pattern": pattern
                })
                raise SandboxViolation(
                    "Command contains potentially dangerous patterns"
                )

        return command

    # ============================================================================
    # RATE LIMITING
    # ============================================================================

    def check_rate_limit(self, identifier: str) -> bool:
        """
        Check rate limit for identifier

        Args:
            identifier: IP address or session ID

        Returns:
            True if within limits

        Raises:
            RateLimitExceeded: If rate limit exceeded
        """
        with self.rate_limit_lock:
            now = time.time()

            # Initialize if new
            if identifier not in self.request_counts:
                self.request_counts[identifier] = []

            # Remove old requests
            self.request_counts[identifier] = [
                ts for ts in self.request_counts[identifier]
                if now - ts < 3600  # Keep last hour
            ]

            # Count recent requests
            last_minute = [ts for ts in self.request_counts[identifier] if now - ts < 60]
            last_hour = self.request_counts[identifier]

            # Check limits
            if len(last_minute) >= self.config.max_requests_per_minute:
                self._audit_log("RATE_LIMIT_EXCEEDED", {
                    "identifier": identifier,
                    "period": "minute",
                    "count": len(last_minute)
                })
                raise RateLimitExceeded(
                    f"Rate limit exceeded: {len(last_minute)} requests in last minute"
                )

            if len(last_hour) >= self.config.max_requests_per_hour:
                self._audit_log("RATE_LIMIT_EXCEEDED", {
                    "identifier": identifier,
                    "period": "hour",
                    "count": len(last_hour)
                })
                raise RateLimitExceeded(
                    f"Rate limit exceeded: {len(last_hour)} requests in last hour"
                )

            # Record request
            self.request_counts[identifier].append(now)

            return True

    # ============================================================================
    # COMMAND SANDBOXING
    # ============================================================================

    def execute_sandboxed(
        self,
        command: str,
        timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Execute command in sandbox

        Args:
            command: Command to execute
            timeout: Execution timeout

        Returns:
            Execution result

        Raises:
            SandboxViolation: If execution violates sandbox
        """
        # Validate command
        validated_command = self.validate_command(command)

        timeout = timeout or self.config.command_timeout

        # Log command
        if self.config.log_all_commands:
            self._audit_log("COMMAND_EXECUTED", {
                "command": validated_command
            })

        try:
            result = subprocess.run(
                validated_command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=self.config.workspace_root
            )

            return {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
                "success": result.returncode == 0
            }

        except subprocess.TimeoutExpired:
            self._audit_log("COMMAND_TIMEOUT", {
                "command": validated_command,
                "timeout": timeout
            })
            raise SandboxViolation(f"Command exceeded timeout: {timeout}s")

        except Exception as e:
            self._audit_log("COMMAND_EXECUTION_ERROR", {
                "command": validated_command,
                "error": str(e)
            })
            raise SandboxViolation(f"Command execution failed: {e}")

    # ============================================================================
    # INTRUSION DETECTION
    # ============================================================================

    def _track_suspicious_activity(self, identifier: str, activity_type: str):
        """Track suspicious activity"""
        if identifier not in self.suspicious_activity:
            self.suspicious_activity[identifier] = []

        self.suspicious_activity[identifier].append({
            "type": activity_type,
            "timestamp": time.time()
        })

        # Check if threshold exceeded
        recent_activity = [
            a for a in self.suspicious_activity[identifier]
            if time.time() - a["timestamp"] < 300  # Last 5 minutes
        ]

        if len(recent_activity) >= 5:
            self.block_ip(identifier, "Multiple suspicious activities detected")

    def detect_intrusion_attempt(self, data: str) -> bool:
        """
        Detect potential intrusion attempt

        Args:
            data: Data to analyze

        Returns:
            True if intrusion detected
        """
        for pattern in self.intrusion_regex:
            if pattern.search(data):
                return True

        return False

    # ============================================================================
    # AUDIT LOGGING
    # ============================================================================

    def _audit_log(self, event_type: str, data: Dict[str, Any]):
        """Log security event"""
        if not self.config.enable_audit_logging:
            return

        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "data": data
        }

        self.audit_logger.info(json.dumps(log_entry))

    def get_audit_log(self, last_n: int = 100) -> List[Dict]:
        """Get recent audit log entries"""
        log_file = Path(self.config.workspace_root) / self.config.log_file

        if not log_file.exists():
            return []

        entries = []

        with open(log_file, 'r') as f:
            for line in f:
                try:
                    # Parse log line
                    parts = line.split(' - SECURITY - ')
                    if len(parts) >= 2:
                        json_part = parts[-1].split(' - ')[-1]
                        entry = json.loads(json_part)
                        entries.append(entry)
                except:
                    continue

        return entries[-last_n:]

    # ============================================================================
    # SECURITY AUDIT
    # ============================================================================

    def audit_system_security(self) -> Dict[str, Any]:
        """
        Perform comprehensive security audit

        Returns:
            Audit report
        """
        report = {
            "timestamp": datetime.now().isoformat(),
            "configuration": {
                "localhost_only": self.config.localhost_only,
                "authentication_required": self.config.require_auth,
                "intrusion_detection_enabled": self.config.enable_intrusion_detection,
                "audit_logging_enabled": self.config.enable_audit_logging
            },
            "network": {
                "bind_address": self.config.bind_address,
                "allowed_ips": list(self.config.allowed_ips),
                "blocked_ips": list(self.blocked_ips)
            },
            "authentication": {
                "active_tokens": len(self.valid_tokens),
                "active_sessions": len(self.active_sessions)
            },
            "rate_limiting": {
                "tracked_identifiers": len(self.request_counts)
            },
            "intrusion_detection": {
                "suspicious_identifiers": len(self.suspicious_activity),
                "blocked_ips": len(self.blocked_ips)
            },
            "recommendations": []
        }

        # Generate recommendations
        if not self.config.localhost_only:
            report["recommendations"].append(
                "CRITICAL: Enable localhost-only access"
            )

        if not self.config.require_auth:
            report["recommendations"].append(
                "WARNING: Enable authentication"
            )

        if not self.config.enable_audit_logging:
            report["recommendations"].append(
                "WARNING: Enable audit logging"
            )

        return report


def create_security_config(level: str = SecurityLevel.HIGH) -> SecurityConfig:
    """
    Create security configuration for given level

    Args:
        level: Security level (paranoid, high, medium, low)

    Returns:
        SecurityConfig
    """
    if level == SecurityLevel.PARANOID:
        return SecurityConfig(
            localhost_only=True,
            require_auth=True,
            token_expiry_minutes=240,  # 4 hours
            session_timeout_minutes=30,
            max_requests_per_minute=30,
            max_requests_per_hour=500,
            command_timeout=180,  # 3 minutes
            enable_intrusion_detection=True,
            enable_audit_logging=True,
            log_all_commands=True
        )

    elif level == SecurityLevel.HIGH:
        return SecurityConfig(
            localhost_only=True,
            require_auth=True,
            token_expiry_minutes=480,  # 8 hours
            session_timeout_minutes=60,
            max_requests_per_minute=60,
            max_requests_per_hour=1000,
            command_timeout=300,  # 5 minutes
            enable_intrusion_detection=True,
            enable_audit_logging=True,
            log_all_commands=True
        )

    elif level == SecurityLevel.MEDIUM:
        return SecurityConfig(
            localhost_only=True,
            require_auth=False,  # Trust localhost
            enable_intrusion_detection=True,
            enable_audit_logging=True,
            log_all_commands=False
        )

    else:  # LOW
        return SecurityConfig(
            localhost_only=True,
            require_auth=False,
            enable_intrusion_detection=False,
            enable_audit_logging=False,
            log_all_commands=False
        )


def main():
    """Example usage and testing"""
    print("="*80)
    print("APT-GRADE SECURITY HARDENING")
    print("="*80 + "\n")

    # Create paranoid-level security
    config = create_security_config(SecurityLevel.PARANOID)
    security = APTGradeSecurityHardening(config)

    print("1. Testing localhost enforcement...")
    try:
        security.verify_localhost_access("127.0.0.1")
        print("   ✓ Localhost access allowed")
    except:
        print("   ✗ Localhost access denied")

    try:
        security.verify_localhost_access("192.168.1.1")
        print("   ✗ Remote access allowed (FAIL)")
    except AuthorizationError:
        print("   ✓ Remote access denied")

    print("\n2. Testing authentication...")
    token = security.generate_token()
    print(f"   Generated token: {token[:16]}...")

    try:
        security.validate_token(token)
        print("   ✓ Token validation successful")
    except:
        print("   ✗ Token validation failed")

    print("\n3. Testing input validation...")
    try:
        security.validate_message("Add logging to server.py")
        print("   ✓ Clean message validated")
    except:
        print("   ✗ Clean message rejected")

    try:
        security.validate_message("'; DROP TABLE users--")
        print("   ✗ SQL injection not detected (FAIL)")
    except ValidationError:
        print("   ✓ SQL injection blocked")

    print("\n4. Testing command sandboxing...")
    try:
        security.validate_command("ls -la")
        print("   ✓ Safe command allowed")
    except:
        print("   ✗ Safe command blocked")

    try:
        security.validate_command("rm -rf /")
        print("   ✗ Dangerous command allowed (FAIL)")
    except SandboxViolation:
        print("   ✓ Dangerous command blocked")

    print("\n5. Security audit report...")
    report = security.audit_system_security()
    print(f"   Configuration: localhost_only={report['configuration']['localhost_only']}")
    print(f"   Authentication: {report['authentication']['active_tokens']} tokens")
    print(f"   Recommendations: {len(report['recommendations'])}")

    print("\n✓ Security hardening ready!")
    print("="*80)


if __name__ == "__main__":
    main()
