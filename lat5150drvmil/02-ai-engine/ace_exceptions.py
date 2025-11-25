#!/usr/bin/env python3
"""
ACE-FCA Exception Hierarchy Module
-----------------------------------
Provides custom exception classes for ACE-FCA system to improve error handling,
debugging, and error recovery.

Addresses: Generic exception handling, unclear error sources, debugging difficulties
"""

from typing import Optional, Dict, Any
from enum import Enum


class ErrorSeverity(Enum):
    """Severity level of errors"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Category of errors"""
    CONFIGURATION = "configuration"
    VALIDATION = "validation"
    EXECUTION = "execution"
    RESOURCE = "resource"
    INTEGRATION = "integration"
    TIMEOUT = "timeout"


# =============================================================================
# Base Exception
# =============================================================================

class ACEError(Exception):
    """
    Base exception for all ACE-FCA errors.

    Provides structured error information including severity, category,
    context, and recovery hints.
    """

    def __init__(
        self,
        message: str,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        category: ErrorCategory = ErrorCategory.EXECUTION,
        context: Optional[Dict[str, Any]] = None,
        recovery_hint: Optional[str] = None,
        cause: Optional[Exception] = None
    ):
        super().__init__(message)
        self.message = message
        self.severity = severity
        self.category = category
        self.context = context or {}
        self.recovery_hint = recovery_hint
        self.cause = cause

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging/serialization"""
        return {
            'error_type': self.__class__.__name__,
            'message': self.message,
            'severity': self.severity.value,
            'category': self.category.value,
            'context': self.context,
            'recovery_hint': self.recovery_hint,
            'cause': str(self.cause) if self.cause else None
        }

    def __str__(self) -> str:
        parts = [f"{self.__class__.__name__}: {self.message}"]

        if self.context:
            parts.append(f"Context: {self.context}")

        if self.recovery_hint:
            parts.append(f"Recovery: {self.recovery_hint}")

        if self.cause:
            parts.append(f"Caused by: {self.cause}")

        return "\n".join(parts)


# =============================================================================
# Configuration Errors
# =============================================================================

class ConfigurationError(ACEError):
    """Base class for configuration-related errors"""

    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('category', ErrorCategory.CONFIGURATION)
        super().__init__(message, **kwargs)


class InvalidConfigurationError(ConfigurationError):
    """Configuration is invalid or malformed"""

    def __init__(self, config_key: str, reason: str, **kwargs):
        message = f"Invalid configuration for '{config_key}': {reason}"
        kwargs.setdefault('context', {})
        kwargs['context']['config_key'] = config_key
        super().__init__(message, **kwargs)


class MissingConfigurationError(ConfigurationError):
    """Required configuration is missing"""

    def __init__(self, config_key: str, **kwargs):
        message = f"Missing required configuration: '{config_key}'"
        kwargs.setdefault('context', {})
        kwargs['context']['config_key'] = config_key
        kwargs.setdefault('recovery_hint', f"Add '{config_key}' to configuration")
        super().__init__(message, **kwargs)


# =============================================================================
# Validation Errors
# =============================================================================

class ValidationError(ACEError):
    """Base class for validation errors"""

    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('category', ErrorCategory.VALIDATION)
        super().__init__(message, **kwargs)


class InvalidInputError(ValidationError):
    """Input validation failed"""

    def __init__(self, field: str, value: Any, reason: str, **kwargs):
        message = f"Invalid input for '{field}': {reason}"
        kwargs.setdefault('context', {})
        kwargs['context'].update({
            'field': field,
            'value': str(value),
            'reason': reason
        })
        super().__init__(message, **kwargs)


class InvalidTaskError(ValidationError):
    """Task description or structure is invalid"""

    def __init__(self, task_description: str, reason: str, **kwargs):
        message = f"Invalid task: {reason}"
        kwargs.setdefault('context', {})
        kwargs['context'].update({
            'task': task_description,
            'reason': reason
        })
        super().__init__(message, **kwargs)


# =============================================================================
# Execution Errors
# =============================================================================

class ExecutionError(ACEError):
    """Base class for execution errors"""

    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('category', ErrorCategory.EXECUTION)
        super().__init__(message, **kwargs)


class PhaseExecutionError(ExecutionError):
    """Error during phase execution"""

    def __init__(self, phase_name: str, reason: str, **kwargs):
        message = f"Phase '{phase_name}' failed: {reason}"
        kwargs.setdefault('context', {})
        kwargs['context'].update({
            'phase': phase_name,
            'reason': reason
        })
        super().__init__(message, **kwargs)


class SubagentExecutionError(ExecutionError):
    """Error during subagent execution"""

    def __init__(self, agent_type: str, reason: str, **kwargs):
        message = f"Subagent '{agent_type}' failed: {reason}"
        kwargs.setdefault('context', {})
        kwargs['context'].update({
            'agent_type': agent_type,
            'reason': reason
        })
        super().__init__(message, **kwargs)


class CommandExecutionError(ExecutionError):
    """Error executing external command"""

    def __init__(self, command: str, returncode: int, stderr: str = "", **kwargs):
        message = f"Command failed with exit code {returncode}: {command}"
        kwargs.setdefault('context', {})
        kwargs['context'].update({
            'command': command,
            'returncode': returncode,
            'stderr': stderr
        })
        kwargs.setdefault('recovery_hint', "Check command syntax and permissions")
        super().__init__(message, **kwargs)


# =============================================================================
# Resource Errors
# =============================================================================

class ResourceError(ACEError):
    """Base class for resource-related errors"""

    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('category', ErrorCategory.RESOURCE)
        super().__init__(message, **kwargs)


class SubagentNotFoundError(ResourceError):
    """Requested subagent type not found in registry"""

    def __init__(self, agent_type: str, available_types: list, **kwargs):
        message = f"Subagent type '{agent_type}' not found"
        kwargs.setdefault('context', {})
        kwargs['context'].update({
            'requested_type': agent_type,
            'available_types': available_types
        })
        kwargs.setdefault('recovery_hint', f"Available types: {', '.join(available_types)}")
        super().__init__(message, **kwargs)


class FileSystemError(ResourceError):
    """File system operation failed"""

    def __init__(self, operation: str, path: str, reason: str, **kwargs):
        message = f"File system {operation} failed for '{path}': {reason}"
        kwargs.setdefault('context', {})
        kwargs['context'].update({
            'operation': operation,
            'path': path,
            'reason': reason
        })
        super().__init__(message, **kwargs)


class ContextLimitError(ResourceError):
    """Context token limit exceeded"""

    def __init__(self, current_tokens: int, max_tokens: int, **kwargs):
        message = f"Context limit exceeded: {current_tokens}/{max_tokens} tokens"
        kwargs.setdefault('severity', ErrorSeverity.HIGH)
        kwargs.setdefault('context', {})
        kwargs['context'].update({
            'current_tokens': current_tokens,
            'max_tokens': max_tokens,
            'overflow': current_tokens - max_tokens
        })
        kwargs.setdefault('recovery_hint', "Enable compression or reduce input size")
        super().__init__(message, **kwargs)


# =============================================================================
# Integration Errors
# =============================================================================

class IntegrationError(ACEError):
    """Base class for integration errors"""

    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('category', ErrorCategory.INTEGRATION)
        super().__init__(message, **kwargs)


class AIEngineError(IntegrationError):
    """Error communicating with AI engine"""

    def __init__(self, engine_name: str, reason: str, **kwargs):
        message = f"AI engine '{engine_name}' error: {reason}"
        kwargs.setdefault('context', {})
        kwargs['context'].update({
            'engine': engine_name,
            'reason': reason
        })
        super().__init__(message, **kwargs)


class APIError(IntegrationError):
    """External API call failed"""

    def __init__(self, api_name: str, status_code: Optional[int] = None, reason: str = "", **kwargs):
        message = f"API '{api_name}' failed"
        if status_code:
            message += f" with status {status_code}"
        if reason:
            message += f": {reason}"

        kwargs.setdefault('context', {})
        kwargs['context'].update({
            'api': api_name,
            'status_code': status_code,
            'reason': reason
        })
        super().__init__(message, **kwargs)


class MCPError(IntegrationError):
    """MCP protocol error"""

    def __init__(self, server_name: str, tool_name: Optional[str] = None, reason: str = "", **kwargs):
        message = f"MCP server '{server_name}' error"
        if tool_name:
            message += f" in tool '{tool_name}'"
        if reason:
            message += f": {reason}"

        kwargs.setdefault('context', {})
        kwargs['context'].update({
            'server': server_name,
            'tool': tool_name,
            'reason': reason
        })
        super().__init__(message, **kwargs)


# =============================================================================
# Timeout Errors
# =============================================================================

class TimeoutError(ACEError):
    """Base class for timeout errors"""

    def __init__(self, message: str, **kwargs):
        kwargs.setdefault('category', ErrorCategory.TIMEOUT)
        kwargs.setdefault('severity', ErrorSeverity.MEDIUM)
        super().__init__(message, **kwargs)


class PhaseTimeoutError(TimeoutError):
    """Phase execution timed out"""

    def __init__(self, phase_name: str, timeout_seconds: int, **kwargs):
        message = f"Phase '{phase_name}' timed out after {timeout_seconds}s"
        kwargs.setdefault('context', {})
        kwargs['context'].update({
            'phase': phase_name,
            'timeout': timeout_seconds
        })
        kwargs.setdefault('recovery_hint', "Increase timeout or simplify task")
        super().__init__(message, **kwargs)


class CommandTimeoutError(TimeoutError):
    """External command timed out"""

    def __init__(self, command: str, timeout_seconds: int, **kwargs):
        message = f"Command timed out after {timeout_seconds}s: {command}"
        kwargs.setdefault('context', {})
        kwargs['context'].update({
            'command': command,
            'timeout': timeout_seconds
        })
        super().__init__(message, **kwargs)


# =============================================================================
# Utility Functions
# =============================================================================

def wrap_exception(original: Exception, context: Optional[Dict[str, Any]] = None) -> ACEError:
    """
    Wrap a standard Python exception in an ACEError.

    Args:
        original: The original exception
        context: Additional context information

    Returns:
        ACEError wrapping the original exception
    """
    # Map common exceptions to specific ACE errors
    if isinstance(original, FileNotFoundError):
        return FileSystemError(
            operation="read",
            path=str(original.filename) if hasattr(original, 'filename') else "unknown",
            reason=str(original),
            cause=original,
            context=context
        )
    elif isinstance(original, PermissionError):
        return FileSystemError(
            operation="access",
            path=str(original.filename) if hasattr(original, 'filename') else "unknown",
            reason="Permission denied",
            cause=original,
            context=context
        )
    elif isinstance(original, ValueError):
        return ValidationError(
            message=str(original),
            cause=original,
            context=context
        )
    else:
        # Generic wrapping
        return ACEError(
            message=str(original),
            cause=original,
            context=context
        )


def format_error_for_logging(error: ACEError) -> str:
    """
    Format an ACEError for logging.

    Args:
        error: The error to format

    Returns:
        Formatted string suitable for logging
    """
    lines = [
        f"[{error.severity.value.upper()}] {error.__class__.__name__}",
        f"Message: {error.message}",
        f"Category: {error.category.value}"
    ]

    if error.context:
        lines.append("Context:")
        for key, value in error.context.items():
            lines.append(f"  {key}: {value}")

    if error.recovery_hint:
        lines.append(f"Recovery Hint: {error.recovery_hint}")

    if error.cause:
        lines.append(f"Caused by: {error.cause.__class__.__name__}: {error.cause}")

    return "\n".join(lines)


# =============================================================================
# Example Usage and Testing
# =============================================================================

if __name__ == "__main__":
    import json

    print("Example 1: SubagentNotFoundError")
    print("=" * 80)
    try:
        raise SubagentNotFoundError(
            agent_type="unknown",
            available_types=["research", "planning", "implementation"]
        )
    except ACEError as e:
        print(e)
        print("\nAs dict:")
        print(json.dumps(e.to_dict(), indent=2))

    print("\n" + "=" * 80)
    print("Example 2: PhaseExecutionError with cause")
    print("=" * 80)
    try:
        try:
            raise ValueError("Invalid model response")
        except ValueError as ve:
            raise PhaseExecutionError(
                phase_name="Research",
                reason="AI engine returned invalid response",
                cause=ve,
                severity=ErrorSeverity.HIGH
            )
    except ACEError as e:
        print(format_error_for_logging(e))

    print("\n" + "=" * 80)
    print("Example 3: ContextLimitError")
    print("=" * 80)
    try:
        raise ContextLimitError(
            current_tokens=10000,
            max_tokens=8192
        )
    except ACEError as e:
        print(e)

    print("\n" + "=" * 80)
    print("Example 4: Wrapping standard exception")
    print("=" * 80)
    try:
        try:
            with open("/nonexistent/file.txt") as f:
                pass
        except Exception as e:
            wrapped = wrap_exception(e, context={'operation': 'load_config'})
            raise wrapped
    except ACEError as e:
        print(format_error_for_logging(e))
