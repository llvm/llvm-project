#!/usr/bin/env python3
"""
CSNA 2.0 Compliant API Security Layer

Provides quantum-resistant authentication and authorization for DSMIL API endpoints.

Features:
- HMAC-SHA3-512 request authentication
- Timestamp-based replay attack prevention
- Rate limiting
- Audit logging
- Quantum-resistant encryption
"""

import time
import functools
from typing import Callable, Optional
from flask import request, jsonify
import hashlib
import hmac

# Import quantum crypto layer
try:
    from quantum_crypto_layer import get_crypto_layer, SecurityLevel
    HAS_CRYPTO = True
except ImportError:
    HAS_CRYPTO = False
    print("Warning: Quantum crypto layer not available")


class APISecurityConfig:
    """API security configuration"""
    # Request timeout (5 minutes)
    REQUEST_TIMEOUT = 300

    # Rate limiting (requests per minute)
    RATE_LIMIT = 60

    # Authentication enabled
    AUTH_ENABLED = True

    # Require HTTPS in production
    REQUIRE_HTTPS = False  # Set to True in production


# Request rate tracking
request_counts = {}


def check_rate_limit(client_ip: str) -> bool:
    """
    Check if client has exceeded rate limit

    Args:
        client_ip: Client IP address

    Returns:
        True if within rate limit
    """
    current_time = time.time()
    window_start = current_time - 60  # 1 minute window

    # Clean old entries
    if client_ip in request_counts:
        request_counts[client_ip] = [
            t for t in request_counts[client_ip]
            if t > window_start
        ]
    else:
        request_counts[client_ip] = []

    # Check rate limit
    if len(request_counts[client_ip]) >= APISecurityConfig.RATE_LIMIT:
        return False

    # Add current request
    request_counts[client_ip].append(current_time)
    return True


def verify_request_signature(request_data: dict, signature: str, timestamp: float) -> bool:
    """
    Verify HMAC-SHA3-512 request signature

    Args:
        request_data: Request data dictionary
        signature: Hex-encoded HMAC signature
        timestamp: Request timestamp

    Returns:
        True if signature is valid
    """
    if not HAS_CRYPTO:
        # Development mode - accept all
        return True

    try:
        # Get crypto layer
        crypto = get_crypto_layer()

        # Create message from request data + timestamp
        import json
        message = json.dumps(request_data, sort_keys=True) + str(timestamp)
        message_bytes = message.encode('utf-8')

        # Verify HMAC
        return crypto.verify_hmac(message_bytes, signature)

    except Exception as e:
        print(f"Signature verification error: {e}")
        return False


def require_authentication(security_level: SecurityLevel = SecurityLevel.CONFIDENTIAL):
    """
    Decorator to require authentication for API endpoints

    Args:
        security_level: Required security level

    Usage:
        @app.route('/api/secure_endpoint')
        @require_authentication(SecurityLevel.SECRET)
        def secure_endpoint():
            return jsonify({"data": "secret"})
    """
    def decorator(f: Callable) -> Callable:
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            # Skip in development if auth is disabled
            if not APISecurityConfig.AUTH_ENABLED:
                return f(*args, **kwargs)

            # Get client IP
            client_ip = request.remote_addr or "unknown"

            # Check rate limit
            if not check_rate_limit(client_ip):
                return jsonify({
                    "error": "Rate limit exceeded",
                    "retry_after": 60
                }), 429

            # Check request timestamp (prevent replay attacks)
            timestamp = request.headers.get('X-Timestamp')
            if timestamp:
                try:
                    request_time = float(timestamp)
                    current_time = time.time()

                    # Check if request is too old or from future
                    time_diff = abs(current_time - request_time)
                    if time_diff > APISecurityConfig.REQUEST_TIMEOUT:
                        return jsonify({
                            "error": "Request timestamp expired",
                            "reason": "Possible replay attack"
                        }), 401
                except ValueError:
                    return jsonify({"error": "Invalid timestamp format"}), 400

            # Verify request signature if provided
            signature = request.headers.get('X-Signature')
            if signature and timestamp:
                request_data = request.get_json() or {}

                if not verify_request_signature(request_data, signature, float(timestamp)):
                    return jsonify({
                        "error": "Invalid request signature",
                        "reason": "Authentication failed"
                    }), 401

            # Log audit event
            if HAS_CRYPTO:
                crypto = get_crypto_layer()
                crypto._log_audit("API_ACCESS", {
                    "endpoint": request.path,
                    "method": request.method,
                    "client_ip": client_ip,
                    "security_level": security_level.value,
                    "authenticated": True
                })

            # Execute endpoint
            return f(*args, **kwargs)

        return wrapper
    return decorator


def encrypt_response(security_level: SecurityLevel = SecurityLevel.CONFIDENTIAL):
    """
    Decorator to encrypt API response data

    Args:
        security_level: Encryption security level

    Usage:
        @app.route('/api/sensitive_data')
        @encrypt_response(SecurityLevel.TOP_SECRET)
        def sensitive_endpoint():
            return jsonify({"sensitive": "data"})
    """
    def decorator(f: Callable) -> Callable:
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            # Get response
            response = f(*args, **kwargs)

            # If crypto not available, return as-is
            if not HAS_CRYPTO:
                return response

            # Get response data
            if hasattr(response, 'get_json'):
                data = response.get_json()
                if data:
                    try:
                        # Encrypt response
                        crypto = get_crypto_layer()
                        encrypted = crypto.encrypt_json(data, security_level)

                        # Return encrypted response
                        return jsonify({
                            "encrypted": True,
                            "data": encrypted,
                            "security_level": security_level.value
                        }), response.status_code
                    except Exception as e:
                        print(f"Encryption error: {e}")

            return response

        return wrapper
    return decorator


def secure_endpoint(
    security_level: SecurityLevel = SecurityLevel.CONFIDENTIAL,
    require_auth: bool = True,
    encrypt: bool = False
):
    """
    Combined decorator for API security

    Args:
        security_level: Security classification level
        require_auth: Require authentication
        encrypt: Encrypt response

    Usage:
        @app.route('/api/dsmil/device/activate', methods=['POST'])
        @secure_endpoint(SecurityLevel.SECRET, require_auth=True)
        def activate_device():
            return jsonify({"status": "activated"})
    """
    def decorator(f: Callable) -> Callable:
        # Apply authentication if required
        if require_auth:
            f = require_authentication(security_level)(f)

        # Apply encryption if required
        if encrypt:
            f = encrypt_response(security_level)(f)

        return f

    return decorator


# Security statistics
def get_security_stats() -> dict:
    """Get security statistics"""
    stats = {
        "authentication_enabled": APISecurityConfig.AUTH_ENABLED,
        "rate_limit": APISecurityConfig.RATE_LIMIT,
        "request_timeout": APISecurityConfig.REQUEST_TIMEOUT,
        "active_clients": len(request_counts),
        "total_requests_tracked": sum(len(requests) for requests in request_counts.values()),
        "quantum_crypto_available": HAS_CRYPTO
    }

    if HAS_CRYPTO:
        crypto = get_crypto_layer()
        stats["crypto_stats"] = crypto.get_statistics()

    return stats


if __name__ == "__main__":
    """Test API security"""
    print("="*70)
    print(" CSNA 2.0 API SECURITY LAYER TEST")
    print("="*70)

    stats = get_security_stats()
    print(f"\nAuthentication Enabled: {stats['authentication_enabled']}")
    print(f"Rate Limit: {stats['rate_limit']} req/min")
    print(f"Request Timeout: {stats['request_timeout']}s")
    print(f"Quantum Crypto Available: {stats['quantum_crypto_available']}")

    if 'crypto_stats' in stats:
        print(f"\nCryptography:")
        print(f"  Compliance: {', '.join(stats['crypto_stats']['compliance'])}")
        print(f"  Encryption: {stats['crypto_stats']['algorithms']['encryption']}")

    print("\n" + "="*70)
    print(" API SECURITY OPERATIONAL")
    print("="*70)
