# DSMIL AI MCP Server - Security Documentation

## Overview

The DSMIL AI MCP Server implements multiple layers of security to protect against cyber attacks and unauthorized access. This document details the security architecture, threat model, and hardening measures.

**Classification:** UNCLASSIFIED // FOR OFFICIAL USE ONLY
**Version:** 1.1.0 (Security Hardened)
**Last Updated:** 2025-11-06

---

## Security Features

### 1. Authentication

**Token-Based Authentication:**
- SHA-256 hashed authentication tokens
- Constant-time comparison to prevent timing attacks
- Tokens stored with file permissions 0x600 (owner read/write only)
- Token generation uses cryptographically secure random (os.urandom)

**Configuration Location:**
```
~/.dsmil/mcp_security.json
```

**Generating a Token:**
```python
from mcp_security import get_security_manager
security = get_security_manager()
token = security.generate_token()
print(f"Auth Token: {token}")
```

**Current Status:** Authentication is enabled by default but not enforced during initial deployment. Will require enforcement in production.

---

### 2. Rate Limiting

**Protection Against:**
- Denial of Service (DoS) attacks
- Resource exhaustion
- Brute force attempts

**Limits:**
- **60 requests per minute** per client per tool
- **10 burst requests** allowed
- Per-client tracking using unique client IDs
- Time-window based reset (sliding window)

**Implementation:**
- Client ID generated from: `sha256(hostname:pid)`
- Rate limit state maintained in memory
- Resets every 60 seconds

**Response on Limit Exceeded:**
```
Error: Rate limit exceeded. Please wait before retrying.
```

---

### 3. Input Validation

#### Query Validation

**Maximum Lengths:**
- AI queries: 10,000 characters
- File paths: 4,096 characters

**Blocked Patterns:**
- SQL injection attempts: `'; DROP TABLE`, `' OR '1'='1`
- XSS attempts: `<script>`, `javascript:`
- Path traversal: `../../`
- System file access: `/etc/passwd`, `/etc/shadow`

**Implementation:**
```python
valid, error = security.validate_query(query)
if not valid:
    # Request rejected and audited
    return error_response
```

#### File Path Validation

**Security Checks:**
1. **Path Length** - Maximum 4096 characters
2. **Path Traversal Prevention** - Blocks `../` sequences
3. **Blocked Paths** - `/etc/shadow`, `/etc/passwd`, `/root`, `/boot`, `/sys`, `/proc`
4. **File Extension Validation** - Only allowed types accepted
5. **Sandboxing** - Restricts to allowed directories
6. **Dotfile Protection** - Blocks access to hidden files

**Allowed File Extensions:**
```
.pdf, .txt, .md, .log
.c, .h, .cpp, .hpp
.py, .sh, .json, .yaml
```

**Allowed Directories (Sandboxing):**
```
$HOME (user home directory)
/tmp
/var/tmp
```

**Example Validation:**
```python
valid, error = security.validate_filepath("/home/user/document.pdf")
# Returns: (True, None)

valid, error = security.validate_filepath("/etc/shadow")
# Returns: (False, "Access to /etc is blocked")
```

---

### 4. Audit Logging

**All Operations Logged:**
- Tool invocations
- Authentication attempts (success/failure)
- Rate limit violations
- Input validation failures
- Exceptions and errors

**Log Location:**
```
~/.dsmil/mcp_audit.log
```

**Log Format:**
```
2025-11-06 12:34:56 | INFO | {"timestamp": "2025-11-06T12:34:56", "client_id": "abc123", "tool": "dsmil_ai_query", "arguments": {"query_length": 42, "model": "fast"}, "success": true, "error": null}
```

**Sensitive Data Handling:**
- Passwords, tokens, secrets are redacted as `[REDACTED]`
- Long strings truncated to 1000 chars with `[TRUNCATED]`
- Filenames logged but not file contents

**Audit Log Rotation:**
- Manual rotation recommended
- Logs append-only
- File permissions: 0x600 (owner read/write only)

---

### 5. Sandboxing

**File Access Restrictions:**
- Only allowed directories accessible
- Blocks system directories (`/etc`, `/root`, `/boot`, `/sys`, `/proc`)
- Blocks dotfiles/hidden files (configurable)
- Path resolution to absolute paths (prevents symlink attacks)

**Configuration:**
```json
{
  "sandboxing": {
    "restrict_file_access": true,
    "allowed_directories": [
      "/home/user",
      "/tmp",
      "/var/tmp"
    ],
    "deny_dotfiles": true
  }
}
```

**Symlink Protection:**
- All paths resolved with `Path.resolve()` before validation
- Prevents escaping sandbox via symbolic links

---

## Threat Model

### Threats Mitigated

| Threat | Mitigation | Status |
|--------|-----------|--------|
| **Unauthorized Access** | Token authentication | ✅ Implemented |
| **DoS Attacks** | Rate limiting (60 req/min) | ✅ Implemented |
| **Resource Exhaustion** | Rate limiting + input size limits | ✅ Implemented |
| **SQL Injection** | Input validation, pattern detection | ✅ Implemented |
| **XSS Attacks** | Input validation, script tag detection | ✅ Implemented |
| **Path Traversal** | Path validation, sandboxing | ✅ Implemented |
| **Privilege Escalation** | Sandboxing, blocked paths | ✅ Implemented |
| **Information Disclosure** | Audit log redaction, error sanitization | ✅ Implemented |
| **Timing Attacks** | Constant-time token comparison | ✅ Implemented |
| **Symlink Attacks** | Path resolution before validation | ✅ Implemented |

### Residual Risks

| Risk | Severity | Mitigation Status |
|------|----------|-------------------|
| **TOCTOU Attacks** | Low | Partially mitigated (path resolution) |
| **Memory Exhaustion** | Medium | Needs monitoring (future work) |
| **AI Prompt Injection** | Medium | Input validation, content filters (partial) |
| **Side-Channel Attacks** | Low | Minimal sensitive computation |

---

## Security Configuration

### Default Configuration

```json
{
  "authentication": {
    "enabled": true,
    "token_hash": null,
    "require_token": true
  },
  "rate_limiting": {
    "enabled": true,
    "requests_per_minute": 60,
    "burst_requests": 10
  },
  "input_validation": {
    "max_query_length": 10000,
    "max_filepath_length": 4096,
    "allowed_file_extensions": [".pdf", ".txt", ".md", ".log", ".c", ".h", ".py", ".sh"],
    "blocked_paths": ["/etc/shadow", "/etc/passwd", "/root", "/boot", "/sys", "/proc"]
  },
  "audit": {
    "enabled": true,
    "log_all_requests": true,
    "log_failed_auth": true,
    "log_rate_limit": true
  },
  "sandboxing": {
    "restrict_file_access": true,
    "allowed_directories": ["$HOME", "/tmp", "/var/tmp"],
    "deny_dotfiles": true
  }
}
```

### Hardening Recommendations

#### Production Deployment

1. **Enable Token Authentication:**
   ```python
   from mcp_security import get_security_manager
   security = get_security_manager()
   token = security.generate_token()
   # Securely share token with authorized clients
   ```

2. **Tighten Rate Limits:**
   ```json
   {
     "rate_limiting": {
       "requests_per_minute": 30,
       "burst_requests": 5
     }
   }
   ```

3. **Restrict Sandboxing:**
   ```json
   {
     "sandboxing": {
       "allowed_directories": ["/home/user/safe_directory"]
     }
   }
   ```

4. **Monitor Audit Logs:**
   ```bash
   tail -f ~/.dsmil/mcp_audit.log | grep -E "WARNING|ERROR"
   ```

5. **Regular Security Audits:**
   - Review audit logs weekly
   - Check for failed authentication attempts
   - Monitor rate limit violations
   - Validate security configuration

---

## Attack Surface Analysis

### Entry Points

1. **MCP stdio Interface** (Primary)
   - Communication via stdin/stdout
   - MCP protocol messages
   - Mitigated by: Rate limiting, input validation

2. **AI Query Tool** (`dsmil_ai_query`)
   - User-provided prompts
   - Mitigated by: Query validation, length limits, pattern detection

3. **RAG File Operations** (`dsmil_rag_add_file`, `dsmil_rag_add_folder`)
   - User-provided file paths
   - Mitigated by: Path validation, sandboxing, extension filtering

4. **RAG Search Tool** (`dsmil_rag_search`)
   - User-provided search queries
   - Mitigated by: Query validation, length limits

### Attack Scenarios

#### Scenario 1: Path Traversal Attack
**Attack:** Client attempts to read `/etc/shadow`
```json
{
  "tool": "dsmil_rag_add_file",
  "arguments": {"filepath": "../../../../etc/shadow"}
}
```

**Defense:**
1. Path normalized to absolute: `/etc/shadow`
2. Blocked paths check fails
3. Request denied: `"Access to /etc is blocked"`
4. Audit log entry created

**Result:** ✅ Attack blocked

---

#### Scenario 2: DoS via Request Flooding
**Attack:** Client sends 100 requests in 10 seconds

**Defense:**
1. First 60 requests succeed
2. Request 61 denied: `"Rate limit exceeded"`
3. Client must wait 60 seconds for reset
4. All attempts logged

**Result:** ✅ Attack mitigated

---

#### Scenario 3: SQL Injection in Query
**Attack:** Client sends malicious query
```json
{
  "tool": "dsmil_ai_query",
  "arguments": {"query": "What is 2+2?'; DROP TABLE users; --"}
}
```

**Defense:**
1. Query validation detects: `'; DROP TABLE`
2. Request denied: `"Query contains suspicious pattern"`
3. Audit log entry with WARNING level
4. No database access (DSMIL uses file-based storage)

**Result:** ✅ Attack blocked

---

#### Scenario 4: Unauthorized Access
**Attack:** Client without auth token attempts access

**Defense:**
1. Authentication check fails (no token or invalid token)
2. Request denied (if token enforcement enabled)
3. Failed auth logged
4. Client cannot proceed

**Result:** ✅ Attack blocked (when auth enforced)

---

## Compliance

### Standards

- **NIST SP 800-53** - Security and Privacy Controls
  - AC-2: Account Management (token-based)
  - AU-2: Audit Events (comprehensive logging)
  - SC-7: Boundary Protection (sandboxing)
  - SI-10: Information Input Validation

- **OWASP Top 10** (2021)
  - A01: Broken Access Control → Token auth, sandboxing
  - A03: Injection → Input validation, pattern detection
  - A05: Security Misconfiguration → Secure defaults
  - A09: Security Logging Failures → Comprehensive audit

### MIL-SPEC Integration

- Operates within **Mode 5** security constraints
- Logs integrated with DSMIL audit framework
- Compatible with hardware attestation (TPM)
- Post-Quantum Cryptography ready (via TPM device)

---

## Monitoring and Incident Response

### Real-Time Monitoring

**Check Security Status:**
```bash
# Via MCP client
Use dsmil_security_status

# Via Python
python3 -c "from mcp_security import get_security_manager; print(get_security_manager().get_security_status())"
```

**Monitor Audit Log:**
```bash
tail -f ~/.dsmil/mcp_audit.log
```

**Filter for Security Events:**
```bash
grep -E "WARNING|ERROR|failed|blocked|exceeded" ~/.dsmil/mcp_audit.log
```

### Incident Response

**Suspected Breach:**
1. Check audit logs for anomalies
2. Review failed authentication attempts
3. Check rate limit violations
4. Verify security configuration

**Reset Security:**
```bash
# Regenerate auth token
python3 -c "from mcp_security import get_security_manager; print(get_security_manager().generate_token())"

# Clear rate limits (restart server)
pkill -f dsmil_mcp_server.py

# Review and tighten configuration
vim ~/.dsmil/mcp_security.json
```

---

## Testing and Validation

### Security Tests

**Test 1: Path Traversal Prevention**
```python
from mcp_security import get_security_manager
security = get_security_manager()

# Should fail
valid, error = security.validate_filepath("../../../etc/passwd")
assert not valid, "Path traversal not blocked!"
```

**Test 2: Rate Limiting**
```python
# Send 65 requests rapidly
for i in range(65):
    result = security.check_rate_limit("test_client", "test_tool")
    if i < 60:
        assert result, f"Request {i} should succeed"
    else:
        assert not result, f"Request {i} should be rate limited"
```

**Test 3: Input Validation**
```python
# Should fail - SQL injection
valid, error = security.validate_query("'; DROP TABLE users; --")
assert not valid, "SQL injection not detected!"

# Should fail - XSS
valid, error = security.validate_query("<script>alert('xss')</script>")
assert not valid, "XSS not detected!"
```

---

## Future Enhancements

### Planned Features

1. **TLS/mTLS Support** - Encrypted transport (when MCP supports it)
2. **Role-Based Access Control (RBAC)** - Different permission levels
3. **IP Whitelisting** - Restrict to known clients
4. **Anomaly Detection** - ML-based threat detection
5. **SIEM Integration** - Export to security monitoring systems
6. **Hardware Token Support** - YubiKey, TPM-based auth

### Under Consideration

- **Request Signing** - Cryptographic request verification
- **Quota Management** - Per-client resource quotas
- **Geofencing** - Restrict by geographic location
- **Time-Based Access Control** - Allow/deny by time of day

---

## References

- **MCP Specification:** https://modelcontextprotocol.io/
- **OWASP Top 10:** https://owasp.org/Top10/
- **NIST SP 800-53:** https://csrc.nist.gov/publications/detail/sp/800-53/rev-5/final
- **DSMIL Security Documentation:** `/home/user/LAT5150DRVMIL/00-documentation/`

---

**Classification:** UNCLASSIFIED // FOR OFFICIAL USE ONLY
**Contact:** DSMIL Security Team
**Last Review:** 2025-11-06
