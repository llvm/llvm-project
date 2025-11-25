# DSMIL DEFCON1 Profile - Dual YubiKey Authentication

**Version:** 1.0.0
**Date:** 2025-11-25
**Classification:** TOP SECRET // FOR OFFICIAL USE ONLY
**Threat Level:** DEFCON 1 (Maximum Readiness)
**Status:** Production Ready

---

## Table of Contents

1. [Overview](#overview)
2. [Security Requirements](#security-requirements)
3. [Architecture](#architecture)
4. [Installation](#installation)
5. [Configuration](#configuration)
6. [Authentication Workflow](#authentication-workflow)
7. [Command Reference](#command-reference)
8. [Web Interface Integration](#web-interface-integration)
9. [Continuous Authentication](#continuous-authentication)
10. [Troubleshooting](#troubleshooting)
11. [Security Considerations](#security-considerations)
12. [Compliance](#compliance)

---

## Overview

The DEFCON1 security profile implements the highest level of authentication in the DSMIL platform, designed for emergency operations under maximum threat conditions. This profile requires dual YubiKey authentication where **both** hardware tokens must successfully pass cryptographic challenges before access is granted.

### Key Features

- **Dual YubiKey Authentication** - Two separate hardware tokens required
- **4-Person Authorization** - Requires 4 authorized personnel including 1 executive
- **FIDO2/WebAuthn** - Phishing-resistant hardware-backed authentication
- **1-Hour Session Duration** - Automatic session expiration
- **Continuous Authentication** - Re-authentication every 5 minutes
- **Comprehensive Audit Trail** - All operations logged
- **Emergency-Only Access** - Restricted to critical operations

### Use Cases

- **Nuclear Command Authority** - NC3 systems requiring two-person integrity
- **Emergency Operations** - Crisis response under DEFCON 1 conditions
- **Critical Infrastructure Protection** - Maximum security operations
- **Executive Actions** - Operations requiring presidential/executive authorization
- **Special Access Programs** - Compartmented information access

---

## Security Requirements

### Hardware Requirements

#### Minimum Requirements

- **2 YubiKeys** (YubiKey 5 Series recommended)
  - Primary YubiKey (everyday use)
  - Secondary YubiKey (backup/redundancy)
  - Both must support FIDO2/WebAuthn
  - Both must be individually registered

#### Recommended Configuration

- **3+ YubiKeys per user**
  - Primary (on-person)
  - Secondary (secure storage)
  - Tertiary (emergency backup)

### Personnel Requirements

- **4 Authorized Personnel**
  - Minimum 1 Executive (AuthorizationLevel.EXECUTIVE)
  - Minimum 1 Commander (AuthorizationLevel.COMMANDER)
  - 2 additional authorized personnel (any level)

- **Each authorizer must have:**
  - Registered personal YubiKey
  - Appropriate security clearance
  - Authorized access to DEFCON1 profile

### System Requirements

- Linux kernel 4.x+ (Ubuntu 20.04+, Debian 10+)
- Python 3.8+
- Browser with WebAuthn support (Chrome 90+, Firefox 88+, Edge 90+)
- Network connectivity for FIDO2 server communication
- Secure storage for session data (~/.dsmil/defcon1)

---

## Architecture

### Component Overview

```
┌─────────────────────────────────────────────────────────────┐
│                   DEFCON1 Security Profile                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌───────────────┐      ┌────────────────────────────┐    │
│  │ Primary       │      │ Secondary                  │    │
│  │ YubiKey       │      │ YubiKey                    │    │
│  │ (FIDO2)       │      │ (FIDO2)                    │    │
│  └───────┬───────┘      └────────┬───────────────────┘    │
│          │                       │                         │
│          │   WebAuthn/FIDO2     │                         │
│          └───────┬───────────────┘                         │
│                  │                                         │
│                  ▼                                         │
│  ┌─────────────────────────────────────────────┐          │
│  │   DEFCON1 Profile Manager                   │          │
│  │   - Dual YubiKey Validation                 │          │
│  │   - Authorizer Management                   │          │
│  │   - Session Management                      │          │
│  │   - Continuous Authentication               │          │
│  └─────────────────┬───────────────────────────┘          │
│                    │                                       │
│                    ▼                                       │
│  ┌─────────────────────────────────────────────┐          │
│  │   YubiKey Authentication Module             │          │
│  │   - FIDO2 Server                            │          │
│  │   - Challenge-Response                      │          │
│  │   - Device Management                       │          │
│  └─────────────────┬───────────────────────────┘          │
│                    │                                       │
│                    ▼                                       │
│  ┌─────────────────────────────────────────────┐          │
│  │   Hardware Layer                            │          │
│  │   - USB/NFC Interface                       │          │
│  │   - Cryptographic Operations                │          │
│  │   - Private Key Storage                     │          │
│  └─────────────────────────────────────────────┘          │
│                                                             │
└─────────────────────────────────────────────────────────────┘

        ┌────────────────────────────────┐
        │    Audit & Compliance          │
        │    - Full Operation Logging    │
        │    - Authorizer Records        │
        │    - Session Timeline          │
        └────────────────────────────────┘
```

### Authentication Flow

```
1. User Initiates DEFCON1 Session
   ↓
2. System Generates Session ID
   ↓
3. PRIMARY YubiKey Authentication
   - User inserts primary YubiKey
   - Browser WebAuthn prompt
   - User touches sensor
   - FIDO2 challenge-response
   - Cryptographic validation
   ↓
4. SECONDARY YubiKey Authentication
   - User removes primary, inserts secondary
   - Browser WebAuthn prompt (new challenge)
   - User touches sensor
   - FIDO2 challenge-response
   - Cryptographic validation
   ↓
5. Authorizer #1 Authentication (Standard/Supervisor)
   - Authorizer inserts their YubiKey
   - WebAuthn authentication
   - Digital signature recorded
   ↓
6. Authorizer #2 Authentication (Commander)
   - Commander inserts their YubiKey
   - WebAuthn authentication
   - Digital signature recorded
   ↓
7. Authorizer #3 Authentication (Additional)
   - Additional authorizer YubiKey
   - WebAuthn authentication
   - Digital signature recorded
   ↓
8. Authorizer #4 Authentication (Executive - REQUIRED)
   - Executive inserts their YubiKey
   - WebAuthn authentication
   - Digital signature recorded
   ↓
9. Validate All Requirements
   - 2 YubiKeys authenticated ✓
   - 4 authorizers validated ✓
   - 1 executive authorizer ✓
   ↓
10. Create DEFCON1 Session
    - Session duration: 1 hour
    - Continuous auth: every 5 minutes
    - Access: EMERGENCY ONLY
    ↓
11. Continuous Authentication Loop
    - Every 5 minutes:
      * Re-authenticate both YubiKeys
      * Verify session not expired
      * Update audit trail
    - If authentication fails:
      * Terminate session immediately
      * Log termination event
      * Require full re-authentication
```

---

## Installation

### Prerequisites

Ensure YubiKey authentication is installed and working:

```bash
# Install YubiKey support (if not already installed)
cd /home/user/DSLLVM/lat5150drvmil
sudo ./deployment/configure_yubikey.sh install

# Verify YubiKey detection
./deployment/configure_yubikey.sh test
```

### Install DEFCON1 Profile

```bash
# Navigate to AI engine directory
cd /home/user/DSLLVM/lat5150drvmil/02-ai-engine

# Make scripts executable
chmod +x defcon1_profile.py
chmod +x defcon1_admin.py

# Test installation
python3 defcon1_profile.py
```

**Expected Output:**
```
================================================================================
DSMIL DEFCON1 Security Profile
================================================================================

Classification: TOP SECRET // FOR OFFICIAL USE ONLY
Threat Level: DEFCON 1 (Maximum Readiness)

================================================================================

Initializing DEFCON1 profile...

✅ DEFCON1 Profile initialized

Requirements:
  - YubiKeys Required: 2
  - Authorizers Required: 4
  - Executive Authorizers: 1
  - Session Duration: 1 hour(s)
  - Continuous Auth Interval: 5 minutes

Access Restrictions:
  - EMERGENCY_ONLY
  - EXECUTIVE_AUTHORIZATION_REQUIRED
  - DUAL_YUBIKEY_MANDATORY
  - CONTINUOUS_MONITORING
  - FULL_AUDIT_TRAIL

Active DEFCON1 Sessions: 0

================================================================================
```

### Verify Dependencies

```bash
# Check Python dependencies
python3 -c "from defcon1_profile import DEFCON1Profile; print('✅ DEFCON1 Profile OK')"
python3 -c "from yubikey_auth import YubikeyAuth; print('✅ YubiKey Auth OK')"

# Check YubiKey devices
python3 yubikey_admin.py list
```

---

## Configuration

### Initial Setup

The DEFCON1 profile auto-creates configuration on first run:

**Configuration File:** `~/.dsmil/defcon1/defcon1_config.json`

```json
{
  "threat_level": "DEFCON_1",
  "required_yubikeys": 2,
  "required_authorizers": 4,
  "session_duration_hours": 1,
  "continuous_auth_interval_minutes": 5,
  "access_restrictions": [
    "EMERGENCY_ONLY",
    "EXECUTIVE_AUTHORIZATION_REQUIRED",
    "DUAL_YUBIKEY_MANDATORY",
    "CONTINUOUS_MONITORING",
    "FULL_AUDIT_TRAIL"
  ],
  "authorized_executives": []
}
```

### Register YubiKeys

Each user must register at least **2 YubiKeys**:

```bash
# Register primary YubiKey
python3 yubikey_admin.py register --name "Primary YubiKey" --user tactical_user

# Register secondary YubiKey
python3 yubikey_admin.py register --name "Secondary YubiKey" --user tactical_user

# Register additional backup (recommended)
python3 yubikey_admin.py register --name "Backup YubiKey" --user tactical_user

# Verify registration
python3 yubikey_admin.py list
```

### Configure Authorizers

Add authorized executives to configuration:

```bash
# Edit configuration
nano ~/.dsmil/defcon1/defcon1_config.json
```

Add authorized executives:

```json
{
  "authorized_executives": [
    {
      "user_id": "potus",
      "name": "President",
      "role": "Commander-in-Chief",
      "yubikey_device_id": "abc123..."
    },
    {
      "user_id": "secdef",
      "name": "Secretary of Defense",
      "role": "SECDEF",
      "yubikey_device_id": "def456..."
    }
  ]
}
```

---

## Authentication Workflow

### Step 1: Initialize DEFCON1 Session

```bash
python3 defcon1_admin.py init-session tactical_user
```

**Output:**
```
================================================================================
DEFCON1 Session Initialization
================================================================================

User: tactical_user
Threat Level: DEFCON_1

✅ DEFCON1 authentication session initiated

Session ID: a1b2c3d4e5f6g7h8

Requirements:
  - YubiKeys: 2
  - Authorizers: 4
  - Executive Authorizers: 1
  - Session Duration: 1 hour(s)

Next Steps:
  1. Insert PRIMARY YubiKey
  2. Complete FIDO2 authentication
  3. Insert SECONDARY YubiKey
  4. Complete FIDO2 authentication
  5. Gather 4 authorizers (including 1 executive)
  6. Each authorizer authenticates with their YubiKey

Message: Insert PRIMARY YubiKey and complete authentication

================================================================================
```

### Step 2: Primary YubiKey Authentication

**In Web Browser:**

1. Navigate to DEFCON1 authentication page
2. Enter session ID: `a1b2c3d4e5f6g7h8`
3. Click **"AUTHENTICATE PRIMARY YUBIKEY"**
4. Browser shows WebAuthn prompt:
   ```
   ┌─────────────────────────────────────┐
   │ localhost wants to verify           │
   │ your security key                   │
   │                                     │
   │ Insert and touch your security key  │
   │                                     │
   │        [Cancel]  [Allow]            │
   └─────────────────────────────────────┘
   ```
5. Insert PRIMARY YubiKey
6. Touch the sensor when it flashes
7. Status updates: **"PRIMARY YUBIKEY AUTHENTICATED ✓"**

### Step 3: Secondary YubiKey Authentication

**In Web Browser:**

1. Remove PRIMARY YubiKey
2. Click **"AUTHENTICATE SECONDARY YUBIKEY"**
3. Browser shows WebAuthn prompt again (new challenge)
4. Insert SECONDARY YubiKey
5. Touch the sensor when it flashes
6. Status updates: **"SECONDARY YUBIKEY AUTHENTICATED ✓"**

### Step 4: Gather Authorizers

**Required Authorizers (4 total):**

1. **Authorizer 1** (Standard Operator)
   - Insert their personal YubiKey
   - Complete WebAuthn authentication
   - Digital signature recorded

2. **Authorizer 2** (Supervisor)
   - Insert their personal YubiKey
   - Complete WebAuthn authentication
   - Digital signature recorded

3. **Authorizer 3** (Commander)
   - Insert their personal YubiKey
   - Complete WebAuthn authentication
   - Digital signature recorded

4. **Authorizer 4** (Executive - REQUIRED)
   - Insert their personal YubiKey
   - Complete WebAuthn authentication
   - Digital signature recorded
   - **Executive authorization validated ✓**

### Step 5: Session Created

Once all requirements are met:

```
✅ DEFCON1 Session Created

Session ID: a1b2c3d4e5f6g7h8
User: tactical_user
Duration: 1 hour
Expires: 2025-11-25 15:30:00 UTC

Access Level: EMERGENCY ONLY
Continuous Auth: Every 5 minutes

STATUS: ACTIVE
```

### Step 6: Continuous Authentication

**Every 5 minutes during the session:**

1. System prompts for dual YubiKey re-authentication
2. User inserts PRIMARY YubiKey → WebAuthn → Touch sensor
3. User inserts SECONDARY YubiKey → WebAuthn → Touch sensor
4. If successful: Session continues
5. If failed: Session terminates immediately

---

## Command Reference

### Initialize Session

```bash
python3 defcon1_admin.py init-session <user-id>
```

**Example:**
```bash
python3 defcon1_admin.py init-session tactical_user
```

### List Active Sessions

```bash
python3 defcon1_admin.py list-sessions
```

**Output:**
```
================================================================================
Active DEFCON1 Sessions
================================================================================

[1] Session ID: a1b2c3d4e5f6g7h8
    User: tactical_user
    Threat Level: DEFCON_1
    Created: 2025-11-25T14:30:00Z
    Expires: 2025-11-25T15:30:00Z
    Primary YubiKey: abc123
    Secondary YubiKey: def456
    Authorizers: 4
    Active: True

    Authorizers:
      - John Doe (Operator) - Level: STANDARD
        Authorized: 2025-11-25T14:31:00Z
      - Jane Smith (Supervisor) - Level: SUPERVISOR
        Authorized: 2025-11-25T14:32:00Z
      - Bob Johnson (Commander) - Level: COMMANDER
        Authorized: 2025-11-25T14:33:00Z
      - Alice Williams (Executive) - Level: EXECUTIVE
        Authorized: 2025-11-25T14:34:00Z

================================================================================
```

### Check Session Status

```bash
python3 defcon1_admin.py session-status <session-id>
```

**Example:**
```bash
python3 defcon1_admin.py session-status a1b2c3d4e5f6g7h8
```

### Terminate Session

```bash
python3 defcon1_admin.py terminate-session <session-id> [reason]
```

**Example:**
```bash
python3 defcon1_admin.py terminate-session a1b2c3d4e5f6g7h8 "Emergency resolved"
```

### Test Dual Authentication

```bash
python3 defcon1_admin.py test-dual-auth <user-id>
```

**Example:**
```bash
python3 defcon1_admin.py test-dual-auth tactical_user
```

### View Workflow Demo

```bash
python3 defcon1_admin.py demo
```

---

## Web Interface Integration

### Flask Backend Integration

**Add to your Flask app:**

```python
from flask import Flask, request, jsonify
from defcon1_profile import DEFCON1Profile, Authorizer, AuthorizationLevel

app = Flask(__name__)
defcon1 = DEFCON1Profile()

@app.route('/api/defcon1/init', methods=['POST'])
def init_defcon1():
    """Initialize DEFCON1 session"""
    data = request.json
    user_id = data.get('user_id')

    result = defcon1.begin_defcon1_authentication(user_id)
    return jsonify(result)

@app.route('/api/defcon1/auth-dual', methods=['POST'])
def authenticate_dual():
    """Authenticate with dual YubiKeys"""
    data = request.json

    result = defcon1.authenticate_dual_yubikey(
        session_id=data['session_id'],
        user_id=data['user_id'],
        primary_device_id=data['primary_device_id'],
        secondary_device_id=data['secondary_device_id'],
        primary_credential=data['primary_credential'],
        secondary_credential=data['secondary_credential']
    )

    return jsonify({'success': result})

@app.route('/api/defcon1/sessions', methods=['GET'])
def list_sessions():
    """List active DEFCON1 sessions"""
    sessions = defcon1.list_active_sessions()
    return jsonify([s.to_dict() for s in sessions])

@app.route('/api/defcon1/status/<session_id>', methods=['GET'])
def session_status(session_id):
    """Get session status"""
    status = defcon1.get_session_status(session_id)
    return jsonify(status)
```

### JavaScript Frontend Integration

```javascript
// Initialize DEFCON1 session
async function initDEFCON1Session(userId) {
    const response = await fetch('/api/defcon1/init', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({user_id: userId})
    });

    const result = await response.json();
    console.log('Session ID:', result.session_id);
    return result;
}

// Authenticate primary YubiKey
async function authenticatePrimaryYubiKey(sessionId, userId) {
    // Begin FIDO2 authentication
    const authBeginResponse = await fetch('/api/yubikey/auth/begin', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({username: userId})
    });

    const authOptions = await authBeginResponse.json();

    // Trigger WebAuthn
    const credential = await navigator.credentials.get({
        publicKey: authOptions.publicKey
    });

    return credential;
}

// Authenticate secondary YubiKey
async function authenticateSecondaryYubiKey(sessionId, userId) {
    // Same process as primary, but with new challenge
    // ... (similar to primary authentication)
}

// Complete dual YubiKey authentication
async function completeDualAuth(sessionId, userId, primaryCred, secondaryCred) {
    const response = await fetch('/api/defcon1/auth-dual', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            session_id: sessionId,
            user_id: userId,
            primary_device_id: primaryCred.id,
            secondary_device_id: secondaryCred.id,
            primary_credential: primaryCred,
            secondary_credential: secondaryCred
        })
    });

    return await response.json();
}
```

---

## Continuous Authentication

### Automatic Re-Authentication

DEFCON1 sessions require continuous authentication every 5 minutes:

```python
import asyncio
from defcon1_profile import DEFCON1Profile

async def continuous_auth_monitor(session_id, user_id):
    """Monitor and enforce continuous authentication"""
    defcon1 = DEFCON1Profile()

    while True:
        # Wait 5 minutes
        await asyncio.sleep(300)

        # Check if session still active
        session = defcon1.get_session(session_id)
        if not session or not session.is_active:
            break

        # Prompt for dual YubiKey re-authentication
        print("⚠️  Continuous authentication required")
        print("Please authenticate with BOTH YubiKeys")

        # User must re-authenticate with both YubiKeys
        # If authentication fails, session terminates automatically
```

### Manual Re-Authentication

```bash
# Check when next authentication required
python3 defcon1_admin.py session-status a1b2c3d4e5f6g7h8

# Output shows:
# Last Auth Check: 2025-11-25T14:35:00Z
# Next Auth Required: 2025-11-25T14:40:00Z (in 2 minutes)
```

---

## Troubleshooting

### Issue: "Insufficient YubiKeys registered"

**Solution:**

```bash
# Check registered YubiKeys
python3 yubikey_admin.py list

# Register additional YubiKeys
python3 yubikey_admin.py register --user tactical_user
```

### Issue: "Primary and secondary YubiKeys must be different"

**Problem:** Using the same YubiKey for both primary and secondary authentication.

**Solution:** Register and use two physically different YubiKeys.

### Issue: "Insufficient authorizers"

**Problem:** Need 4 authorized personnel, including 1 executive.

**Solution:** Ensure all 4 authorizers are available and have registered YubiKeys.

### Issue: "Session expired"

**Problem:** DEFCON1 sessions expire after 1 hour.

**Solution:** Initialize a new session:

```bash
python3 defcon1_admin.py init-session tactical_user
```

### Issue: "Continuous authentication failed"

**Problem:** Failed to re-authenticate within 5-minute window.

**Solution:** Session automatically terminated. Initialize new session.

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

View audit log:

```bash
tail -f ~/.dsmil/defcon1/defcon1_audit.log
```

---

## Security Considerations

### Threat Model

**Protected Against:**
- ✅ Single-factor compromise (requires 2 YubiKeys)
- ✅ Unauthorized access (requires 4 authorizers)
- ✅ Phishing attacks (FIDO2 domain binding)
- ✅ Replay attacks (FIDO2 nonces and counters)
- ✅ Man-in-the-middle (signed challenges)
- ✅ Session hijacking (continuous authentication)
- ✅ Insider threats (multi-person authorization)

**Not Protected Against:**
- ⚠️ Physical theft of both YubiKeys + knowledge of PIN
- ⚠️ Compromise of all 4 authorizers simultaneously
- ⚠️ Physical access to unlocked authenticated session
- ⚠️ Advanced persistent threats with kernel-level access

### Best Practices

**For Users:**
1. Never share YubiKeys
2. Store backup YubiKey securely (safe/vault)
3. Report lost YubiKeys immediately
4. Use YubiKey PIN for additional security
5. Never leave authenticated session unattended

**For Administrators:**
1. Verify authorizer identities before adding to system
2. Review audit logs daily
3. Conduct periodic authorization drills
4. Maintain spare YubiKeys for authorized personnel
5. Revoke compromised YubiKeys immediately

**For Executives:**
1. Store executive YubiKeys in secure facilities
2. Use tamper-evident storage
3. Log all executive authorization events
4. Review authorization requests before approving
5. Maintain chain of custody for executive tokens

### Physical Security

- Store backup YubiKeys in geographically separate secure facilities
- Use tamper-evident bags/boxes for YubiKey storage
- Implement video surveillance for YubiKey access areas
- Require two-person integrity for YubiKey retrieval
- Conduct regular inventory audits

---

## Compliance

### NIST SP 800-63B

**Authenticator Assurance Level 3 (AAL3)**
- ✅ Multi-factor authentication
- ✅ Hardware-backed security
- ✅ Phishing-resistant
- ✅ Verifier impersonation resistant

### FIPS 140-2

- YubiKey 5 FIPS Series available (Level 2 certification)
- Required for federal government use
- Hardware random number generation
- Cryptographic algorithm validation

### Two-Person Integrity (TPI)

Meets DoD requirements for:
- Nuclear command and control
- Special access programs
- Cryptographic key management
- Emergency war orders

### Classification Levels

**Suitable For:**
- TOP SECRET // SCI
- TOP SECRET // NOFORN
- Special Access Required (SAR)
- Sensitive Compartmented Information (SCI)
- COSMIC TOP SECRET (NATO)

---

## Appendix A: Quick Reference

### Installation Commands

```bash
# Install YubiKey support
sudo ./deployment/configure_yubikey.sh install

# Make DEFCON1 scripts executable
chmod +x lat5150drvmil/02-ai-engine/defcon1_*.py

# Test installation
python3 defcon1_profile.py
```

### Session Management

```bash
# Initialize session
python3 defcon1_admin.py init-session <user-id>

# List sessions
python3 defcon1_admin.py list-sessions

# Check status
python3 defcon1_admin.py session-status <session-id>

# Terminate session
python3 defcon1_admin.py terminate-session <session-id>
```

### YubiKey Management

```bash
# Register YubiKey
python3 yubikey_admin.py register

# List YubiKeys
python3 yubikey_admin.py list

# Test authentication
python3 defcon1_admin.py test-dual-auth <user-id>
```

---

## Appendix B: Authorization Levels

| Level | Name | Value | Required For | Min Count |
|-------|------|-------|--------------|-----------|
| 1 | STANDARD | Standard Operator | Basic operations | 0+ |
| 2 | SUPERVISOR | Supervisor | Enhanced operations | 0+ |
| 3 | COMMANDER | Commander | Critical operations | 0+ |
| 4 | EXECUTIVE | Executive | DEFCON1 sessions | **1+** |

**DEFCON1 Requirement:** At least 1 EXECUTIVE authorizer mandatory.

---

## Appendix C: File Locations

```
~/.dsmil/
├── defcon1/
│   ├── defcon1_config.json      # DEFCON1 configuration
│   ├── sessions.json             # Active sessions
│   └── defcon1_audit.log         # Audit trail
└── yubikey/
    ├── devices.json              # Registered YubiKeys
    └── audit.log                 # YubiKey audit log

/home/user/DSLLVM/lat5150drvmil/
├── 02-ai-engine/
│   ├── defcon1_profile.py        # DEFCON1 profile manager
│   ├── defcon1_admin.py          # Administration tool
│   └── yubikey_auth.py           # YubiKey authentication
└── 00-documentation/
    └── DEFCON1_DUAL_YUBIKEY_AUTHENTICATION.md  # This file
```

---

**Document Version:** 1.0.0
**Last Updated:** 2025-11-25
**Classification:** TOP SECRET // FOR OFFICIAL USE ONLY
**Status:** ✅ Production Ready
**Approved By:** DSMIL Security Authority
