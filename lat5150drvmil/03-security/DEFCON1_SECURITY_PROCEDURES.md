# DEFCON1 Security Procedures - Two-Person Integrity Protocol

**Classification:** TOP SECRET // FOR OFFICIAL USE ONLY
**Version:** 1.0.0
**Date:** 2025-11-25

---

## Critical Security Principle

**⚠️ ALL YUBIKEYS MUST NEVER BE IN THE SAME LOCATION SIMULTANEOUSLY**

This document defines strict security procedures for DEFCON1 authentication to ensure proper Two-Person Integrity (TPI) and prevent unauthorized access if the system is compromised.

---

## Core Security Principles

### 1. Physical Separation of Hardware Tokens

**REQUIREMENT:** YubiKeys must be physically separated to prevent compromise.

```
✅ CORRECT: Keys are separated
  - Primary user's YubiKeys stored in different secure facilities
  - Each authorizer keeps their YubiKey on their person or secure storage
  - After authentication, authorizers REMOVE their key and LEAVE

❌ INCORRECT: Keys together
  - All YubiKeys in the same room
  - YubiKeys left on desk after authentication
  - Authorizers remaining at terminal after authentication
```

### 2. Sequential Authentication

**REQUIREMENT:** Authorizers authenticate ONE AT A TIME, then LEAVE.

**Why?** Ensures that:
- No single person can access multiple authorizers' keys
- Compromise of one person doesn't compromise all authorizers
- Physical presence of all authorizers is verified sequentially
- No collusion opportunity with multiple keys present

---

## Authentication Protocol - Step by Step

### Phase 1: Primary User Dual Authentication

**Location:** Secure Terminal
**Duration:** ~2 minutes

```
[15:00] Primary User arrives at terminal
        ↓
[15:01] Authenticates with PRIMARY YubiKey
        - Inserts PRIMARY YubiKey
        - Touches sensor
        - ✅ Primary authenticated
        - REMOVES PRIMARY YubiKey
        - STORES PRIMARY YubiKey securely (safe, pocket, etc.)
        ↓
[15:02] Authenticates with SECONDARY YubiKey
        - Retrieves SECONDARY YubiKey from separate storage
        - Inserts SECONDARY YubiKey
        - Touches sensor
        - ✅ Secondary authenticated
        - REMOVES SECONDARY YubiKey
        - RETURNS SECONDARY YubiKey to secure storage
        ↓
[15:03] Primary user STEPS AWAY from terminal
        - Both YubiKeys are now in separate secure locations
        - Terminal ready for authorizers
```

**Security Check:**
- ✅ Both YubiKeys authenticated
- ✅ Both YubiKeys removed from terminal
- ✅ YubiKeys stored in different locations
- ✅ No YubiKeys physically present at terminal

---

### Phase 2: Authorizer Chain (Sequential Authentication)

**CRITICAL:** Each authorizer follows this exact sequence:

#### Authorizer #1: Standard Operator

```
[15:03] Authorizer #1 arrives
        ↓
[15:04] Authentication
        - Inserts THEIR personal YubiKey
        - System displays: "Authorizer #1: Authenticate"
        - Touches sensor
        - ✅ Authorizer #1 authenticated
        - Digital signature recorded
        ↓
[15:05] Post-Authentication (CRITICAL)
        - REMOVES their YubiKey from terminal
        - SECURES their YubiKey (pocket, lanyard, secure storage)
        - LEAVES the terminal area
        - DOES NOT return until session complete or emergency
        ↓
[15:06] Authorizer #1 has LEFT
        - Their YubiKey is NO LONGER PRESENT
        - Terminal ready for next authorizer
```

**Security Check:**
- ✅ Authorizer #1 authenticated
- ✅ Authorizer #1's YubiKey removed
- ✅ Authorizer #1 has left the area
- ✅ NO YubiKeys currently at terminal

---

#### Authorizer #2: Supervisor

```
[15:06] Authorizer #2 arrives
        ↓
[15:07] Authentication
        - Inserts THEIR personal YubiKey
        - System displays: "Authorizer #2: Authenticate"
        - Touches sensor
        - ✅ Authorizer #2 authenticated
        - Digital signature recorded
        ↓
[15:08] Post-Authentication (CRITICAL)
        - REMOVES their YubiKey from terminal
        - SECURES their YubiKey
        - LEAVES the terminal area
        - DOES NOT return until session complete or emergency
        ↓
[15:09] Authorizer #2 has LEFT
        - Their YubiKey is NO LONGER PRESENT
        - Terminal ready for next authorizer
```

**Security Check:**
- ✅ Authorizer #2 authenticated
- ✅ Authorizer #2's YubiKey removed
- ✅ Authorizer #2 has left the area
- ✅ NO YubiKeys currently at terminal

---

#### Authorizer #3: Commander

```
[15:09] Authorizer #3 arrives
        ↓
[15:10] Authentication
        - Inserts THEIR personal YubiKey
        - System displays: "Authorizer #3: Authenticate"
        - Touches sensor
        - ✅ Authorizer #3 authenticated
        - Digital signature recorded
        ↓
[15:11] Post-Authentication (CRITICAL)
        - REMOVES their YubiKey from terminal
        - SECURES their YubiKey
        - LEAVES the terminal area
        - DOES NOT return until session complete or emergency
        ↓
[15:12] Authorizer #3 has LEFT
        - Their YubiKey is NO LONGER PRESENT
        - Terminal ready for executive authorizer
```

**Security Check:**
- ✅ Authorizer #3 authenticated
- ✅ Authorizer #3's YubiKey removed
- ✅ Authorizer #3 has left the area
- ✅ NO YubiKeys currently at terminal

---

#### Authorizer #4: EXECUTIVE (REQUIRED)

```
[15:12] EXECUTIVE arrives
        - Highest level authorization
        - Presidential/SECDEF/Executive level
        ↓
[15:13] Authentication
        - Inserts THEIR personal YubiKey
        - System displays: "EXECUTIVE: Authenticate"
        - Touches sensor
        - ✅ EXECUTIVE authenticated
        - Digital signature recorded
        - System logs: "EXECUTIVE AUTHORIZATION GRANTED"
        ↓
[15:14] Post-Authentication (CRITICAL)
        - REMOVES their YubiKey from terminal
        - SECURES their YubiKey in executive secure storage
        - LEAVES the terminal area
        - Returns to command post
        ↓
[15:15] EXECUTIVE has LEFT
        - Their YubiKey is NO LONGER PRESENT
        - Executive YubiKey secured in separate facility
```

**Security Check:**
- ✅ Executive authenticated
- ✅ Executive's YubiKey removed
- ✅ Executive has left the area
- ✅ Executive YubiKey in separate secure facility
- ✅ NO YubiKeys currently at terminal

---

### Phase 3: Session Activation

```
[15:15] System Validates All Requirements
        ✓ Primary user: 2 YubiKeys authenticated
        ✓ Authorizer #1: Authenticated and departed
        ✓ Authorizer #2: Authenticated and departed
        ✓ Authorizer #3: Authenticated and departed
        ✓ Authorizer #4 (EXECUTIVE): Authenticated and departed
        ✓ All YubiKeys removed from terminal
        ✓ All YubiKeys in separate secure locations
        ↓
[15:16] 🔓 DEFCON1 SESSION ACTIVATED
        - Duration: 1 hour
        - Access: EMERGENCY ONLY
        - Primary user returns to terminal
        - Session active until 16:16
```

---

## YubiKey Storage Requirements

### Primary User YubiKeys

**PRIMARY YubiKey:**
- **Storage:** On person (pocket, lanyard) or immediate secure storage
- **Access:** Quick access for continuous authentication
- **Location:** With primary user at all times during session

**SECONDARY YubiKey:**
- **Storage:** SEPARATE secure facility (different safe, vault, location)
- **Access:** Retrieved only for authentication
- **Location:** NOT with primary user during normal operations

**Separation Distance:** Minimum 50 feet or different secure zones

### Authorizer YubiKeys

**Each authorizer's YubiKey:**
- **Storage:** On authorizer's person (pocket, lanyard, personal safe)
- **Access:** Authorizer controls access at all times
- **Location:** With authorizer, separate from all other keys

**After Authentication:**
- Authorizer REMOVES key
- Authorizer SECURES key on person
- Authorizer LEAVES terminal area
- Key is NOT LEFT BEHIND

### Executive YubiKey

**Special Handling:**
- **Storage:** Executive secure facility (separate building if possible)
- **Access:** Executive-only access or two-person controlled vault
- **Transport:** Executive carries personally or secure courier
- **After Use:** Returned immediately to executive secure storage

---

## Continuous Authentication Protocol

**Every 5 minutes during session:**

```
[15:21] Continuous Auth Required (5 min elapsed)
        ↓
        Primary user at terminal
        ↓
[15:22] PRIMARY YubiKey Authentication
        - Retrieves PRIMARY YubiKey
        - Inserts and touches sensor
        - ✅ Primary authenticated
        - REMOVES PRIMARY YubiKey
        - SECURES PRIMARY YubiKey
        ↓
[15:23] SECONDARY YubiKey Authentication
        - Retrieves SECONDARY YubiKey from secure storage
        - Inserts and touches sensor
        - ✅ Secondary authenticated
        - REMOVES SECONDARY YubiKey
        - RETURNS to secure storage
        ↓
[15:24] Continuous Auth Complete
        - Session continues
        - Both keys secured separately
        - Next auth at 15:29
```

**Security Check:**
- ✅ Both keys authenticated
- ✅ Both keys immediately removed after use
- ✅ Keys returned to separate secure storage
- ✅ No keys left at terminal

---

## Security Violations - Examples

### ❌ VIOLATION: Multiple YubiKeys Present

**Scenario:**
```
Authorizer #1 authenticates, leaves YubiKey on desk
Authorizer #2 arrives and authenticates
→ TWO YubiKeys now present simultaneously
```

**Risk:** Single adversary could compromise both keys

**Correct Procedure:**
```
Authorizer #1 authenticates → REMOVES key → LEAVES
[Terminal clear]
Authorizer #2 arrives → Authenticates → REMOVES key → LEAVES
```

---

### ❌ VIOLATION: Authorizers Remaining at Terminal

**Scenario:**
```
Authorizer #1 authenticates, stays to "observe"
Authorizer #2 authenticates, stays to "observe"
Authorizer #3 authenticates, stays to "observe"
→ ALL AUTHORIZERS present with their keys
```

**Risk:** Physical attack or coercion could compromise all keys

**Correct Procedure:**
```
Each authorizer:
  1. Authenticate
  2. REMOVE key
  3. IMMEDIATELY LEAVE area
  4. Do not return until session complete
```

---

### ❌ VIOLATION: YubiKeys Stored Together

**Scenario:**
```
Primary user stores both YubiKeys in same desk drawer
→ Single point of compromise
```

**Risk:** Theft of drawer = both YubiKeys lost

**Correct Procedure:**
```
PRIMARY YubiKey: On person or immediate safe
SECONDARY YubiKey: Separate secure facility (different safe, different room)
→ Compromise of one location does NOT expose both keys
```

---

## Emergency Procedures

### Emergency Session Termination

**If security breach detected:**

```
1. Immediate Actions:
   - Sound alarm
   - Terminate DEFCON1 session immediately
   - Lock down all terminals
   - Secure all YubiKeys

2. Authorizer Recall:
   - Contact all authorizers
   - Verify YubiKey possession
   - Report any missing keys

3. Investigation:
   - Review audit logs
   - Check video surveillance
   - Interview all participants
   - Assess compromise scope

4. Recovery:
   - Revoke compromised keys
   - Issue new YubiKeys
   - Re-register all participants
   - Update authorization records
```

### Lost or Stolen YubiKey

**Immediate Response:**

```
1. Report immediately to security officer
2. Revoke lost key in system:
   python3 yubikey_admin.py revoke <device-id> --reason "Lost/Stolen"
3. Issue replacement YubiKey
4. Register new YubiKey
5. Update authorization records
6. Review audit logs for unauthorized use
```

---

## Compliance and Audit

### Required Logs

**Every DEFCON1 session must log:**

- Timestamp of each authentication
- User ID and device ID for each YubiKey
- Physical presence confirmation
- Departure confirmation for each authorizer
- Continuous authentication events
- Session termination reason
- All YubiKey removals and insertions

### Audit Review

**Daily:**
- Review all DEFCON1 sessions
- Verify proper authentication sequence
- Confirm authorizer departure
- Check for anomalies

**Weekly:**
- Review video surveillance (if available)
- Interview random participants
- Verify YubiKey physical security
- Test emergency procedures

**Monthly:**
- Comprehensive security audit
- YubiKey inventory check
- Authorization level review
- Procedure compliance assessment

---

## Training Requirements

### All Personnel

- Two-Person Integrity principles
- Physical security procedures
- YubiKey handling and storage
- Emergency response procedures
- Reporting requirements

### Authorizers

- Authorization responsibilities
- Authentication sequence
- Post-authentication procedures (REMOVAL and DEPARTURE)
- Security violation identification
- Emergency termination procedures

### Executives

- Executive authorization authority
- YubiKey security requirements
- Chain of command procedures
- Emergency override protocols

---

## Quick Reference Card

**Print and post at DEFCON1 terminals:**

```
╔════════════════════════════════════════════════════════════╗
║           DEFCON1 AUTHENTICATION PROCEDURES                ║
╚════════════════════════════════════════════════════════════╝

PRIMARY USER:
  1. Authenticate PRIMARY YubiKey → REMOVE → SECURE
  2. Authenticate SECONDARY YubiKey → REMOVE → SECURE
  3. STEP AWAY from terminal

EACH AUTHORIZER (one at a time):
  1. ARRIVE at terminal
  2. AUTHENTICATE with personal YubiKey
  3. REMOVE YubiKey from terminal
  4. SECURE YubiKey on person
  5. LEAVE terminal area immediately
  6. DO NOT RETURN until session complete

⚠️  CRITICAL: NEVER leave YubiKey at terminal
⚠️  CRITICAL: ALWAYS leave area after authenticating
⚠️  CRITICAL: ALL YubiKeys must be physically separated

CONTINUOUS AUTH (every 5 minutes):
  - Authenticate PRIMARY → REMOVE → SECURE
  - Authenticate SECONDARY → REMOVE → SECURE

EMERGENCY: Press RED BUTTON to terminate session
```

---

## Security Checklist

**Before each DEFCON1 session:**

- [ ] All YubiKeys inventoried and accounted for
- [ ] All authorizers briefed on procedures
- [ ] Secure storage locations verified
- [ ] Video surveillance active (if available)
- [ ] Emergency procedures reviewed
- [ ] Audit logging enabled and tested

**During DEFCON1 session:**

- [ ] Each authorizer removes YubiKey after auth
- [ ] Each authorizer leaves terminal area after auth
- [ ] No YubiKeys left at terminal
- [ ] Continuous auth performed on schedule
- [ ] All events logged in real-time

**After DEFCON1 session:**

- [ ] Session terminated properly
- [ ] All YubiKeys returned to secure storage
- [ ] Audit log reviewed
- [ ] Incident report filed (if applicable)
- [ ] Video surveillance reviewed (if available)
- [ ] Authorizers debriefed

---

**Document Version:** 1.0.0
**Last Updated:** 2025-11-25
**Classification:** TOP SECRET // FOR OFFICIAL USE ONLY
**Approved By:** DSMIL Security Authority

**CRITICAL REMINDER:**
## ALL YUBIKEYS MUST NEVER BE TOGETHER AT ANY TIME
