# DEFCON1 Quick Start Guide

**Classification:** TOP SECRET // FOR OFFICIAL USE ONLY

---

## Who Needs YubiKeys?

### PRIMARY USER (Person accessing the system)
**Needs: 2 YubiKeys**

```
┌──────────────────────────────┐
│ PRIMARY USER: "tactical_user"│
├──────────────────────────────┤
│ YubiKey #1: PRIMARY          │  ← Everyday use YubiKey
│ YubiKey #2: SECONDARY        │  ← Backup YubiKey
└──────────────────────────────┘
```

**Both YubiKeys must authenticate to start the session**

---

### AUTHORIZERS (4 separate people approving access)
**Each needs: 1 YubiKey**

```
┌──────────────────────────────┐
│ AUTHORIZER #1: "john_doe"    │
├──────────────────────────────┤
│ Role: Operator               │
│ Level: STANDARD              │
│ YubiKey: John's personal key │
└──────────────────────────────┘

┌──────────────────────────────┐
│ AUTHORIZER #2: "jane_smith"  │
├──────────────────────────────┤
│ Role: Supervisor             │
│ Level: SUPERVISOR            │
│ YubiKey: Jane's personal key │
└──────────────────────────────┘

┌──────────────────────────────┐
│ AUTHORIZER #3: "bob_jones"   │
├──────────────────────────────┤
│ Role: Commander              │
│ Level: COMMANDER             │
│ YubiKey: Bob's personal key  │
└──────────────────────────────┘

┌──────────────────────────────┐
│ AUTHORIZER #4: "alice_exec"  │  ⚠️ EXECUTIVE REQUIRED
├──────────────────────────────┤
│ Role: Executive/SECDEF       │
│ Level: EXECUTIVE             │
│ YubiKey: Alice's personal key│
└──────────────────────────────┘
```

**Each authorizer authenticates with THEIR OWN YubiKey**

---

## TOTAL YubiKeys Needed: 6

- **2** for primary user (dual auth)
- **1** for Authorizer #1
- **1** for Authorizer #2
- **1** for Authorizer #3
- **1** for Authorizer #4 (Executive)

---

## Step-by-Step: How Authorizers Authenticate

### Scenario: User "tactical_user" needs DEFCON1 access

#### PHASE 1: Primary User Dual Authentication

**Step 1: Primary YubiKey**
```
tactical_user:
  1. Inserts PRIMARY YubiKey (their first key)
  2. Browser prompts: "Touch your security key"
  3. Touches YubiKey sensor
  4. ✅ Primary YubiKey authenticated
```

**Step 2: Secondary YubiKey**
```
tactical_user:
  1. Removes primary YubiKey
  2. Inserts SECONDARY YubiKey (their second key)
  3. Browser prompts: "Touch your security key"
  4. Touches YubiKey sensor
  5. ✅ Secondary YubiKey authenticated
```

**Result:** User has proven they possess BOTH hardware tokens

---

#### PHASE 2: Authorizer Chain (4 different people)

**Authorizer #1: John Doe (Operator)**
```
John:
  1. Walks to the terminal
  2. Inserts HIS personal YubiKey
  3. Browser prompts: "Touch your security key"
  4. Touches HIS YubiKey sensor
  5. System records: "John Doe authorized"
  6. ✅ Authorizer #1 approved
  7. Removes his YubiKey and steps away
```

**Authorizer #2: Jane Smith (Supervisor)**
```
Jane:
  1. Walks to the terminal
  2. Inserts HER personal YubiKey
  3. Browser prompts: "Touch your security key"
  4. Touches HER YubiKey sensor
  5. System records: "Jane Smith authorized"
  6. ✅ Authorizer #2 approved
  7. Removes her YubiKey and steps away
```

**Authorizer #3: Bob Jones (Commander)**
```
Bob:
  1. Walks to the terminal
  2. Inserts HIS personal YubiKey
  3. Browser prompts: "Touch your security key"
  4. Touches HIS YubiKey sensor
  5. System records: "Bob Jones authorized"
  6. ✅ Authorizer #3 approved
  7. Removes his YubiKey and steps away
```

**Authorizer #4: Alice Executive (Executive/SECDEF)**
```
Alice:
  1. Walks to the terminal
  2. Inserts HER personal YubiKey
  3. Browser prompts: "Touch your security key"
  4. Touches HER YubiKey sensor
  5. System records: "Alice Executive authorized (EXECUTIVE LEVEL)"
  6. ✅ EXECUTIVE authorization granted
  7. Removes her YubiKey and steps away
```

---

#### PHASE 3: Session Activated

```
System validates:
  ✓ tactical_user authenticated with 2 YubiKeys
  ✓ John Doe authorized (Operator)
  ✓ Jane Smith authorized (Supervisor)
  ✓ Bob Jones authorized (Commander)
  ✓ Alice Executive authorized (EXECUTIVE) ⚠️ REQUIRED

🔓 DEFCON1 SESSION ACTIVATED
   Duration: 1 hour
   Access: EMERGENCY ONLY
```

---

## Registration Workflow

### Primary User Registration (tactical_user)

```bash
# Terminal 1: Start registration guide
cd /home/user/DSLLVM/lat5150drvmil/02-ai-engine
python3 defcon1_registration_guide.py
```

The guide will walk you through:

1. **Register Primary YubiKey**
   - Insert first YubiKey
   - Open: http://localhost:5001/tactical_yubikey_ui.html
   - Click "REGISTER NEW KEY"
   - Name: "Primary YubiKey"
   - Touch sensor when prompted

2. **Register Secondary YubiKey**
   - REMOVE first YubiKey
   - Insert second YubiKey
   - Click "REGISTER NEW KEY"
   - Name: "Secondary YubiKey"
   - Touch sensor when prompted

### Authorizer Registration

**Each authorizer does this individually:**

```bash
# Example: Authorizer #1 (John Doe)
python3 yubikey_admin.py register --name "John Doe - Operator" --user john_doe
```

**Or via web interface:**
1. John inserts HIS YubiKey
2. Opens: http://localhost:5001/tactical_yubikey_ui.html
3. Clicks "REGISTER NEW KEY"
4. Enters: "John Doe - Operator"
5. Touches HIS YubiKey sensor

**Repeat for all 4 authorizers**

---

## Quick Commands

### Check Registration Status
```bash
# List all registered YubiKeys
python3 yubikey_admin.py list

# Output shows:
# Device 1: Primary YubiKey (tactical_user)
# Device 2: Secondary YubiKey (tactical_user)
# Device 3: John Doe - Operator (john_doe)
# Device 4: Jane Smith - Supervisor (jane_smith)
# Device 5: Bob Jones - Commander (bob_jones)
# Device 6: Alice Executive - Executive (alice_exec)
```

### Test Dual Authentication
```bash
# Test that tactical_user has 2 YubiKeys registered
python3 defcon1_admin.py test-dual-auth tactical_user
```

### Initialize DEFCON1 Session
```bash
# Start a new DEFCON1 session
python3 defcon1_admin.py init-session tactical_user

# Output:
# Session ID: abc123def456
# Requirements:
#   - YubiKeys: 2
#   - Authorizers: 4
#   - Executive Authorizers: 1
```

### Monitor Sessions
```bash
# List active sessions
python3 defcon1_admin.py list-sessions

# Check specific session
python3 defcon1_admin.py session-status abc123def456
```

---

## Web Interface Integration

### Authentication UI Flow

```javascript
// 1. User initiates DEFCON1 session
const session = await initDEFCON1Session('tactical_user');
console.log('Session ID:', session.session_id);

// 2. User authenticates with PRIMARY YubiKey
const primaryCred = await navigator.credentials.get({
    publicKey: primaryAuthOptions
});
console.log('✅ Primary YubiKey authenticated');

// 3. User authenticates with SECONDARY YubiKey
const secondaryCred = await navigator.credentials.get({
    publicKey: secondaryAuthOptions
});
console.log('✅ Secondary YubiKey authenticated');

// 4. Authorizer #1 authenticates
const auth1Cred = await navigator.credentials.get({
    publicKey: auth1Options
});
console.log('✅ Authorizer #1 approved: John Doe');

// 5. Authorizer #2 authenticates
const auth2Cred = await navigator.credentials.get({
    publicKey: auth2Options
});
console.log('✅ Authorizer #2 approved: Jane Smith');

// 6. Authorizer #3 authenticates
const auth3Cred = await navigator.credentials.get({
    publicKey: auth3Options
});
console.log('✅ Authorizer #3 approved: Bob Jones');

// 7. Authorizer #4 (EXECUTIVE) authenticates
const auth4Cred = await navigator.credentials.get({
    publicKey: auth4Options
});
console.log('✅ Authorizer #4 approved: Alice Executive (EXECUTIVE)');

// 8. Create session
const result = await createDEFCON1Session(session.session_id, {
    primary_yubikey: primaryCred,
    secondary_yubikey: secondaryCred,
    authorizers: [auth1Cred, auth2Cred, auth3Cred, auth4Cred]
});

console.log('🔓 DEFCON1 SESSION ACTIVATED');
```

---

## Continuous Authentication

**Every 5 minutes during the session:**

```
System: "⚠️  Continuous authentication required"

tactical_user:
  1. Inserts PRIMARY YubiKey → Touch sensor → ✓
  2. Inserts SECONDARY YubiKey → Touch sensor → ✓

System: "✅ Session continues"
```

**If authentication fails:**
```
System: "❌ Authentication failed - SESSION TERMINATED"
```

---

## Common Questions

### Q: Can I use the same YubiKey for primary and secondary?
**A: NO.** They must be physically different YubiKeys.

### Q: Can authorizers use the same YubiKey?
**A: NO.** Each authorizer must have their own unique YubiKey.

### Q: Do authorizers need to register their YubiKeys?
**A: YES.** Each authorizer must register their personal YubiKey before they can authorize sessions.

### Q: What if an authorizer is not available?
**A: The DEFCON1 session cannot be activated.** All 4 authorizers (including 1 executive) must be present and authenticate.

### Q: Can I reuse authorizers from a previous session?
**A: NO.** Each DEFCON1 session requires fresh authentication from all authorizers.

### Q: What happens if I lose a YubiKey?
**A:**
- **Primary user:** Revoke lost key immediately, use backup/secondary
- **Authorizer:** Revoke lost key, register new key, update authorization records

### Q: How do I revoke a YubiKey?
```bash
python3 yubikey_admin.py revoke <device-id> --reason "Lost/Stolen"
```

---

## Security Reminders

- ⚠️ **NEVER share YubiKeys**
- ⚠️ **Store backup YubiKeys in secure facilities**
- ⚠️ **Report lost YubiKeys immediately**
- ⚠️ **Executive YubiKeys require special handling**
- ⚠️ **All authentications are logged and auditable**
- ⚠️ **Sessions expire after 1 hour**

---

## Complete Example

**Scenario:** Emergency operation requires DEFCON1 access

```
15:00 - tactical_user initiates DEFCON1 session
        → Session ID: abc123def456

15:01 - tactical_user authenticates PRIMARY YubiKey ✓
15:02 - tactical_user authenticates SECONDARY YubiKey ✓

15:03 - John Doe arrives, authenticates (Operator) ✓
15:04 - Jane Smith arrives, authenticates (Supervisor) ✓
15:05 - Bob Jones arrives, authenticates (Commander) ✓
15:06 - Alice Executive arrives, authenticates (EXECUTIVE) ✓

15:07 - System validates all requirements ✓
        🔓 DEFCON1 SESSION ACTIVATED

15:12 - Continuous auth prompt (5 minutes elapsed)
        → tactical_user re-authenticates both YubiKeys ✓

15:17 - Continuous auth prompt
        → tactical_user re-authenticates both YubiKeys ✓

[... continues every 5 minutes ...]

16:07 - Session expires (1 hour elapsed)
        🔒 SESSION TERMINATED
        → Full audit trail preserved
```

---

**Document Version:** 1.0.0
**Last Updated:** 2025-11-25
**Classification:** TOP SECRET // FOR OFFICIAL USE ONLY
