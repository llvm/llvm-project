# DSMIL Yubikey Integration Guide

**Version:** 2.1.0
**Date:** 2025-11-13
**Classification:** UNCLASSIFIED // FOR OFFICIAL USE ONLY
**Status:** Production Ready

---

## Table of Contents

1. [Overview](#overview)
2. [Hardware Requirements](#hardware-requirements)
3. [Installation](#installation)
4. [Configuration](#configuration)
5. [Device Registration](#device-registration)
6. [Authentication](#authentication)
7. [Tactical UI Integration](#tactical-ui-integration)
8. [Administration](#administration)
9. [Troubleshooting](#troubleshooting)
10. [Security Considerations](#security-considerations)
11. [API Reference](#api-reference)
12. [Best Practices](#best-practices)

---

## Overview

The DSMIL platform integrates Yubikey hardware authentication tokens for phishing-resistant, hardware-backed security. This guide covers complete setup, configuration, and usage of Yubikey authentication.

### Features

- **FIDO2/WebAuthn** - Modern, phishing-resistant web authentication
- **Challenge-Response** - Offline authentication without internet connectivity
- **Yubico OTP** - One-time password support (requires API credentials)
- **Multi-Device Support** - Up to 5 Yubikeys per user
- **Device Management** - Active, suspended, and revoked status tracking
- **Audit Logging** - Comprehensive operation logging
- **Browser Integration** - Native WebAuthn support in tactical UI

### Authentication Methods

| Method | Use Case | Online Required | Touch Required | Phishing-Resistant |
|--------|----------|----------------|----------------|-------------------|
| **FIDO2/WebAuthn** | Web interface | Yes | Yes | ✅ Yes |
| **Challenge-Response** | SSH, offline ops | No | Yes | ✅ Yes |
| **Yubico OTP** | Fallback | Yes | Yes | ⚠️ Partial |

### Architecture

```
┌──────────────────────────────────────────────────────┐
│ DSMIL Tactical UI (Browser)                         │
│ - WebAuthn JavaScript API                           │
│ - Touch prompts and status                          │
└──────────────────┬───────────────────────────────────┘
                   │
                   │ HTTPS/WebAuthn
                   ▼
┌──────────────────────────────────────────────────────┐
│ Backend (Python/Flask)                               │
│ - yubikey_auth.py (Authentication module)           │
│ - FIDO2 server (python-fido2)                       │
│ - Device management                                  │
│ - Audit logging                                      │
└──────────────────┬───────────────────────────────────┘
                   │
                   │ USB/NFC
                   ▼
┌──────────────────────────────────────────────────────┐
│ Yubikey Hardware Token                              │
│ - Private key storage (never leaves device)         │
│ - Touch sensor                                       │
│ - Challenge signing                                  │
└──────────────────────────────────────────────────────┘
```

---

## Hardware Requirements

### Supported Yubikey Models

**YubiKey 5 Series** (Recommended)
- YubiKey 5 NFC
- YubiKey 5 Nano
- YubiKey 5C
- YubiKey 5C Nano
- YubiKey 5C NFC
- YubiKey 5Ci

**YubiKey 4 Series**
- YubiKey 4
- YubiKey 4 Nano
- YubiKey 4C
- YubiKey 4C Nano

**Security Key Series**
- Security Key NFC
- Security Key C NFC
- Security Key (Blue)

### Features by Model

| Model | FIDO2 | Challenge-Response | OTP | NFC |
|-------|-------|-------------------|-----|-----|
| YubiKey 5 NFC | ✅ | ✅ | ✅ | ✅ |
| YubiKey 5 Nano | ✅ | ✅ | ✅ | ❌ |
| YubiKey 5C | ✅ | ✅ | ✅ | ❌ |
| YubiKey 5C NFC | ✅ | ✅ | ✅ | ✅ |
| YubiKey 4 | ✅ | ✅ | ✅ | ❌ |
| Security Key | ✅ | ❌ | ❌ | Varies |

### System Requirements

**Operating System:**
- Linux kernel 4.x+ (Ubuntu 20.04+, Debian 10+, RHEL 8+, CentOS 8+)
- USB port or NFC reader (for NFC-enabled keys)

**Browser (for WebAuthn):**
- Chrome 90+
- Firefox 88+
- Edge 90+
- Safari 14+

**Software:**
- Python 3.8+
- libfido2
- yubikey-manager
- pcscd (PC/SC smart card daemon)

---

## Installation

### Quick Install (Automated)

```bash
# Clone repository (if not already cloned)
cd /home/user/LAT5150DRVMIL

# Run automated installation script
sudo ./deployment/configure_yubikey.sh install
```

**The script will:**
1. Detect your package manager (apt/dnf/yum/pacman)
2. Install required packages (libfido2, yubikey-manager, pcscd)
3. Install Python dependencies (fido2, yubikey-manager, yubico-client)
4. Configure udev rules for device access
5. Start and enable pcscd daemon
6. Create plugdev group and add current user
7. Test Yubikey detection

**Output:**
```
╔════════════════════════════════════════════════════════════════╗
║ DSMIL Yubikey Configuration                                    ║
╚════════════════════════════════════════════════════════════════╝

[✓] Using apt package manager
[✓] Packages installed
[✓] Python packages installed
[✓] udev rules configured
[✓] pcscd enabled
[✓] pcscd started
[✓] plugdev group exists
[✓] User tactical added to plugdev group
[!] User needs to log out and back in for group changes to take effect

Installation Complete

Next steps:
  1. Log out and back in (for group changes)
  2. Insert your Yubikey
  3. Test: ./configure_yubikey.sh test
  4. Register: python3 02-ai-engine/yubikey_admin.py register
```

### Manual Installation

If you prefer manual installation or the automated script fails:

**Step 1: Install System Packages**

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install -y \
    libfido2-1 \
    libfido2-dev \
    python3-fido2 \
    yubikey-manager \
    yubikey-personalization \
    libu2f-udev \
    pcscd \
    pcsc-tools \
    libpcsclite-dev \
    python3-pip
```

**RHEL/CentOS/Fedora:**
```bash
sudo dnf install -y \
    libfido2 \
    libfido2-devel \
    python3-fido2 \
    yubikey-manager \
    yubikey-personalization \
    pcsc-lite \
    pcsc-tools \
    pcsc-lite-devel \
    python3-pip
```

**Arch Linux:**
```bash
sudo pacman -Sy --noconfirm \
    libfido2 \
    python-fido2 \
    yubikey-manager \
    yubikey-personalization \
    pcsclite \
    ccid \
    python-pip
```

**Step 2: Install Python Packages**

```bash
pip3 install --upgrade fido2 yubikey-manager yubico-client
```

**Step 3: Configure udev Rules**

```bash
# Create udev rules file
sudo tee /etc/udev/rules.d/70-u2f.rules << 'EOF'
# Yubico YubiKey
KERNEL=="hidraw*", SUBSYSTEM=="hidraw", ATTRS{idVendor}=="1050", TAG+="uaccess", GROUP="plugdev", MODE="0660"

# Yubico Security Key
KERNEL=="hidraw*", SUBSYSTEM=="hidraw", ATTRS{idVendor}=="1050", ATTRS{idProduct}=="0407", TAG+="uaccess", GROUP="plugdev", MODE="0660"

# FIDO U2F devices
KERNEL=="hidraw*", SUBSYSTEM=="hidraw", ATTRS{idVendor}=="1050", MODE="0664", GROUP="plugdev"
EOF

# Reload udev rules
sudo udevadm control --reload-rules
sudo udevadm trigger
```

**Step 4: Configure pcscd**

```bash
# Enable and start pcscd
sudo systemctl enable pcscd
sudo systemctl start pcscd

# Verify status
sudo systemctl status pcscd
```

**Step 5: Create plugdev Group**

```bash
# Create group (if doesn't exist)
sudo groupadd plugdev

# Add current user
sudo usermod -a -G plugdev $USER

# Log out and back in for changes to take effect
```

---

## Configuration

### Test Installation

After installation and logging back in:

```bash
# Test Yubikey detection
./deployment/configure_yubikey.sh test
```

**Expected Output:**
```
╔════════════════════════════════════════════════════════════════╗
║ Testing Yubikey Detection                                      ║
╚════════════════════════════════════════════════════════════════╝

Running: ykman list

YubiKey 5 NFC (5.4.3) [USB] Serial: 12345678

[✓] Yubikey detected successfully
[✓] FIDO2 Python module working
[✓] pcscd service running
```

### Check System Status

```bash
./deployment/configure_yubikey.sh status
```

**Output:**
```
╔════════════════════════════════════════════════════════════════╗
║ Yubikey Configuration Status                                   ║
╚════════════════════════════════════════════════════════════════╝

Package Status:
[✓] yubikey-manager: 5.0.0
[✓] python3-fido2: 1.1.0
[✓] pcsc-tools: Installed

Service Status:
[✓] pcscd: Running

Configuration Status:
[✓] udev rules: Configured
[✓] User tactical: In plugdev group

Yubikey Detection:
[✓] Yubikey detected
YubiKey 5 NFC (5.4.3) [USB] Serial: 12345678
```

### Verify Python Module

```bash
python3 02-ai-engine/yubikey_auth.py
```

**Output:**
```
DSMIL Yubikey Authentication Module
============================================================

Dependency Check:
  FIDO2/WebAuthn: ✓ Available
  Yubico OTP:     ✓ Available
  YubiKey Manager:✓ Available

Initializing YubikeyAuth...

Registered Devices:
  Total:     0
  Active:    0
  Suspended: 0
  Revoked:   0

No devices registered.

To register a Yubikey:
  1. Use the admin tool: python yubikey_admin.py register
  2. Or integrate with your application

============================================================
```

---

## Device Registration

### Register Your First Yubikey

**Method 1: Command Line**

```bash
python3 02-ai-engine/yubikey_admin.py register --name "Yubikey 5 NFC" --user tactical
```

**Interactive Prompts:**
```
Yubikey Registration
================================================================================
Device name (e.g., 'Yubikey 5 NFC'): Yubikey 5 NFC
Username: tactical

Registering: Yubikey 5 NFC
User:        tactical

Method:      FIDO2/WebAuthn

Note: FIDO2 registration requires web browser interaction.
      Use the tactical UI to complete registration.
      Or use challenge-response for offline authentication.

Continue with FIDO2 registration? [y/N]: y

Starting FIDO2 registration...

✓ Registration initiated

Next steps:
  1. Open tactical UI in browser
  2. Navigate to Settings > Yubikey Registration
  3. Follow the on-screen instructions
  4. Touch your Yubikey when prompted

Registration data saved for user: tactical
```

**Method 2: Tactical UI (Recommended)**

1. Open tactical UI in browser:
   ```bash
   firefox http://localhost:5001/tactical_yubikey_ui.html
   # Or
   firefox 03-web-interface/tactical_yubikey_ui.html
   ```

2. Click **"⚿ REGISTER NEW KEY"** button in Mission Control panel

3. Enter device name when prompted:
   ```
   Enter a name for this Yubikey (e.g., "Yubikey 5 NFC"):
   > Yubikey 5 NFC
   ```

4. Browser shows WebAuthn prompt:
   ```
   ┌─────────────────────────────────────┐
   │ localhost wants to create           │
   │ a passkey for tactical_user         │
   │                                     │
   │ Insert and touch your security key  │
   │                                     │
   │        [Cancel]  [Allow]            │
   └─────────────────────────────────────┘
   ```

5. **Insert Yubikey** and **touch the sensor**

6. Success message appears:
   ```
   SYSTEM: Yubikey "Yubikey 5 NFC" registered successfully!
   Device ID: a3f4b5c2d1e6f7g8
   ```

7. Device now appears in Yubikey Settings

### Register Multiple Devices

You can register up to 5 Yubikeys per user for redundancy:

```bash
# Register backup Yubikey
python3 02-ai-engine/yubikey_admin.py register \
    --name "Yubikey Backup" \
    --user tactical
```

**Use Cases for Multiple Keys:**
- Primary key (everyday use)
- Backup key (secure storage)
- Travel key (portable)
- Office key (desk drawer)
- Emergency key (safe deposit)

---

## Authentication

### Web Interface Authentication

**Automatic Prompt:**

When you open the tactical UI, it automatically prompts for authentication:

```
┌──────────────────────────────────────────────┐
│              🔑                              │
│   YUBIKEY AUTHENTICATION REQUIRED           │
│                                              │
│ Insert your Yubikey and touch the sensor   │
│ to authenticate.                            │
│                                              │
│     [AUTHENTICATE]  [CANCEL]                │
└──────────────────────────────────────────────┘
```

**Steps:**
1. Click **[AUTHENTICATE]** button
2. Browser shows WebAuthn prompt
3. Insert Yubikey (if not already inserted)
4. Touch the sensor when it flashes
5. Status updates to "YUBIKEY VERIFIED"
6. All operations unlocked

**Session Duration:**
- Lasts for browser session
- Cleared on tab close
- Cleared on browser restart
- Cleared on logout

### Manual Authentication Test

```bash
python3 02-ai-engine/yubikey_admin.py test
```

**Output:**
```
Yubikey Authentication Test
============================================================

Testing 1 device(s)...

Device: Yubikey 5 NFC (a3f4b5c2d1e6f7g8)
  Testing Challenge-Response...
  ✓ Challenge-Response authentication successful
```

### Challenge-Response (Offline Authentication)

For SSH or offline operations, use challenge-response:

**Setup Challenge-Response:**

```bash
# Configure slot 2 for challenge-response
python3 -c "
from yubikey_auth import YubikeyAuth
auth = YubikeyAuth()
auth.setup_challenge_response('a3f4b5c2d1e6f7g8', slot=2)
"
```

**Authenticate with Challenge-Response:**

```python
from yubikey_auth import YubikeyAuth

auth = YubikeyAuth()
success = auth.authenticate_challenge_response('a3f4b5c2d1e6f7g8')

if success:
    print("✓ Authenticated")
else:
    print("✗ Authentication failed")
```

**Use Case - SSH Access:**
```bash
# PAM integration (future feature)
# Add to /etc/pam.d/sshd:
auth required pam_yubico.so mode=challenge-response
```

---

## Tactical UI Integration

### UI Features

The tactical UI (`tactical_yubikey_ui.html`) includes:

**Mission Control Panel (Left Sidebar):**
- Display mode selector (5 TEMPEST modes)
- Yubikey settings access
- Register new Yubikey button
- Test authentication button
- System controls

**Status Panel (Right Sidebar):**
- Security status indicator
- Yubikey status indicator
- Operations counter
- TEMPEST mode display
- EM reduction percentage

**Yubikey Settings Modal:**
- Registered devices list
- Device status (active/suspended/revoked)
- Device information (ID, credentials, last used)
- Authentication log (last 50 events)

### User Workflow

**First-Time Setup:**
```
1. Open tactical UI
2. See message: "No Yubikey devices registered"
3. Click "⚿ REGISTER NEW KEY"
4. Follow registration wizard
5. Touch Yubikey when prompted
6. Registration complete
7. UI prompts for authentication
8. Touch Yubikey again
9. Status: "YUBIKEY VERIFIED"
10. Full access granted
```

**Daily Use:**
```
1. Open tactical UI
2. See auth prompt: "Insert Yubikey and touch"
3. Touch Yubikey
4. Status: "YUBIKEY VERIFIED"
5. Use interface normally
```

### Authentication States

**NOT AUTHENTICATED:**
```
Security Status: AUTHENTICATING...
Yubikey Status:  NOT AUTHENTICATED

Behavior:
- Auth prompt displayed
- Operations blocked
- Send button disabled
- Error on message attempt
```

**AUTHENTICATED:**
```
Security Status: AUTHENTICATED
Yubikey Status:  YUBIKEY VERIFIED

Behavior:
- Auth prompt hidden
- Operations enabled
- Send button active
- Full functionality
```

**NO DEVICES:**
```
Security Status: AUTHENTICATING...
Yubikey Status:  NO DEVICES REGISTERED

Behavior:
- Registration prompt
- Operations blocked
- Settings modal shows empty list
```

### Protected Operations

All operations require authentication:

```javascript
// Sending messages
if (!authenticated) {
    addMessage('SYSTEM',
        'ERROR: Authentication required. Please authenticate with your Yubikey.',
        'error');
    showYubikeyAuth();
    return;
}
```

**Protected Actions:**
- Send messages
- Stream responses
- Execute operations
- Access system functions
- Modify settings

---

## Administration

### List All Devices

```bash
python3 02-ai-engine/yubikey_admin.py list
```

**Output:**
```
Registered Yubikey Devices (2):
============================================================

Device ID:      a3f4b5c2d1e6f7g8
Name:           Yubikey 5 NFC
Status:         ACTIVE
Serial:         12345678
Firmware:       5.4.3
Auth Methods:   fido2, cr
Credentials:    1
Created:        2025-11-13 10:30:00 UTC
Last Used:      2025-11-13 14:25:33 UTC

Device ID:      b4g5c6d2e7f8h9i0
Name:           Yubikey Backup
Status:         ACTIVE
Serial:         87654321
Firmware:       5.4.3
Auth Methods:   fido2
Credentials:    1
Created:        2025-11-13 10:35:00 UTC
Last Used:      Never

============================================================
Summary: 2 total, 2 active, 0 suspended, 0 revoked
```

### Show Device Information

```bash
python3 02-ai-engine/yubikey_admin.py info a3f4b5c2d1e6f7g8
```

**Output:**
```
Yubikey Device Information:
============================================================
Device ID:      a3f4b5c2d1e6f7g8
Name:           Yubikey 5 NFC
Status:         ACTIVE
Serial Number:  12345678
Firmware:       5.4.3
Created:        2025-11-13 10:30:00 UTC
Last Used:      2025-11-13 14:25:33 UTC

Authentication Methods:
  - FIDO2
  - CR

Challenge-Response:
  Slot: 2

FIDO2 Credentials (1):

  Credential 1:
    Name:       Yubikey 5 NFC
    Status:     active
    ID:         f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8...
    AAGUID:     2fc0579f-8113-47ea-b116-bb5a8db9202a
    Sign Count: 42
    Created:    2025-11-13 10:30:15 UTC
    Last Used:  2025-11-13 14:25:33 UTC

============================================================
```

### Suspend Device (Temporary)

Use when investigating security incident or device is temporarily unavailable:

```bash
python3 02-ai-engine/yubikey_admin.py suspend a3f4b5c2d1e6f7g8 \
    --reason "Under investigation"
```

**Output:**
```
Suspend Yubikey Device
============================================================
Device ID:   a3f4b5c2d1e6f7g8
Name:        Yubikey 5 NFC
Status:      ACTIVE

Reason: Under investigation

Are you sure? [y/N]: y

✓ Device suspended: Yubikey 5 NFC

To reactivate: yubikey_admin.py reactivate a3f4b5c2d1e6f7g8
```

**Effect:**
- Device cannot authenticate
- User must use another registered device
- Can be reactivated later

### Reactivate Suspended Device

```bash
python3 02-ai-engine/yubikey_admin.py reactivate a3f4b5c2d1e6f7g8
```

**Output:**
```
Reactivate Yubikey Device
============================================================
Device ID:   a3f4b5c2d1e6f7g8
Name:        Yubikey 5 NFC
Status:      SUSPENDED

Reactivate this device? [y/N]: y

✓ Device reactivated: Yubikey 5 NFC
```

### Revoke Device (Permanent)

Use when device is lost, stolen, or compromised:

```bash
python3 02-ai-engine/yubikey_admin.py revoke a3f4b5c2d1e6f7g8 \
    --reason "Device lost"
```

**Output:**
```
Revoke Yubikey Device
============================================================
Device ID:   a3f4b5c2d1e6f7g8
Name:        Yubikey 5 NFC
Status:      ACTIVE

⚠️  WARNING: Revocation is permanent!
Reason: Device lost

Are you sure? [y/N]: y

✓ Device revoked: Yubikey 5 NFC
```

**Effect:**
- Device permanently disabled
- Cannot authenticate
- Cannot be reactivated
- User must register new device

---

## Troubleshooting

### Yubikey Not Detected

**Symptoms:**
- `ykman list` shows nothing
- Browser doesn't prompt for touch
- "No Yubikey detected" error

**Solutions:**

**1. Check USB Connection:**
```bash
# List USB devices
lsusb | grep Yubico

# Expected output:
# Bus 001 Device 005: ID 1050:0407 Yubico.com Yubikey 5 NFC
```

**2. Check udev Rules:**
```bash
# Verify rules exist
ls -l /etc/udev/rules.d/70-u2f.rules

# Reload rules
sudo udevadm control --reload-rules
sudo udevadm trigger
```

**3. Check Group Membership:**
```bash
# Verify user in plugdev group
groups | grep plugdev

# If not, add user
sudo usermod -a -G plugdev $USER

# Log out and back in
```

**4. Check pcscd Service:**
```bash
# Check status
systemctl status pcscd

# If not running
sudo systemctl start pcscd
sudo systemctl enable pcscd
```

**5. Replug Yubikey:**
```bash
# Remove Yubikey, wait 5 seconds, reinsert
# Check detection
ykman list
```

### Browser Doesn't Show WebAuthn Prompt

**Symptoms:**
- Click "AUTHENTICATE" but nothing happens
- No touch prompt appears
- Console error: "WebAuthn not supported"

**Solutions:**

**1. Check Browser Support:**
```javascript
// Open browser console (F12)
console.log(window.PublicKeyCredential !== undefined);
// Should output: true
```

**Supported Browsers:**
- Chrome/Chromium 90+
- Firefox 88+
- Edge 90+
- Safari 14+

**2. Check Secure Context:**

WebAuthn requires HTTPS or localhost:

```bash
# Localhost works
http://localhost:5001/tactical_yubikey_ui.html ✓
http://127.0.0.1:5001/tactical_yubikey_ui.html ✓

# Plain HTTP doesn't work (unless localhost)
http://192.168.1.100:5001/tactical_yubikey_ui.html ✗

# HTTPS works
https://tactical.example.com/tactical_yubikey_ui.html ✓
```

**3. Check Console Errors:**

Open browser console (F12) and look for errors:
```
Failed to execute 'create' on 'CredentialsContainer':
The origin is not a valid secure context.

Solution: Use HTTPS or localhost
```

**4. Clear Browser Cache:**
```bash
# Chrome/Chromium
Ctrl+Shift+Del → Clear browsing data

# Firefox
Ctrl+Shift+Del → Clear recent history
```

### Authentication Fails

**Symptoms:**
- Touch Yubikey but authentication fails
- Error: "Authentication verification failed"
- Status remains "NOT AUTHENTICATED"

**Solutions:**

**1. Check Device Status:**
```bash
python3 02-ai-engine/yubikey_admin.py list
```

Ensure device status is ACTIVE, not SUSPENDED or REVOKED.

**2. Check Server Connection:**

Open browser console (F12):
```javascript
// Test server connectivity
fetch('http://127.0.0.1:5001/api/yubikey/devices')
    .then(r => r.json())
    .then(d => console.log('Devices:', d));

// Should show registered devices
```

**3. Check Backend Logs:**
```bash
# Check Flask logs
tail -f /tmp/dsmil-backend.log

# Check audit log
tail -f ~/.dsmil/yubikey/audit.log
```

**4. Re-register Device:**

If all else fails, revoke and re-register:
```bash
# Revoke old device
python3 02-ai-engine/yubikey_admin.py revoke <device-id>

# Register new device
python3 02-ai-engine/yubikey_admin.py register
```

### Multiple Yubikeys Detected

**Symptoms:**
- Have multiple Yubikeys plugged in
- Unclear which one to use

**Solution:**

Remove all but one Yubikey, or use serial number:

```bash
# List all connected Yubikeys
ykman list

# Select specific Yubikey by serial
ykman --device 12345678 info
```

### Permission Denied Errors

**Symptoms:**
```
PermissionError: [Errno 13] Permission denied: '/dev/hidraw0'
```

**Solutions:**

**1. Check Group Membership:**
```bash
groups | grep plugdev
```

**2. Check udev Rules:**
```bash
ls -l /dev/hidraw* | grep plugdev
```

**3. Reboot:**

Sometimes a reboot is required for udev rules to fully apply:
```bash
sudo reboot
```

---

## Security Considerations

### Threat Model

**Protected Against:**
- ✅ Phishing (FIDO2 domain binding)
- ✅ Man-in-the-middle (signed challenges)
- ✅ Replay attacks (nonces, counters)
- ✅ Credential theft (hardware-backed)
- ✅ Password guessing (no passwords)
- ✅ Social engineering (physical device required)
- ✅ Remote attacks (local device required)

**Not Protected Against:**
- ⚠️ Physical theft of unlocked system
- ⚠️ Physical theft of Yubikey + knowledge of PIN
- ⚠️ Advanced persistent threats with physical access
- ⚠️ Firmware vulnerabilities (keep updated)
- ⚠️ Browser vulnerabilities (keep updated)

### Best Practices

**Device Management:**
1. Register multiple Yubikeys (primary + backup)
2. Store backup Yubikey in secure location
3. Use descriptive device names
4. Regularly review registered devices
5. Revoke lost/stolen devices immediately

**Operational Security:**
1. Never share Yubikeys
2. Don't leave Yubikey plugged in unattended
3. Use screen lock when away from system
4. Review authentication logs regularly
5. Report suspicious activity immediately

**Physical Security:**
1. Attach Yubikey to keychain
2. Keep backup in safe/vault
3. Don't photograph or publish serial numbers
4. Use tamper-evident storage for backups
5. Register device loss with security officer

**PIN Protection:**

Some Yubikeys support PIN protection:

```bash
# Set FIDO2 PIN
ykman fido access change-pin

# Require PIN for authentication
# (increases security, requires PIN + touch)
```

**Touch Requirement:**

Always require touch for authentication:
```python
# In yubikey_auth.py
user_verification=UserVerificationRequirement.REQUIRED
```

This prevents silent background authentication.

### Compliance

**NIST SP 800-63B:**
- Yubikey qualifies as **Authenticator Assurance Level 3 (AAL3)**
- Hardware-backed, phishing-resistant
- Suitable for high-security applications

**FIPS 140-2:**
- YubiKey 5 FIPS Series available
- Level 2 certification
- Required for federal use

**TEMPEST:**
- No additional EM emissions from Yubikey
- USB communication minimal
- Compatible with all TEMPEST modes

---

## API Reference

### Python API

**Initialize Authentication:**
```python
from yubikey_auth import YubikeyAuth

auth = YubikeyAuth(storage_dir="~/.dsmil/yubikey")
```

**Begin FIDO2 Registration:**
```python
result = auth.begin_fido2_registration(
    user="tactical_user",
    user_display_name="Tactical User"
)
# Returns: {'publicKey': {...}, 'status': 'success'}
```

**Complete FIDO2 Registration:**
```python
device_id = auth.complete_fido2_registration(
    user="tactical_user",
    client_data={...},  # From browser
    device_name="Yubikey 5 NFC"
)
# Returns: Device ID (string)
```

**Begin FIDO2 Authentication:**
```python
result = auth.begin_fido2_authentication(user="tactical_user")
# Returns: {'publicKey': {...}, 'status': 'success'}
```

**Complete FIDO2 Authentication:**
```python
success = auth.complete_fido2_authentication(
    user="tactical_user",
    credential_id="f3a4b5c6...",
    client_data={...}  # From browser
)
# Returns: True/False
```

**List Devices:**
```python
devices = auth.list_devices(status=DeviceStatus.ACTIVE)
# Returns: List[YubikeyDevice]
```

**Get Device:**
```python
device = auth.get_device(device_id="a3f4b5c2...")
# Returns: YubikeyDevice or None
```

**Revoke Device:**
```python
success = auth.revoke_device(
    device_id="a3f4b5c2...",
    reason="Device lost"
)
# Returns: True/False
```

### REST API Endpoints

**Backend must implement these endpoints for UI integration:**

**POST /api/yubikey/register/begin**
```json
Request:
{
  "username": "tactical_user",
  "displayName": "Tactical User"
}

Response:
{
  "publicKey": {
    "challenge": "base64...",
    "rp": {"id": "localhost", "name": "DSMIL"},
    "user": {"id": "base64...", "name": "tactical_user", "displayName": "Tactical User"},
    "pubKeyCredParams": [...],
    "timeout": 60000,
    "attestation": "none",
    "authenticatorSelection": {
      "userVerification": "preferred"
    }
  },
  "status": "success"
}
```

**POST /api/yubikey/register/complete**
```json
Request:
{
  "username": "tactical_user",
  "deviceName": "Yubikey 5 NFC",
  "credential": {
    "id": "base64...",
    "rawId": "base64...",
    "response": {
      "attestationObject": "base64...",
      "clientDataJSON": "base64..."
    },
    "type": "public-key"
  }
}

Response:
{
  "device_id": "a3f4b5c2d1e6f7g8",
  "status": "success",
  "message": "Device registered successfully"
}
```

**POST /api/yubikey/auth/begin**
```json
Request:
{
  "username": "tactical_user"
}

Response:
{
  "publicKey": {
    "challenge": "base64...",
    "timeout": 60000,
    "rpId": "localhost",
    "allowCredentials": [
      {
        "type": "public-key",
        "id": "base64..."
      }
    ],
    "userVerification": "preferred"
  },
  "status": "success"
}
```

**POST /api/yubikey/auth/complete**
```json
Request:
{
  "username": "tactical_user",
  "credential": {
    "id": "base64...",
    "rawId": "base64...",
    "response": {
      "authenticatorData": "base64...",
      "clientDataJSON": "base64...",
      "signature": "base64...",
      "userHandle": "base64..."
    },
    "type": "public-key"
  }
}

Response:
{
  "status": "success",
  "authenticated": true,
  "message": "Authentication successful"
}
```

**GET /api/yubikey/devices**
```json
Response:
[
  {
    "device_id": "a3f4b5c2d1e6f7g8",
    "device_name": "Yubikey 5 NFC",
    "status": "active",
    "serial_number": "12345678",
    "firmware_version": "5.4.3",
    "auth_methods": ["fido2", "cr"],
    "credentials": [
      {
        "credential_id": "f3a4b5c6...",
        "device_name": "Yubikey 5 NFC",
        "status": "active",
        "sign_count": 42,
        "created_at": "2025-11-13T10:30:15Z",
        "last_used": "2025-11-13T14:25:33Z"
      }
    ],
    "created_at": "2025-11-13T10:30:00Z",
    "last_used": "2025-11-13T14:25:33Z"
  }
]
```

---

## Best Practices

### For Users

**Daily Operations:**
1. Keep Yubikey on keychain
2. Touch only when prompted
3. Don't share your Yubikey
4. Report lost device immediately
5. Use screen lock when away

**Registration:**
1. Register 2-3 devices (primary + backups)
2. Test each device after registration
3. Store backup securely
4. Document which key is which
5. Update device names descriptively

**If Device Lost:**
1. Report to security officer immediately
2. Revoke device using another Yubikey
3. Register replacement device
4. Update documentation
5. Change passwords as precaution

### For Administrators

**Setup:**
1. Install Yubikey support on all tactical systems
2. Configure automated testing
3. Document recovery procedures
4. Train users on proper usage
5. Establish device procurement process

**Management:**
1. Review registered devices weekly
2. Audit authentication logs daily
3. Revoke inactive devices (>90 days)
4. Track device inventory
5. Maintain spare devices

**Security:**
1. Require Yubikey for all protected operations
2. Enforce touch requirement
3. Enable PIN where appropriate
4. Monitor for suspicious activity
5. Update firmware regularly

**Incident Response:**
1. Immediate revocation of compromised devices
2. Force re-authentication of all users
3. Audit recent authentication logs
4. Investigate source of compromise
5. Issue replacement devices

---

## Appendix A: Quick Reference

### Installation Commands
```bash
# Automated install
sudo ./deployment/configure_yubikey.sh install

# Test detection
./deployment/configure_yubikey.sh test

# Check status
./deployment/configure_yubikey.sh status
```

### Device Management
```bash
# Register device
python3 02-ai-engine/yubikey_admin.py register

# List devices
python3 02-ai-engine/yubikey_admin.py list

# Show device info
python3 02-ai-engine/yubikey_admin.py info <device-id>

# Suspend device
python3 02-ai-engine/yubikey_admin.py suspend <device-id>

# Reactivate device
python3 02-ai-engine/yubikey_admin.py reactivate <device-id>

# Revoke device
python3 02-ai-engine/yubikey_admin.py revoke <device-id>

# Test authentication
python3 02-ai-engine/yubikey_admin.py test
```

### Troubleshooting
```bash
# Check USB detection
lsusb | grep Yubico

# Check ykman detection
ykman list

# Check group membership
groups | grep plugdev

# Check pcscd service
systemctl status pcscd

# View audit log
tail -f ~/.dsmil/yubikey/audit.log
```

---

## Appendix B: Supported Yubikey Features

| Feature | YubiKey 5 | YubiKey 4 | Security Key |
|---------|-----------|-----------|--------------|
| FIDO2/WebAuthn | ✅ | ✅ | ✅ |
| FIDO U2F | ✅ | ✅ | ✅ |
| Challenge-Response | ✅ | ✅ | ❌ |
| Yubico OTP | ✅ | ✅ | ❌ |
| OATH-TOTP/HOTP | ✅ | ✅ | ❌ |
| OpenPGP | ✅ | ✅ | ❌ |
| PIV (Smart Card) | ✅ | ✅ | ❌ |
| NFC | Varies | ❌ | Varies |
| USB-C | Varies | Varies | Varies |

---

## Appendix C: Additional Resources

**Official Documentation:**
- Yubico Developer: https://developers.yubico.com/
- WebAuthn Guide: https://webauthn.guide/
- FIDO Alliance: https://fidoalliance.org/

**DSMIL Documentation:**
- `00-documentation/TACTICAL_INTERFACE_GUIDE.md` - Tactical UI guide
- `00-documentation/TEMPEST_COMPLIANCE.md` - TEMPEST features
- `00-documentation/XEN_INTEGRATION_GUIDE.md` - Xen VM integration
- `01-source/kernel/API_REFERENCE.md` - Driver API reference

**Python Modules:**
- `02-ai-engine/yubikey_auth.py` - Authentication module
- `02-ai-engine/yubikey_admin.py` - Admin tool

**Scripts:**
- `deployment/configure_yubikey.sh` - Installation script

---

**Document Version:** 2.1.0
**Last Updated:** 2025-11-13
**Classification:** UNCLASSIFIED // FOR OFFICIAL USE ONLY
**Status:** ✅ Production Ready
