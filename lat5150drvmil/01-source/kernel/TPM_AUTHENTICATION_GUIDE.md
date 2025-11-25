# TPM Authentication Guide for DSMIL Driver

**Version:** 5.2.0
**Date:** 2025-11-13
**Target:** TPM 2.0 Hardware

---

## Table of Contents

1. [Overview](#overview)
2. [TPM 2.0 Basics](#tpm-20-basics)
3. [DSMIL TPM Integration](#dsmil-tpm-integration)
4. [Setup and Prerequisites](#setup-and-prerequisites)
5. [Authentication Flow](#authentication-flow)
6. [Using tpm2-tools](#using-tpm2-tools)
7. [Complete Examples](#complete-examples)
8. [TPM PCR Measurements](#tpm-pcr-measurements)
9. [Remote Attestation](#remote-attestation)
10. [Troubleshooting](#troubleshooting)

---

## Overview

The DSMIL driver uses TPM 2.0 for hardware-backed authentication, providing:

- **Hardware security** - Private keys never leave TPM
- **Challenge-response authentication** - Cryptographically secure
- **Platform attestation** - Remote verification via PCR measurements
- **Audit trail** - TPM-recorded security events
- **Graceful fallback** - Works without TPM using CAP_SYS_ADMIN

### Why TPM Authentication?

Traditional authentication methods (passwords, keys on disk) can be compromised. TPM provides:

1. **Hardware protection** - Keys stored in tamper-resistant hardware
2. **Attestation** - Cryptographic proof of platform state
3. **Binding** - Operations bound to specific hardware
4. **Sealing** - Data encrypted to specific PCR values

---

## TPM 2.0 Basics

### What is a TPM?

A **Trusted Platform Module (TPM)** is a hardware security chip that provides:
- Cryptographic key generation and storage
- Platform Configuration Registers (PCRs) for measurement
- Secure random number generation
- Cryptographic operations (sign, encrypt, HMAC)

### TPM 2.0 vs TPM 1.2

DSMIL requires **TPM 2.0** (not TPM 1.2):

| Feature | TPM 1.2 | TPM 2.0 |
|---------|---------|---------|
| Algorithm Agility | Fixed (SHA-1, RSA) | Flexible (SHA-256, ECC, etc.) |
| PCR Banks | Single | Multiple (SHA-1, SHA-256, etc.) |
| Authorization | Complex | Simplified sessions |
| **DSMIL Support** | ❌ No | ✅ Yes |

### Platform Configuration Registers (PCRs)

PCRs are 32 registers (0-31) that store measurements:
- **Extend-only** - Can only add measurements, not overwrite
- **Resets** - Only reset on reboot
- **Banks** - Multiple hash algorithms (SHA-1, SHA-256, etc.)

DSMIL uses these PCRs:
- **PCR 16** - Authentication events
- **PCR 17** - Protected token access
- **PCR 18** - BIOS operations
- **PCR 23** - Security events

---

## DSMIL TPM Integration

### Architecture

```
┌─────────────────────────────────────────────┐
│          User Application                    │
│  (Protected token access required)           │
└──────────────────┬──────────────────────────┘
                   │
                   │ 1. Get challenge
                   ▼
┌─────────────────────────────────────────────┐
│          DSMIL Driver                        │
│  ┌──────────────────────────────────────┐   │
│  │  TPM Authentication Context          │   │
│  │  - Generate 256-bit challenge        │   │
│  │  - Manage session state              │   │
│  │  - Validate signatures                │   │
│  │  - Extend PCRs                        │   │
│  └──────────────────────────────────────┘   │
└──────────────────┬──────────────────────────┘
                   │
                   │ 2. Sign challenge
                   ▼
┌─────────────────────────────────────────────┐
│          TPM 2.0 Chip                        │
│  ┌──────────────────────────────────────┐   │
│  │  Signing Key (0x81000001)            │   │
│  │  - RSA 2048 or ECC P-256             │   │
│  │  - Private key never leaves TPM       │   │
│  └──────────────────────────────────────┘   │
│  ┌──────────────────────────────────────┐   │
│  │  PCR Registers (16,17,18,23)         │   │
│  │  - Measurements recorded              │   │
│  │  - Attestation evidence               │   │
│  └──────────────────────────────────────┘   │
└─────────────────────────────────────────────┘
                   │
                   │ 3. Return signature
                   ▼
┌─────────────────────────────────────────────┐
│          User Application                    │
│  - Submit signature to driver                │
│  - Access granted if valid                   │
└─────────────────────────────────────────────┘
```

### Authentication Modes

DSMIL supports multiple TPM authentication modes:

| Mode | Description | Security Level | Complexity |
|------|-------------|----------------|------------|
| **Challenge-Response** | TPM signs random challenge | High | Low |
| **Key-based** | TPM key used for encryption | High | Medium |
| **HMAC** | TPM generates HMAC | Medium | Low |
| **External** | External authenticator (future) | Varies | High |

**Default mode:** Challenge-Response (recommended)

---

## Setup and Prerequisites

### 1. Check TPM Availability

```bash
# Check TPM device
ls -l /dev/tpm*
# Expected output:
# crw-rw---- 1 tss tss  10, 224 Nov 13 10:00 /dev/tpm0
# crw-rw-rw- 1 tss tss 253,   0 Nov 13 10:00 /dev/tpmrm0

# Check TPM version
cat /sys/class/tpm/tpm0/tpm_version_major
# Expected output: 2
```

**If no TPM device:**
- Physical system: Enable TPM in BIOS/UEFI settings
- Virtual machine: Pass through TPM device or use vTPM (swtpm)

### 2. Install TPM Tools

```bash
# Ubuntu/Debian
sudo apt-get install -y tpm2-tools tpm2-abrmd

# Fedora/RHEL
sudo dnf install -y tpm2-tools tpm2-abrmd

# Arch Linux
sudo pacman -S tpm2-tools tpm2-abrmd
```

### 3. Start TPM Resource Manager

```bash
# Enable and start TPM resource manager
sudo systemctl enable tpm2-abrmd
sudo systemctl start tpm2-abrmd

# Verify
sudo systemctl status tpm2-abrmd
```

### 4. Create TPM Signing Key

```bash
# Create primary key in owner hierarchy
tpm2_createprimary -C o -g sha256 -G rsa -c primary.ctx

# Create signing key
tpm2_create -C primary.ctx -g sha256 -G rsa \
    -r key.priv -u key.pub -a "fixedtpm|fixedparent|sensitivedataorigin|userwithauth|sign"

# Load key into TPM
tpm2_load -C primary.ctx -r key.priv -u key.pub -c key.ctx

# Persist key to permanent handle
tpm2_evictcontrol -C o -c key.ctx 0x81000001

# Verify
tpm2_readpublic -c 0x81000001
```

**Note:** Handle `0x81000001` is used by DSMIL examples. You can use any persistent handle.

### 5. Load DSMIL Driver with TPM

```bash
# Load driver
sudo insmod dsmil-104dev.ko

# Verify TPM status
cat /sys/class/dsmil/dsmil0/tpm_status
# Expected:
# state:           ready
# available:       yes
# chip_present:    yes
# auth_mode:       challenge
```

---

## Authentication Flow

### Step-by-Step Process

#### Step 1: Get Challenge from Driver

```c
int fd = open("/dev/dsmil0", O_RDWR);

struct dsmil_tpm_challenge_data chal;
if (ioctl(fd, DSMIL_IOC_TPM_GET_CHALLENGE, &chal) < 0) {
    perror("DSMIL_IOC_TPM_GET_CHALLENGE");
    return -1;
}

printf("Challenge ID: 0x%08x\n", chal.challenge_id);
printf("TPM Available: %s\n", chal.tpm_available ? "yes" : "no");
```

**What happens:**
- Driver generates cryptographically secure 256-bit random challenge
- Challenge valid for 60 seconds (default)
- Challenge ID used to correlate response

#### Step 2: Sign Challenge with TPM

```bash
# Save challenge to file
echo -n "<challenge_hex>" | xxd -r -p > /tmp/challenge.bin

# Sign with TPM using persistent key
tpm2_sign -c 0x81000001 -g sha256 -o /tmp/signature.bin /tmp/challenge.bin

# Verify signature was created
ls -lh /tmp/signature.bin
# Expected: ~256 bytes for RSA 2048
```

**What happens:**
- TPM uses private key (never leaves TPM) to sign challenge
- Signature proves possession of TPM
- Signature format: PKCS#1 v1.5 or PSS (depending on TPM configuration)

#### Step 3: Submit Response to Driver

```c
// Read signature from file
FILE *fp = fopen("/tmp/signature.bin", "rb");
unsigned char signature[256];
size_t sig_len = fread(signature, 1, sizeof(signature), fp);
fclose(fp);

// Prepare authentication request
struct dsmil_auth_request auth;
memset(&auth, 0, sizeof(auth));
auth.auth_method = 1; // DSMIL_AUTH_METHOD_CHALLENGE

// Pack challenge ID + signature
memcpy(auth.auth_data, &chal.challenge_id, sizeof(chal.challenge_id));
memcpy(auth.auth_data + 4, signature, sig_len);
auth.auth_data_len = 4 + sig_len;

// Submit authentication
if (ioctl(fd, DSMIL_IOC_AUTHENTICATE, &auth) < 0) {
    perror("DSMIL_IOC_AUTHENTICATE");
    return -1;
}

printf("Authentication successful!\n");
```

**What happens:**
- Driver verifies challenge ID matches active challenge
- Driver validates signature (implementation-specific)
- On success:
  - Session created (valid for 5 minutes default)
  - PCR 16 extended with authentication event
  - Audit log entry created
  - User can now access protected tokens

#### Step 4: Access Protected Token

```c
// Now we can write protected tokens
struct dsmil_token_request req;
req.token_id = 0x8500; // TOKEN_SECURITY_MASTER
req.value = 0xCAFEBABE;

if (ioctl(fd, DSMIL_IOC_WRITE_TOKEN, &req) < 0) {
    perror("DSMIL_IOC_WRITE_TOKEN");
    return -1;
}

printf("Protected token 0x%04x written successfully\n", req.token_id);
```

**What happens:**
- Driver checks for valid authenticated session
- Driver checks CAP_SYS_ADMIN capability
- PCR 17 extended with protected token access
- Audit log entry created
- Token written to SMBIOS

---

## Using tpm2-tools

### Essential TPM Commands

#### List Persistent Handles

```bash
tpm2_getcap handles-persistent
# Output:
# - 0x81000001
```

#### Read Public Key

```bash
tpm2_readpublic -c 0x81000001 -o /tmp/public.pem -f pem
cat /tmp/public.pem
# Output: PEM-encoded RSA public key
```

#### Read PCR Values

```bash
# Read DSMIL PCRs
tpm2_pcrread sha256:16,17,18,23

# Output:
# sha256:
#   16: 0x0000000000000000000000000000000000000000000000000000000000000000
#   17: 0x0000000000000000000000000000000000000000000000000000000000000000
#   18: 0x0000000000000000000000000000000000000000000000000000000000000000
#   23: 0x0000000000000000000000000000000000000000000000000000000000000000
```

#### Extend PCR Manually

```bash
# Extend PCR 23 with test data
echo "test event" | tpm2_pcrextend 23:sha256=0

# Verify
tpm2_pcrread sha256:23
```

#### Create Quote (Attestation)

```bash
# Create attestation quote
tpm2_quote -c 0x81000001 -l sha256:16,17,18,23 \
    -q "nonce" -m /tmp/quote.msg -s /tmp/quote.sig -o /tmp/pcr.out

# View quote
cat /tmp/quote.msg | xxd
```

### Advanced: PCR Policy

```bash
# Create policy requiring specific PCR values
tpm2_pcrread sha256:16,17,18,23 -o /tmp/pcr.dat
tpm2_createpolicy --policy-pcr -l sha256:16,17,18,23 \
    -f /tmp/pcr.dat -L /tmp/pcr.policy

# Create key bound to PCR policy
tpm2_create -C primary.ctx -g sha256 -G rsa \
    -L /tmp/pcr.policy \
    -r key_pcr.priv -u key_pcr.pub
```

---

## Complete Examples

### Example 1: Simple Authentication

**auth_simple.c:**
```c
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <string.h>

#define DSMIL_IOC_MAGIC 'D'
#define DSMIL_IOC_TPM_GET_CHALLENGE _IOR(DSMIL_IOC_MAGIC, 11, struct dsmil_tpm_challenge_data)
#define DSMIL_IOC_AUTHENTICATE      _IOW(DSMIL_IOC_MAGIC, 3, struct dsmil_auth_request)

struct dsmil_tpm_challenge_data {
    unsigned char challenge[32];
    unsigned int challenge_id;
    unsigned char tpm_available;
};

struct dsmil_auth_request {
    unsigned int auth_method;
    unsigned int auth_data_len;
    unsigned char auth_data[256];
};

void save_challenge(struct dsmil_tpm_challenge_data *chal) {
    FILE *fp = fopen("/tmp/challenge.bin", "wb");
    fwrite(chal->challenge, 1, 32, fp);
    fclose(fp);
}

int load_signature(unsigned char *sig) {
    FILE *fp = fopen("/tmp/signature.bin", "rb");
    if (!fp) return -1;
    int len = fread(sig, 1, 256, fp);
    fclose(fp);
    return len;
}

int main() {
    int fd, ret;
    struct dsmil_tpm_challenge_data chal;
    struct dsmil_auth_request auth;

    printf("=== DSMIL TPM Authentication ===\n\n");

    // Open device
    fd = open("/dev/dsmil0", O_RDWR);
    if (fd < 0) {
        perror("open /dev/dsmil0");
        return 1;
    }

    // Step 1: Get challenge
    printf("Step 1: Getting challenge from driver...\n");
    ret = ioctl(fd, DSMIL_IOC_TPM_GET_CHALLENGE, &chal);
    if (ret < 0) {
        perror("DSMIL_IOC_TPM_GET_CHALLENGE");
        close(fd);
        return 1;
    }

    printf("  Challenge ID: 0x%08x\n", chal.challenge_id);
    printf("  TPM Available: %s\n", chal.tpm_available ? "yes" : "no");

    if (!chal.tpm_available) {
        printf("  WARNING: TPM not available, using fallback auth\n");
    }

    // Save challenge for TPM signing
    save_challenge(&chal);
    printf("  Challenge saved to /tmp/challenge.bin\n");

    // Step 2: Sign with TPM (external)
    printf("\nStep 2: Signing challenge with TPM...\n");
    printf("  Run: tpm2_sign -c 0x81000001 -g sha256 -o /tmp/signature.bin /tmp/challenge.bin\n");
    printf("  Press Enter when signature is ready...");
    getchar();

    // Step 3: Load signature
    unsigned char signature[256];
    int sig_len = load_signature(signature);
    if (sig_len < 0) {
        fprintf(stderr, "ERROR: Failed to load signature from /tmp/signature.bin\n");
        close(fd);
        return 1;
    }

    printf("  Signature loaded (%d bytes)\n", sig_len);

    // Step 4: Submit authentication
    printf("\nStep 3: Submitting authentication...\n");
    memset(&auth, 0, sizeof(auth));
    auth.auth_method = 1; // Challenge-response
    memcpy(auth.auth_data, &chal.challenge_id, sizeof(chal.challenge_id));
    memcpy(auth.auth_data + 4, signature, sig_len);
    auth.auth_data_len = 4 + sig_len;

    ret = ioctl(fd, DSMIL_IOC_AUTHENTICATE, &auth);
    if (ret < 0) {
        perror("DSMIL_IOC_AUTHENTICATE");
        close(fd);
        return 1;
    }

    printf("  ✓ Authentication successful!\n");
    printf("\nAuthenticated session active for 5 minutes\n");
    printf("You can now access protected tokens\n");

    close(fd);
    return 0;
}
```

**Compile and run:**
```bash
gcc -o auth_simple auth_simple.c
sudo ./auth_simple
```

---

### Example 2: Automated Authentication Script

**auth_auto.sh:**
```bash
#!/bin/bash
set -e

echo "=== DSMIL Automated TPM Authentication ==="

# Configuration
DEVICE="/dev/dsmil0"
TPM_KEY="0x81000001"
TEMP_DIR="/tmp/dsmil_auth"

mkdir -p "$TEMP_DIR"

# Step 1: Get challenge
echo -e "\n[1/4] Getting challenge from driver..."
cat > "$TEMP_DIR/get_challenge.c" << 'EOF'
#include <stdio.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <unistd.h>

#define DSMIL_IOC_MAGIC 'D'
#define DSMIL_IOC_TPM_GET_CHALLENGE _IOR(DSMIL_IOC_MAGIC, 11, struct dsmil_tpm_challenge_data)

struct dsmil_tpm_challenge_data {
    unsigned char challenge[32];
    unsigned int challenge_id;
    unsigned char tpm_available;
};

int main() {
    int fd = open("/dev/dsmil0", O_RDWR);
    if (fd < 0) return 1;

    struct dsmil_tpm_challenge_data chal;
    if (ioctl(fd, DSMIL_IOC_TPM_GET_CHALLENGE, &chal) < 0) {
        close(fd);
        return 1;
    }

    // Write challenge to stdout
    fwrite(&chal, sizeof(chal), 1, stdout);

    close(fd);
    return 0;
}
EOF

gcc -o "$TEMP_DIR/get_challenge" "$TEMP_DIR/get_challenge.c"
"$TEMP_DIR/get_challenge" > "$TEMP_DIR/challenge.dat"

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to get challenge"
    exit 1
fi

# Extract challenge and ID
dd if="$TEMP_DIR/challenge.dat" of="$TEMP_DIR/challenge.bin" bs=1 count=32 skip=0 2>/dev/null
dd if="$TEMP_DIR/challenge.dat" of="$TEMP_DIR/challenge_id.bin" bs=1 count=4 skip=32 2>/dev/null

CHALLENGE_ID=$(xxd -p -l 4 "$TEMP_DIR/challenge_id.bin" | tr -d '\n')
echo "Challenge ID: 0x$CHALLENGE_ID"

# Step 2: Sign with TPM
echo -e "\n[2/4] Signing challenge with TPM..."
tpm2_sign -c "$TPM_KEY" -g sha256 \
    -o "$TEMP_DIR/signature.bin" \
    "$TEMP_DIR/challenge.bin" 2>/dev/null

if [ $? -ne 0 ]; then
    echo "ERROR: TPM signing failed"
    exit 1
fi

SIG_SIZE=$(stat -c%s "$TEMP_DIR/signature.bin")
echo "Signature created ($SIG_SIZE bytes)"

# Step 3: Submit authentication
echo -e "\n[3/4] Submitting authentication..."
cat > "$TEMP_DIR/authenticate.c" << 'EOF'
#include <stdio.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <string.h>
#include <unistd.h>

#define DSMIL_IOC_MAGIC 'D'
#define DSMIL_IOC_AUTHENTICATE _IOW(DSMIL_IOC_MAGIC, 3, struct dsmil_auth_request)

struct dsmil_auth_request {
    unsigned int auth_method;
    unsigned int auth_data_len;
    unsigned char auth_data[256];
};

int main(int argc, char *argv[]) {
    int fd = open("/dev/dsmil0", O_RDWR);
    if (fd < 0) return 1;

    // Read challenge ID and signature from stdin
    unsigned int challenge_id;
    unsigned char signature[256];

    fread(&challenge_id, sizeof(challenge_id), 1, stdin);
    size_t sig_len = fread(signature, 1, sizeof(signature), stdin);

    // Prepare auth request
    struct dsmil_auth_request auth;
    memset(&auth, 0, sizeof(auth));
    auth.auth_method = 1;
    memcpy(auth.auth_data, &challenge_id, sizeof(challenge_id));
    memcpy(auth.auth_data + 4, signature, sig_len);
    auth.auth_data_len = 4 + sig_len;

    // Submit
    int ret = ioctl(fd, DSMIL_IOC_AUTHENTICATE, &auth);

    close(fd);
    return (ret < 0) ? 1 : 0;
}
EOF

gcc -o "$TEMP_DIR/authenticate" "$TEMP_DIR/authenticate.c"
cat "$TEMP_DIR/challenge_id.bin" "$TEMP_DIR/signature.bin" | \
    sudo "$TEMP_DIR/authenticate"

if [ $? -ne 0 ]; then
    echo "ERROR: Authentication failed"
    exit 1
fi

echo "✓ Authentication successful!"

# Step 4: Verify session
echo -e "\n[4/4] Verifying session..."
TPM_STATUS=$(cat /sys/class/dsmil/dsmil0/tpm_status)
if echo "$TPM_STATUS" | grep -q "session_active:  yes"; then
    echo "✓ Session active"
    echo ""
    echo "$TPM_STATUS"
else
    echo "WARNING: Session not active"
fi

echo -e "\n=== Authentication Complete ==="
echo "You can now access protected tokens for the next 5 minutes"

# Cleanup
rm -rf "$TEMP_DIR"
```

**Usage:**
```bash
chmod +x auth_auto.sh
./auth_auto.sh
```

---

### Example 3: Protected Token Access

**protected_token_write.c:**
```c
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <errno.h>
#include <string.h>

#define DSMIL_IOC_MAGIC 'D'
#define DSMIL_IOC_WRITE_TOKEN _IOW(DSMIL_IOC_MAGIC, 2, struct dsmil_token_request)

struct dsmil_token_request {
    unsigned short token_id;
    unsigned int value;
};

int main(int argc, char *argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <token_id> <value>\n", argv[0]);
        fprintf(stderr, "Example: %s 0x8500 0xCAFEBABE\n", argv[0]);
        return 1;
    }

    unsigned short token_id = strtol(argv[1], NULL, 16);
    unsigned int value = strtol(argv[2], NULL, 16);

    printf("=== Protected Token Write ===\n");
    printf("Token ID: 0x%04x\n", token_id);
    printf("Value:    0x%08x\n\n", value);

    int fd = open("/dev/dsmil0", O_RDWR);
    if (fd < 0) {
        perror("open");
        return 1;
    }

    struct dsmil_token_request req;
    req.token_id = token_id;
    req.value = value;

    printf("Writing token...\n");
    int ret = ioctl(fd, DSMIL_IOC_WRITE_TOKEN, &req);

    if (ret < 0) {
        if (errno == EPERM) {
            fprintf(stderr, "\nERROR: Permission denied\n");
            fprintf(stderr, "Protected token requires authentication:\n");
            fprintf(stderr, "  1. Run authentication script first\n");
            fprintf(stderr, "  2. Ensure you have CAP_SYS_ADMIN (run with sudo)\n");
        } else {
            perror("ioctl");
        }
        close(fd);
        return 1;
    }

    printf("✓ Token written successfully\n");

    // Read audit log
    printf("\nChecking audit log...\n");
    FILE *fp = popen("cat /sys/class/dsmil/dsmil0/last_audit 2>/dev/null", "r");
    if (fp) {
        char line[256];
        while (fgets(line, sizeof(line), fp)) {
            printf("  %s", line);
        }
        pclose(fp);
    }

    close(fd);
    return 0;
}
```

**Compile and use:**
```bash
gcc -o protected_token_write protected_token_write.c

# Must authenticate first
./auth_auto.sh

# Then write protected token
sudo ./protected_token_write 0x8500 0xCAFEBABE
```

---

## TPM PCR Measurements

### PCR Usage by DSMIL

| PCR | Purpose | Events Measured |
|-----|---------|-----------------|
| **16** | Authentication | Auth attempts, successes, failures |
| **17** | Token Access | Protected token reads/writes |
| **18** | BIOS Operations | Failovers, sync operations, health changes |
| **23** | Security Events | Security violations, policy changes |

### Reading PCR Values

```bash
# Read all DSMIL PCRs
tpm2_pcrread sha256:16,17,18,23

# Read specific PCR
tpm2_pcrread sha256:16

# Read with multiple banks
tpm2_pcrread sha1:16,17,18,23 sha256:16,17,18,23
```

### Monitoring PCR Changes

```bash
#!/bin/bash
# monitor_pcr.sh - Monitor PCR changes in real-time

echo "Monitoring DSMIL PCRs (Ctrl+C to stop)..."

# Initial values
tpm2_pcrread sha256:16,17,18,23 > /tmp/pcr_prev.txt

while true; do
    sleep 5

    # Current values
    tpm2_pcrread sha256:16,17,18,23 > /tmp/pcr_curr.txt

    # Compare
    if ! diff -q /tmp/pcr_prev.txt /tmp/pcr_curr.txt > /dev/null 2>&1; then
        echo ""
        echo "[$(date)] PCR Change Detected:"
        diff /tmp/pcr_prev.txt /tmp/pcr_curr.txt || true
        cp /tmp/pcr_curr.txt /tmp/pcr_prev.txt
    fi
done
```

### PCR Event Log

```bash
# Get TPM event log (kernel boot log)
sudo cat /sys/kernel/security/tpm0/binary_bios_measurements | \
    tpm2_eventlog /dev/stdin

# Filter for DSMIL events (PCR 16-18, 23)
sudo cat /sys/kernel/security/tpm0/binary_bios_measurements | \
    tpm2_eventlog /dev/stdin | grep -E "PCR: (16|17|18|23)"
```

---

## Remote Attestation

### Overview

Remote attestation allows a remote party to verify the platform state using TPM quotes.

### Creating an Attestation Quote

```bash
#!/bin/bash
# create_quote.sh - Create TPM attestation quote

NONCE="$1"
if [ -z "$NONCE" ]; then
    echo "Usage: $0 <nonce>"
    exit 1
fi

# Create quote of DSMIL PCRs
tpm2_quote -c 0x81000001 \
    -l sha256:16,17,18,23 \
    -q "$NONCE" \
    -m quote.msg \
    -s quote.sig \
    -o pcr.out \
    -g sha256

echo "Quote created:"
echo "  Message:   quote.msg"
echo "  Signature: quote.sig"
echo "  PCR Data:  pcr.out"

# Export public key for verification
tpm2_readpublic -c 0x81000001 -o public.pem -f pem

echo "  Public Key: public.pem"
echo ""
echo "Send these files to the attestation server for verification"
```

### Verifying an Attestation Quote

```python
#!/usr/bin/env python3
# verify_quote.py - Verify TPM attestation quote

import sys
import hashlib
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.backends import default_backend

def verify_quote(public_key_file, quote_msg_file, quote_sig_file, expected_nonce):
    # Load public key
    with open(public_key_file, 'rb') as f:
        public_key = serialization.load_pem_public_key(
            f.read(),
            backend=default_backend()
        )

    # Read quote message
    with open(quote_msg_file, 'rb') as f:
        quote_msg = f.read()

    # Read signature
    with open(quote_sig_file, 'rb') as f:
        signature = f.read()

    # Verify signature
    try:
        public_key.verify(
            signature,
            quote_msg,
            padding.PKCS1v15(),
            hashes.SHA256()
        )
        print("✓ Signature valid")
    except Exception as e:
        print(f"✗ Signature invalid: {e}")
        return False

    # Verify nonce in quote
    if expected_nonce.encode() not in quote_msg:
        print("✗ Nonce mismatch")
        return False
    print("✓ Nonce correct")

    # Extract PCR values from quote (simplified)
    print("✓ Quote verified successfully")
    return True

if __name__ == '__main__':
    if len(sys.argv) != 5:
        print(f"Usage: {sys.argv[0]} <public.pem> <quote.msg> <quote.sig> <nonce>")
        sys.exit(1)

    result = verify_quote(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
    sys.exit(0 if result else 1)
```

**Usage:**
```bash
# On client (DSMIL system)
./create_quote.sh "random_nonce_12345"
# Send quote.msg, quote.sig, pcr.out, public.pem to server

# On server
./verify_quote.py public.pem quote.msg quote.sig "random_nonce_12345"
```

---

## Troubleshooting

### TPM Not Available

**Symptom:**
```bash
cat /sys/class/dsmil/dsmil0/tpm_status
# state: unavailable
# available: no
```

**Solutions:**

1. **Check TPM device:**
   ```bash
   ls -l /dev/tpm*
   # If missing, TPM not enabled in BIOS or not present
   ```

2. **Enable TPM in BIOS:**
   - Reboot and enter BIOS/UEFI settings
   - Find Security → TPM/Trusted Computing
   - Enable TPM 2.0 (not TPM 1.2)
   - Save and reboot

3. **Check kernel modules:**
   ```bash
   lsmod | grep tpm
   # Should see: tpm, tpm_tis, tpm_crb, etc.

   # Load if missing
   sudo modprobe tpm
   sudo modprobe tpm_tis
   ```

4. **Virtual machine - pass through or vTPM:**
   ```bash
   # For QEMU with swtpm
   sudo apt-get install swtpm swtpm-tools
   swtpm socket --tpmstate dir=/tmp/mytpm --ctrl type=unixio,path=/tmp/mytpm/swtpm-sock &
   ```

---

### Authentication Fails

**Symptom:**
```
ioctl: Permission denied
```

**Solutions:**

1. **Check privileges:**
   ```bash
   # Must run with sudo
   sudo ./auth_program
   ```

2. **Verify signing key exists:**
   ```bash
   tpm2_readpublic -c 0x81000001
   # If error, key not persisted - recreate key
   ```

3. **Check challenge hasn't expired:**
   ```bash
   # Challenge valid for 60 seconds
   # Must sign and submit within timeout
   ```

4. **Verify signature format:**
   ```bash
   # RSA 2048 signature should be ~256 bytes
   ls -l /tmp/signature.bin
   ```

5. **Check kernel messages:**
   ```bash
   dmesg | tail -20
   # Look for DSMIL TPM errors
   ```

---

### PCR Not Extended

**Symptom:**
```bash
# PCR values unchanged after operations
tpm2_pcrread sha256:16,17,18,23
# All zeros
```

**Solutions:**

1. **Verify TPM is active:**
   ```bash
   cat /sys/class/dsmil/dsmil0/tpm_status
   # Check: chip_present: yes
   ```

2. **Check driver messages:**
   ```bash
   dmesg | grep "DSMIL TPM"
   # Look for: "PCR extend failed" messages
   ```

3. **Verify PCR is not sealed:**
   ```bash
   # Some PCRs may be policy-sealed
   # Try extending PCR 23 manually
   echo "test" | tpm2_pcrextend 23:sha256=0
   ```

---

### Session Timeout

**Symptom:**
```
Protected token write succeeds immediately after auth,
but fails after a few minutes
```

**Solution:**
```bash
# Check session timeout setting
cat /sys/module/dsmil_104dev/parameters/auth_timeout
# Default: 300 seconds (5 minutes)

# Increase timeout (before loading driver)
sudo rmmod dsmil-104dev
sudo insmod dsmil-104dev.ko auth_timeout=1800  # 30 minutes
```

---

### TPM Resource Manager Issues

**Symptom:**
```
tpm2_sign: Failed to load key context
```

**Solution:**
```bash
# Restart TPM resource manager
sudo systemctl restart tpm2-abrmd

# Or use kernel resource manager (no daemon needed)
# Use /dev/tpmrm0 instead of /dev/tpm0
TPM2TOOLS_TCTI=device:/dev/tpmrm0 tpm2_sign -c 0x81000001 ...
```

---

## Best Practices

1. **Key Management:**
   - Store TPM keys in persistent handles (0x81000000-0x81FFFFFF)
   - Use unique keys per system
   - Never export private keys

2. **Session Management:**
   - Invalidate sessions when done (`DSMIL_IOC_TPM_INVALIDATE`)
   - Use appropriate timeouts for your use case
   - Monitor session_active in tpm_status

3. **PCR Usage:**
   - Monitor PCR values for unexpected changes
   - Use quotes for remote attestation
   - Document PCR event meanings

4. **Error Handling:**
   - Check all ioctl return values
   - Handle TPM unavailable gracefully
   - Log authentication failures

5. **Security:**
   - Always require CAP_SYS_ADMIN
   - Use TPM when available (don't disable)
   - Audit all protected token access
   - Monitor audit logs regularly

---

## Conclusion

This guide provides comprehensive coverage of TPM authentication for the DSMIL driver. For additional information:

- **DRIVER_USAGE_GUIDE.md** - General driver usage
- **API_REFERENCE.md** - Complete API documentation
- **TESTING_GUIDE.md** - Testing procedures

For TPM 2.0 documentation:
- [TPM 2.0 Library Specification](https://trustedcomputinggroup.org/resource/tpm-library-specification/)
- [tpm2-tools Documentation](https://github.com/tpm2-software/tpm2-tools)

For questions or issues, contact the development team or file an issue in the project repository.
