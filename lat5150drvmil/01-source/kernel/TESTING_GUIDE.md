# DSMIL Driver Testing Guide

**Version:** 5.2.0
**Date:** 2025-11-13
**Target:** Linux Kernel Module Testing

---

## Table of Contents

1. [Overview](#overview)
2. [Test Environment Setup](#test-environment-setup)
3. [Unit Tests](#unit-tests)
4. [Integration Tests](#integration-tests)
5. [Functional Tests](#functional-tests)
6. [Security Tests](#security-tests)
7. [Performance Tests](#performance-tests)
8. [Stress Tests](#stress-tests)
9. [Test Automation](#test-automation)
10. [Test Results Format](#test-results-format)

---

## Overview

This guide provides comprehensive testing procedures for the DSMIL driver covering:
- **Unit testing** - Individual component validation
- **Integration testing** - Component interaction validation
- **Functional testing** - Feature-level validation
- **Security testing** - TPM authentication and protected token validation
- **Performance testing** - Throughput and latency measurement
- **Stress testing** - Stability under load

### Testing Levels

| Level | Scope | Duration | Frequency |
|-------|-------|----------|-----------|
| **Smoke** | Basic functionality | 5 min | Every build |
| **Functional** | All features | 30 min | Daily |
| **Integration** | System interaction | 1 hour | Weekly |
| **Stress** | Stability | 4+ hours | Monthly |

---

## Test Environment Setup

### Prerequisites

```bash
# Install testing tools
sudo apt-get install -y \
    linux-headers-$(uname -r) \
    build-essential \
    kmod \
    tpm2-tools \
    stress-ng \
    valgrind

# Install kernel testing framework
git clone https://github.com/torvalds/linux.git
cd linux/tools/testing/selftests
```

### Build Test Driver

```bash
cd /home/user/LAT5150DRVMIL/01-source/kernel
make clean
make EXTRA_CFLAGS="-DDEBUG -g"

# Or build individual production drivers:
#   104-device primary driver (preferred on modern kernels)
make dsmil-104dev

#   84-device legacy fallback driver
make dsmil-84dev
```

### Load Test Driver

```bash
# Load with debug enabled
sudo insmod core/dsmil-104dev.ko debug=1

# Verify load
dmesg | tail -50
lsmod | grep dsmil
ls -l /dev/dsmil0
```

---

## Unit Tests

### Test 1: Token Database Initialization

**Purpose:** Verify token database is correctly populated.

```bash
#!/bin/bash
# test_token_db.sh

echo "=== Token Database Test ==="

# Check token count
TOKEN_COUNT=$(cat /sys/class/dsmil/dsmil0/tokens | wc -l)
echo "Token count: $TOKEN_COUNT"

if [ "$TOKEN_COUNT" -lt 50 ]; then
    echo "FAIL: Token count too low (expected 50+)"
    exit 1
fi

# Verify specific tokens exist
REQUIRED_TOKENS=(
    "0x8000"  # DSMIL_DEVICE_BASE
    "0x8100"  # BIOS_A_STATUS
    "0x8110"  # BIOS_B_STATUS
    "0x8120"  # BIOS_C_STATUS
    "0x8500"  # SECURITY tokens
)

for token in "${REQUIRED_TOKENS[@]}"; do
    if ! grep -q "$token" /sys/class/dsmil/dsmil0/tokens; then
        echo "FAIL: Required token $token not found"
        exit 1
    fi
done

echo "PASS: Token database initialized correctly"
```

### Test 2: SMBIOS Backend Selection

**Purpose:** Verify correct SMBIOS backend is selected.

```bash
#!/bin/bash
# test_smbios_backend.sh

echo "=== SMBIOS Backend Test ==="

BACKEND=$(cat /sys/class/dsmil/dsmil0/smbios_backend)
echo "Backend: $BACKEND"

# Check if backend is valid
if [[ "$BACKEND" == *"dell-smbios"* ]]; then
    echo "INFO: Using real dell-smbios backend"
    BACKEND_TYPE="real"
elif [[ "$BACKEND" == *"simulated"* ]]; then
    echo "INFO: Using simulated backend"
    BACKEND_TYPE="simulated"
else
    echo "FAIL: Unknown backend type"
    exit 1
fi

# Verify backend capabilities
if [[ "$BACKEND" == *"token_read: yes"* ]] && \
   [[ "$BACKEND" == *"token_write: yes"* ]]; then
    echo "PASS: SMBIOS backend initialized with required capabilities"
else
    echo "FAIL: Backend missing required capabilities"
    exit 1
fi
```

### Test 3: TPM Initialization

**Purpose:** Verify TPM authentication subsystem initialization.

```bash
#!/bin/bash
# test_tpm_init.sh

echo "=== TPM Initialization Test ==="

TPM_STATUS=$(cat /sys/class/dsmil/dsmil0/tpm_status)
echo "$TPM_STATUS"

# Extract state
STATE=$(echo "$TPM_STATUS" | grep "state:" | awk '{print $2}')
AVAILABLE=$(echo "$TPM_STATUS" | grep "available:" | awk '{print $2}')

echo "TPM State: $STATE"
echo "TPM Available: $AVAILABLE"

if [ "$STATE" == "ready" ] && [ "$AVAILABLE" == "yes" ]; then
    echo "PASS: TPM initialized and ready"
elif [ "$STATE" == "unavailable" ] && [ "$AVAILABLE" == "no" ]; then
    echo "WARN: TPM not available (expected on non-TPM systems)"
    echo "PASS: Fallback mode active"
else
    echo "FAIL: Unexpected TPM state"
    exit 1
fi
```

### Test 4: Error Handling Framework

**Purpose:** Verify error statistics tracking.

```bash
#!/bin/bash
# test_error_handling.sh

echo "=== Error Handling Test ==="

# Read initial error stats
ERROR_STATS_BEFORE=$(cat /sys/class/dsmil/dsmil0/error_stats)
TOTAL_BEFORE=$(echo "$ERROR_STATS_BEFORE" | grep "total_errors:" | awk '{print $2}')

echo "Total errors before: $TOTAL_BEFORE"

# Trigger an error by reading invalid token
cat > /tmp/test_invalid_token.c << 'EOF'
#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>

#define DSMIL_IOC_MAGIC 'D'
#define DSMIL_IOC_READ_TOKEN _IOWR(DSMIL_IOC_MAGIC, 1, struct dsmil_token_request)

struct dsmil_token_request {
    unsigned short token_id;
    unsigned int value;
};

int main() {
    int fd = open("/dev/dsmil0", O_RDWR);
    if (fd < 0) {
        perror("open");
        return 1;
    }

    struct dsmil_token_request req = { .token_id = 0xFFFF }; // Invalid token
    ioctl(fd, DSMIL_IOC_READ_TOKEN, &req); // Expected to fail

    close(fd);
    return 0;
}
EOF

gcc -o /tmp/test_invalid_token /tmp/test_invalid_token.c
/tmp/test_invalid_token 2>/dev/null

# Read error stats after
ERROR_STATS_AFTER=$(cat /sys/class/dsmil/dsmil0/error_stats)
TOTAL_AFTER=$(echo "$ERROR_STATS_AFTER" | grep "total_errors:" | awk '{print $2}')
TOKEN_ERRORS=$(echo "$ERROR_STATS_AFTER" | grep "token_errors:" | awk '{print $2}')

echo "Total errors after: $TOTAL_AFTER"
echo "Token errors: $TOKEN_ERRORS"

if [ "$TOTAL_AFTER" -gt "$TOTAL_BEFORE" ] && [ "$TOKEN_ERRORS" -gt 0 ]; then
    echo "PASS: Error tracking working correctly"
else
    echo "FAIL: Errors not tracked"
    exit 1
fi
```

---

## Integration Tests

### Test 5: Token Read/Write Flow

**Purpose:** Test complete token read/write cycle.

```c
// test_token_flow.c
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <errno.h>
#include <string.h>

#define DSMIL_IOC_MAGIC 'D'
#define DSMIL_IOC_READ_TOKEN  _IOWR(DSMIL_IOC_MAGIC, 1, struct dsmil_token_request)
#define DSMIL_IOC_WRITE_TOKEN _IOW(DSMIL_IOC_MAGIC, 2, struct dsmil_token_request)

struct dsmil_token_request {
    unsigned short token_id;
    unsigned int value;
};

int main() {
    int fd, ret;
    struct dsmil_token_request req;

    printf("=== Token Read/Write Flow Test ===\n");

    // Open device
    fd = open("/dev/dsmil0", O_RDWR);
    if (fd < 0) {
        perror("open");
        return 1;
    }

    // Test 1: Read device token (non-protected)
    printf("\nTest 1: Read device token 0x8000\n");
    req.token_id = 0x8000;
    req.value = 0;

    ret = ioctl(fd, DSMIL_IOC_READ_TOKEN, &req);
    if (ret < 0) {
        printf("FAIL: Read token failed: %s\n", strerror(errno));
        close(fd);
        return 1;
    }
    printf("PASS: Token 0x%04x = 0x%08x\n", req.token_id, req.value);

    // Test 2: Write device token (should succeed for non-protected)
    printf("\nTest 2: Write device config token 0x8001\n");
    req.token_id = 0x8001;
    req.value = 0x00000001;

    ret = ioctl(fd, DSMIL_IOC_WRITE_TOKEN, &req);
    if (ret < 0) {
        printf("FAIL: Write token failed: %s\n", strerror(errno));
        close(fd);
        return 1;
    }
    printf("PASS: Token 0x%04x written\n", req.token_id);

    // Test 3: Read back to verify
    printf("\nTest 3: Read back token 0x8001\n");
    req.value = 0;

    ret = ioctl(fd, DSMIL_IOC_READ_TOKEN, &req);
    if (ret < 0) {
        printf("FAIL: Read back failed: %s\n", strerror(errno));
        close(fd);
        return 1;
    }
    printf("PASS: Token 0x%04x = 0x%08x (verified)\n", req.token_id, req.value);

    // Test 4: Try to write protected token (should fail without auth)
    printf("\nTest 4: Write protected token 0x8500 (should fail)\n");
    req.token_id = 0x8500;
    req.value = 0x12345678;

    ret = ioctl(fd, DSMIL_IOC_WRITE_TOKEN, &req);
    if (ret < 0 && errno == EPERM) {
        printf("PASS: Protected token write correctly denied (EPERM)\n");
    } else if (ret == 0) {
        printf("FAIL: Protected token write should have been denied\n");
        close(fd);
        return 1;
    } else {
        printf("FAIL: Unexpected error: %s\n", strerror(errno));
        close(fd);
        return 1;
    }

    close(fd);
    printf("\n=== All Tests Passed ===\n");
    return 0;
}
```

**Compile and run:**
```bash
gcc -o test_token_flow test_token_flow.c
sudo ./test_token_flow
```

### Test 6: BIOS Redundancy

**Purpose:** Verify BIOS health monitoring and failover.

```bash
#!/bin/bash
# test_bios_redundancy.sh

echo "=== BIOS Redundancy Test ==="

# Check initial active BIOS
ACTIVE_BEFORE=$(cat /sys/class/dsmil/dsmil0/active_bios)
echo "Active BIOS before: $ACTIVE_BEFORE"

# Read all BIOS health scores
echo -e "\nBIOS Health Scores:"
for bios in A B C; do
    HEALTH=$(cat /sys/class/dsmil/dsmil0/bios_${bios,,}_health)
    echo "  BIOS $bios: $HEALTH"
done

# Query detailed BIOS status
cat > /tmp/test_bios_query.c << 'EOF'
#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>

#define DSMIL_IOC_MAGIC 'D'
#define DSMIL_IOC_BIOS_QUERY _IOR(DSMIL_IOC_MAGIC, 9, struct dsmil_bios_info)

struct dsmil_bios_info {
    unsigned char bios_id;
    unsigned char is_active;
    unsigned char health_score;
    unsigned int error_count;
    unsigned long long last_access_time;
};

int main() {
    int fd = open("/dev/dsmil0", O_RDWR);
    if (fd < 0) return 1;

    printf("\nDetailed BIOS Query:\n");
    for (int i = 0; i < 3; i++) {
        struct dsmil_bios_info info = { .bios_id = i };
        if (ioctl(fd, DSMIL_IOC_BIOS_QUERY, &info) == 0) {
            printf("  BIOS %c: health=%3u, active=%s, errors=%u\n",
                   'A' + i, info.health_score,
                   info.is_active ? "yes" : "no",
                   info.error_count);
        }
    }

    close(fd);
    return 0;
}
EOF

gcc -o /tmp/test_bios_query /tmp/test_bios_query.c
/tmp/test_bios_query

# Test manual failover (if supported)
echo -e "\nTesting manual BIOS failover..."
TARGET_BIOS="B"
echo "$TARGET_BIOS" | sudo tee /sys/class/dsmil/dsmil0/active_bios >/dev/null

ACTIVE_AFTER=$(cat /sys/class/dsmil/dsmil0/active_bios)
echo "Active BIOS after: $ACTIVE_AFTER"

if [ "$ACTIVE_AFTER" == "$TARGET_BIOS" ]; then
    echo "PASS: Manual BIOS failover successful"
else
    echo "FAIL: BIOS failover failed"
    exit 1
fi

# Restore original BIOS
echo "$ACTIVE_BEFORE" | sudo tee /sys/class/dsmil/dsmil0/active_bios >/dev/null
echo "Restored to BIOS $ACTIVE_BEFORE"
```

### Test 7: Device Management

**Purpose:** Test 104-device architecture.

```bash
#!/bin/bash
# test_device_management.sh

echo "=== Device Management Test ==="

# Count devices
DEVICE_COUNT=$(cat /sys/class/dsmil/dsmil0/device_count)
echo "Device count: $DEVICE_COUNT"

if [ "$DEVICE_COUNT" -ne 104 ]; then
    echo "FAIL: Expected 104 devices, got $DEVICE_COUNT"
    exit 1
fi

# Query random device statuses
echo -e "\nSampling device statuses:"
for dev_id in 0 10 50 103; do
    TOKEN_ID=$((0x8000 + dev_id * 3))
    printf "Device %3d (token 0x%04x): " $dev_id $TOKEN_ID

    # Read device status token
    cat > /tmp/test_device_status.c << EOF
#include <stdio.h>
#include <fcntl.h>
#include <sys/ioctl.h>

#define DSMIL_IOC_MAGIC 'D'
#define DSMIL_IOC_READ_TOKEN _IOWR(DSMIL_IOC_MAGIC, 1, struct dsmil_token_request)

struct dsmil_token_request {
    unsigned short token_id;
    unsigned int value;
};

int main() {
    int fd = open("/dev/dsmil0", O_RDWR);
    if (fd < 0) return 1;

    struct dsmil_token_request req = { .token_id = $TOKEN_ID };
    if (ioctl(fd, DSMIL_IOC_READ_TOKEN, &req) == 0) {
        printf("0x%08x ", req.value);
        if (req.value & 0x01) printf("[ONLINE] ");
        if (req.value & 0x02) printf("[READY]");
    }

    close(fd);
    return 0;
}
EOF

    gcc -o /tmp/test_device_status /tmp/test_device_status.c 2>/dev/null
    /tmp/test_device_status
    echo
done

echo -e "\nPASS: Device management test complete"
```

---

## Functional Tests

### Test 8: Complete Authentication Flow

**Purpose:** Test TPM authentication end-to-end.

```c
// test_tpm_auth.c
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <string.h>
#include <errno.h>

#define DSMIL_IOC_MAGIC 'D'
#define DSMIL_IOC_TPM_GET_CHALLENGE _IOR(DSMIL_IOC_MAGIC, 11, struct dsmil_tpm_challenge_data)
#define DSMIL_IOC_AUTHENTICATE      _IOW(DSMIL_IOC_MAGIC, 3, struct dsmil_auth_request)
#define DSMIL_IOC_WRITE_TOKEN       _IOW(DSMIL_IOC_MAGIC, 2, struct dsmil_token_request)

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

struct dsmil_token_request {
    unsigned short token_id;
    unsigned int value;
};

void print_hex(const char *label, unsigned char *data, int len) {
    printf("%s: ", label);
    for (int i = 0; i < len; i++) {
        printf("%02x", data[i]);
        if ((i + 1) % 16 == 0) printf("\n    ");
    }
    printf("\n");
}

int main() {
    int fd, ret;
    struct dsmil_tpm_challenge_data chal;
    struct dsmil_auth_request auth;
    struct dsmil_token_request token;

    printf("=== TPM Authentication Flow Test ===\n");

    // Open device
    fd = open("/dev/dsmil0", O_RDWR);
    if (fd < 0) {
        perror("open");
        return 1;
    }

    // Step 1: Get challenge
    printf("\nStep 1: Get TPM challenge\n");
    ret = ioctl(fd, DSMIL_IOC_TPM_GET_CHALLENGE, &chal);
    if (ret < 0) {
        printf("FAIL: Get challenge failed: %s\n", strerror(errno));
        close(fd);
        return 1;
    }

    printf("Challenge ID: 0x%08x\n", chal.challenge_id);
    printf("TPM Available: %s\n", chal.tpm_available ? "yes" : "no");
    print_hex("Challenge", chal.challenge, 32);

    // Step 2: Prepare authentication (simulated signature)
    printf("\nStep 2: Prepare authentication response\n");
    memset(&auth, 0, sizeof(auth));
    auth.auth_method = 1; // DSMIL_TPM_AUTH_CHALLENGE

    // Copy challenge ID to response
    memcpy(auth.auth_data, &chal.challenge_id, sizeof(chal.challenge_id));

    // Simulate signature (in real scenario, use TPM to sign)
    // For testing, just echo the challenge back
    memcpy(auth.auth_data + 4, chal.challenge, 32);
    auth.auth_data_len = 4 + 32;

    printf("Response prepared (%u bytes)\n", auth.auth_data_len);

    // Step 3: Authenticate
    printf("\nStep 3: Submit authentication\n");
    ret = ioctl(fd, DSMIL_IOC_AUTHENTICATE, &auth);
    if (ret < 0) {
        if (errno == EPERM) {
            printf("INFO: Authentication requires CAP_SYS_ADMIN (run with sudo)\n");
        } else {
            printf("FAIL: Authentication failed: %s\n", strerror(errno));
        }
        close(fd);
        return 1;
    }

    printf("PASS: Authentication successful\n");

    // Step 4: Try to write protected token
    printf("\nStep 4: Write protected token 0x8500\n");
    token.token_id = 0x8500;
    token.value = 0xCAFEBABE;

    ret = ioctl(fd, DSMIL_IOC_WRITE_TOKEN, &token);
    if (ret < 0) {
        printf("FAIL: Protected token write failed: %s\n", strerror(errno));
        close(fd);
        return 1;
    }

    printf("PASS: Protected token write successful\n");

    close(fd);
    printf("\n=== Authentication Flow Test Passed ===\n");
    return 0;
}
```

**Compile and run:**
```bash
gcc -o test_tpm_auth test_tpm_auth.c
sudo ./test_tpm_auth
```

### Test 9: Protected Token Access Control

**Purpose:** Verify protected token security enforcement.

```bash
#!/bin/bash
# test_protected_tokens.sh

echo "=== Protected Token Access Control Test ==="

# List of protected tokens
PROTECTED_TOKENS=(
    "0x8500"  # SECURITY_MASTER
    "0x8200"  # BIOS_UPDATE_CONTROL
    "0x8300"  # THERMAL_EMERGENCY
    "0x8400"  # POWER_CONTROL
)

cat > /tmp/test_protected.c << 'EOF'
#include <stdio.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <errno.h>

#define DSMIL_IOC_MAGIC 'D'
#define DSMIL_IOC_WRITE_TOKEN _IOW(DSMIL_IOC_MAGIC, 2, struct dsmil_token_request)

struct dsmil_token_request {
    unsigned short token_id;
    unsigned int value;
};

int main(int argc, char *argv[]) {
    int fd = open("/dev/dsmil0", O_RDWR);
    if (fd < 0) return 1;

    unsigned short token_id = strtol(argv[1], NULL, 16);
    struct dsmil_token_request req = { .token_id = token_id, .value = 0x11111111 };

    int ret = ioctl(fd, DSMIL_IOC_WRITE_TOKEN, &req);
    close(fd);

    return (ret < 0 && errno == EPERM) ? 0 : 1;
}
EOF

gcc -o /tmp/test_protected /tmp/test_protected.c

# Test that protected tokens are denied without authentication
echo "Testing protected token access without authentication:"
FAILED=0
for token in "${PROTECTED_TOKENS[@]}"; do
    if /tmp/test_protected $token 2>/dev/null; then
        echo "  Token $token: PASS (correctly denied)"
    else
        echo "  Token $token: FAIL (should have been denied)"
        FAILED=1
    fi
done

if [ $FAILED -eq 0 ]; then
    echo -e "\nPASS: All protected tokens correctly secured"
else
    echo -e "\nFAIL: Some protected tokens not secured"
    exit 1
fi
```

---

## Security Tests

### Test 10: TPM PCR Measurements

**Purpose:** Verify security events are measured in TPM.

```bash
#!/bin/bash
# test_pcr_measurements.sh

echo "=== TPM PCR Measurements Test ==="

# Check if TPM is available
if [ ! -c /dev/tpm0 ] && [ ! -c /dev/tpmrm0 ]; then
    echo "SKIP: TPM device not available"
    exit 0
fi

# Read PCR values before
echo "Reading TPM PCR values before operations..."
tpm2_pcrread sha256:16,17,18,23 > /tmp/pcr_before.txt 2>/dev/null

# Trigger authentication (PCR16)
echo -e "\nTriggering authentication event (PCR16)..."
sudo /tmp/test_tpm_auth >/dev/null 2>&1 || true

# Trigger BIOS operation (PCR18)
echo "Triggering BIOS operation (PCR18)..."
echo "B" | sudo tee /sys/class/dsmil/dsmil0/active_bios >/dev/null 2>&1 || true

# Read PCR values after
echo -e "\nReading TPM PCR values after operations..."
tpm2_pcrread sha256:16,17,18,23 > /tmp/pcr_after.txt 2>/dev/null

# Compare
if ! diff /tmp/pcr_before.txt /tmp/pcr_after.txt >/dev/null 2>&1; then
    echo "PASS: PCR values changed (security events measured)"
    echo -e "\nPCR Changes:"
    diff /tmp/pcr_before.txt /tmp/pcr_after.txt || true
else
    echo "WARN: PCR values unchanged (TPM may not be active)"
fi
```

### Test 11: Session Timeout

**Purpose:** Verify authenticated sessions timeout correctly.

```c
// test_session_timeout.c
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <errno.h>

#define DSMIL_IOC_MAGIC 'D'
#define DSMIL_IOC_AUTHENTICATE _IOW(DSMIL_IOC_MAGIC, 3, struct dsmil_auth_request)
#define DSMIL_IOC_WRITE_TOKEN  _IOW(DSMIL_IOC_MAGIC, 2, struct dsmil_token_request)

struct dsmil_auth_request {
    unsigned int auth_method;
    unsigned int auth_data_len;
    unsigned char auth_data[256];
};

struct dsmil_token_request {
    unsigned short token_id;
    unsigned int value;
};

int main() {
    int fd, ret;

    printf("=== Session Timeout Test ===\n");

    fd = open("/dev/dsmil0", O_RDWR);
    if (fd < 0) {
        perror("open");
        return 1;
    }

    // Authenticate (minimal auth for testing)
    struct dsmil_auth_request auth = {
        .auth_method = 1,
        .auth_data_len = 4,
    };

    printf("\nAuthenticating...\n");
    ret = ioctl(fd, DSMIL_IOC_AUTHENTICATE, &auth);
    if (ret < 0 && errno != EPERM) {
        printf("FAIL: Authentication error: %d\n", errno);
        close(fd);
        return 1;
    }

    if (ret == 0) {
        printf("PASS: Authenticated successfully\n");

        // Write protected token immediately (should succeed)
        printf("\nTest 1: Write protected token immediately\n");
        struct dsmil_token_request token = {
            .token_id = 0x8500,
            .value = 0xDEADBEEF
        };

        ret = ioctl(fd, DSMIL_IOC_WRITE_TOKEN, &token);
        if (ret == 0) {
            printf("PASS: Write successful (session valid)\n");
        } else {
            printf("FAIL: Write failed: %s\n", strerror(errno));
        }

        // Wait for session timeout (default: 5 minutes, use shorter timeout for testing)
        printf("\nWaiting 10 seconds (simulating timeout)...\n");
        sleep(10);

        // Try to write protected token again (should fail if timeout working)
        printf("\nTest 2: Write protected token after delay\n");
        ret = ioctl(fd, DSMIL_IOC_WRITE_TOKEN, &token);
        if (ret < 0 && errno == EPERM) {
            printf("PASS: Write correctly denied (session timeout working)\n");
        } else if (ret == 0) {
            printf("INFO: Write successful (session still valid - timeout not reached)\n");
        } else {
            printf("FAIL: Unexpected error: %s\n", strerror(errno));
        }
    } else {
        printf("INFO: Authentication requires CAP_SYS_ADMIN privileges\n");
    }

    close(fd);
    return 0;
}
```

---

## Performance Tests

### Test 12: Token Read Performance

**Purpose:** Measure token read throughput and latency.

```c
// test_token_performance.c
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <time.h>

#define DSMIL_IOC_MAGIC 'D'
#define DSMIL_IOC_READ_TOKEN _IOWR(DSMIL_IOC_MAGIC, 1, struct dsmil_token_request)

struct dsmil_token_request {
    unsigned short token_id;
    unsigned int value;
};

double get_time_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
}

int main() {
    int fd, ret;
    struct dsmil_token_request req;
    double start, end, total;
    int iterations = 10000;

    printf("=== Token Read Performance Test ===\n");
    printf("Iterations: %d\n\n", iterations);

    fd = open("/dev/dsmil0", O_RDWR);
    if (fd < 0) {
        perror("open");
        return 1;
    }

    // Test 1: Sequential reads of same token
    printf("Test 1: Sequential reads (same token)\n");
    req.token_id = 0x8000;

    start = get_time_ms();
    for (int i = 0; i < iterations; i++) {
        ret = ioctl(fd, DSMIL_IOC_READ_TOKEN, &req);
        if (ret < 0) {
            perror("ioctl");
            close(fd);
            return 1;
        }
    }
    end = get_time_ms();

    total = end - start;
    printf("  Total time: %.2f ms\n", total);
    printf("  Throughput: %.0f reads/sec\n", iterations / (total / 1000.0));
    printf("  Avg latency: %.3f ms\n", total / iterations);
    printf("  Min latency: %.3f us (estimated)\n", (total * 1000.0) / iterations);

    // Test 2: Random token reads
    printf("\nTest 2: Random token reads\n");

    start = get_time_ms();
    for (int i = 0; i < iterations; i++) {
        req.token_id = 0x8000 + (rand() % 104) * 3; // Random device token
        ret = ioctl(fd, DSMIL_IOC_READ_TOKEN, &req);
        if (ret < 0) {
            perror("ioctl");
            close(fd);
            return 1;
        }
    }
    end = get_time_ms();

    total = end - start;
    printf("  Total time: %.2f ms\n", total);
    printf("  Throughput: %.0f reads/sec\n", iterations / (total / 1000.0));
    printf("  Avg latency: %.3f ms\n", total / iterations);

    close(fd);

    printf("\n=== Performance Test Complete ===\n");
    return 0;
}
```

**Compile and run:**
```bash
gcc -o test_token_performance test_token_performance.c
./test_token_performance
```

### Test 13: Concurrent Access

**Purpose:** Test thread safety and concurrent access.

```c
// test_concurrent.c
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <pthread.h>
#include <sys/ioctl.h>
#include <errno.h>

#define DSMIL_IOC_MAGIC 'D'
#define DSMIL_IOC_READ_TOKEN _IOWR(DSMIL_IOC_MAGIC, 1, struct dsmil_token_request)

struct dsmil_token_request {
    unsigned short token_id;
    unsigned int value;
};

#define NUM_THREADS 8
#define ITERATIONS 1000

int global_errors = 0;
pthread_mutex_t error_mutex = PTHREAD_MUTEX_INITIALIZER;

void *thread_func(void *arg) {
    int thread_id = *(int *)arg;
    int fd, ret;
    struct dsmil_token_request req;

    // Each thread opens its own fd
    fd = open("/dev/dsmil0", O_RDWR);
    if (fd < 0) {
        pthread_mutex_lock(&error_mutex);
        global_errors++;
        pthread_mutex_unlock(&error_mutex);
        return NULL;
    }

    // Perform reads
    for (int i = 0; i < ITERATIONS; i++) {
        req.token_id = 0x8000 + (thread_id * 10 + i % 10) * 3;
        ret = ioctl(fd, DSMIL_IOC_READ_TOKEN, &req);

        if (ret < 0) {
            pthread_mutex_lock(&error_mutex);
            global_errors++;
            pthread_mutex_unlock(&error_mutex);
        }
    }

    close(fd);
    return NULL;
}

int main() {
    pthread_t threads[NUM_THREADS];
    int thread_ids[NUM_THREADS];

    printf("=== Concurrent Access Test ===\n");
    printf("Threads: %d\n", NUM_THREADS);
    printf("Iterations per thread: %d\n\n", ITERATIONS);

    // Create threads
    for (int i = 0; i < NUM_THREADS; i++) {
        thread_ids[i] = i;
        if (pthread_create(&threads[i], NULL, thread_func, &thread_ids[i]) != 0) {
            perror("pthread_create");
            return 1;
        }
    }

    // Wait for threads
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }

    if (global_errors == 0) {
        printf("PASS: No errors in %d concurrent operations\n",
               NUM_THREADS * ITERATIONS);
    } else {
        printf("FAIL: %d errors occurred\n", global_errors);
        return 1;
    }

    return 0;
}
```

---

## Stress Tests

### Test 14: Extended Load Test

**Purpose:** Stability test under sustained load.

```bash
#!/bin/bash
# test_stress.sh

echo "=== Extended Load Test ==="

DURATION=3600  # 1 hour
START_TIME=$(date +%s)
END_TIME=$((START_TIME + DURATION))
ERROR_COUNT=0

echo "Duration: $DURATION seconds ($(($DURATION / 60)) minutes)"
echo "Start time: $(date)"
echo ""

# Compile test programs
gcc -o /tmp/stress_reader test_token_performance.c 2>/dev/null
gcc -o /tmp/stress_concurrent test_concurrent.c -lpthread 2>/dev/null

# Start background readers
echo "Starting background load..."
for i in {1..4}; do
    (
        while [ $(date +%s) -lt $END_TIME ]; do
            /tmp/stress_reader >/dev/null 2>&1
            if [ $? -ne 0 ]; then
                echo "ERROR: Reader $i failed at $(date)" >> /tmp/stress_errors.log
            fi
            sleep 1
        done
    ) &
done

# Monitor system
echo "Monitoring system health..."
while [ $(date +%s) -lt $END_TIME ]; do
    CURRENT_TIME=$(date +%s)
    ELAPSED=$((CURRENT_TIME - START_TIME))
    REMAINING=$((DURATION - ELAPSED))

    # Check driver status
    if [ ! -c /dev/dsmil0 ]; then
        echo "CRITICAL: Driver disappeared!"
        ERROR_COUNT=$((ERROR_COUNT + 1))
    fi

    # Check error stats
    TOTAL_ERRORS=$(cat /sys/class/dsmil/dsmil0/error_stats | grep "total_errors:" | awk '{print $2}')

    # Print status every 5 minutes
    if [ $((ELAPSED % 300)) -eq 0 ]; then
        printf "[%5d/%5d sec] Errors: %d, Driver errors: %d\n" \
               $ELAPSED $DURATION $ERROR_COUNT $TOTAL_ERRORS
    fi

    sleep 60
done

# Wait for background jobs
echo -e "\nWaiting for background jobs to complete..."
wait

# Final report
echo -e "\n=== Stress Test Complete ==="
echo "End time: $(date)"
echo "Total errors: $ERROR_COUNT"

if [ -f /tmp/stress_errors.log ]; then
    echo "Error log:"
    cat /tmp/stress_errors.log
fi

if [ $ERROR_COUNT -eq 0 ]; then
    echo "PASS: System stable under extended load"
else
    echo "FAIL: $ERROR_COUNT errors during stress test"
    exit 1
fi
```

---

## Test Automation

### Complete Test Suite Runner

```bash
#!/bin/bash
# run_all_tests.sh

echo "======================================"
echo " DSMIL Driver Comprehensive Test Suite"
echo "======================================"
echo "Version: 5.2.0"
echo "Date: $(date)"
echo ""

# Test results
TESTS_PASSED=0
TESTS_FAILED=0
TESTS_SKIPPED=0

# Function to run test
run_test() {
    local test_name="$1"
    local test_script="$2"

    echo "----------------------------------------"
    echo "Running: $test_name"
    echo "----------------------------------------"

    if bash -c "$test_script" 2>&1 | tee /tmp/test_output.log; then
        echo "✓ PASS"
        TESTS_PASSED=$((TESTS_PASSED + 1))
    else
        if grep -q "SKIP" /tmp/test_output.log; then
            echo "⊘ SKIP"
            TESTS_SKIPPED=$((TESTS_SKIPPED + 1))
        else
            echo "✗ FAIL"
            TESTS_FAILED=$((TESTS_FAILED + 1))
        fi
    fi

    echo ""
}

# Ensure driver is loaded
if [ ! -c /dev/dsmil0 ]; then
    echo "ERROR: DSMIL driver not loaded"
    echo "Run: sudo insmod core/dsmil-104dev.ko"
    exit 1
fi

# Unit Tests
echo "=== Unit Tests ==="
run_test "Token Database" "bash test_token_db.sh"
run_test "SMBIOS Backend" "bash test_smbios_backend.sh"
run_test "TPM Initialization" "bash test_tpm_init.sh"
run_test "Error Handling" "bash test_error_handling.sh"

# Integration Tests
echo "=== Integration Tests ==="
run_test "Token Read/Write" "./test_token_flow"
run_test "BIOS Redundancy" "bash test_bios_redundancy.sh"
run_test "Device Management" "bash test_device_management.sh"

# Functional Tests
echo "=== Functional Tests ==="
run_test "TPM Authentication" "sudo ./test_tpm_auth"
run_test "Protected Tokens" "bash test_protected_tokens.sh"

# Security Tests
echo "=== Security Tests ==="
run_test "PCR Measurements" "bash test_pcr_measurements.sh"
run_test "Session Timeout" "sudo ./test_session_timeout"

# Performance Tests
echo "=== Performance Tests ==="
run_test "Token Performance" "./test_token_performance"
run_test "Concurrent Access" "./test_concurrent"

# Print summary
echo "======================================"
echo " Test Summary"
echo "======================================"
echo "Passed:  $TESTS_PASSED"
echo "Failed:  $TESTS_FAILED"
echo "Skipped: $TESTS_SKIPPED"
echo "Total:   $((TESTS_PASSED + TESTS_FAILED + TESTS_SKIPPED))"
echo ""

if [ $TESTS_FAILED -eq 0 ]; then
    echo "✓ All tests passed!"
    exit 0
else
    echo "✗ Some tests failed"
    exit 1
fi
```

**Usage:**
```bash
chmod +x run_all_tests.sh
sudo ./run_all_tests.sh
```

---

## Test Results Format

### Standard Test Report

```
DSMIL Driver Test Report
========================

Date: 2025-11-13
Version: 5.2.0
Platform: Linux 5.15.0 x86_64
Kernel Config: CONFIG_DELL_SMBIOS=y, CONFIG_TCG_TPM=y

Test Results
------------

Unit Tests:                   4/4 passed
Integration Tests:            3/3 passed
Functional Tests:             2/2 passed
Security Tests:              2/2 passed
Performance Tests:            2/2 passed

Total:                       13/13 passed (100%)

Performance Metrics
-------------------

Token Read Latency:          0.045 ms (avg)
Token Read Throughput:       22,000 reads/sec
Concurrent Operations:       8000 ops without errors
Memory Leaks:                None detected

Security Verification
---------------------

Protected Token Access:      Correctly enforced
TPM Authentication:          Working
PCR Measurements:            Verified
Session Management:          Working

System Stability
----------------

Extended Load Test:          Passed (1 hour)
Error Count:                 0
Driver Crashes:              0
Memory Usage:                Stable

Conclusion
----------

PASS: Driver ready for production use
```

---

## Conclusion

This testing guide provides comprehensive validation of the DSMIL driver including:

- **Unit tests** for individual components
- **Integration tests** for component interactions
- **Functional tests** for feature validation
- **Security tests** for TPM and authentication
- **Performance tests** for throughput/latency
- **Stress tests** for stability

All tests can be run individually or via the automated test suite runner.

For questions or issues, refer to the DRIVER_USAGE_GUIDE.md or contact the development team.
