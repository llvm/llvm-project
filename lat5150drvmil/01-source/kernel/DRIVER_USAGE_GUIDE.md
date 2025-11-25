# DSMIL Driver Usage Guide
**Version 5.2.0**

Complete guide to using the Dell MIL-SPEC 104-Device DSMIL Driver with TPM 2.0 authentication.

---

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Module Parameters](#module-parameters)
4. [IOCTL Interface](#ioctl-interface)
5. [Sysfs Interface](#sysfs-interface)
6. [TPM Authentication](#tpm-authentication)
7. [Protected Tokens](#protected-tokens)
8. [BIOS Management](#bios-management)
9. [Example Programs](#example-programs)
10. [Troubleshooting](#troubleshooting)

---

## Overview

The DSMIL driver provides comprehensive management for:
- **104 DSMIL devices** across 9 groups
- **3 redundant BIOS systems** (A/B/C) with automatic failover
- **500+ tokens** for device/system configuration
- **TPM 2.0 authentication** for protected operations
- **Real-time monitoring** and audit logging

### Key Features

- ✅ Hardware-backed TPM 2.0 authentication
- ✅ Automatic SMBIOS backend selection (real or simulated)
- ✅ Comprehensive error handling and audit logging
- ✅ BIOS health monitoring with automatic failover
- ✅ Token caching for performance (Red-Black tree)
- ✅ Protected token access control
- ✅ Sysfs monitoring interface

---

## Installation

### Prerequisites

- Linux kernel 6.14.0 or later
- Kernel headers installed: `linux-headers-$(uname -r)`
- Build tools: `gcc`, `make`, `kmod`
- (Optional) TPM 2.0 chip for hardware authentication

### Build Driver

```bash
cd /path/to/LAT5150DRVMIL/01-source/kernel
make
```

### Load Driver

**Basic load** (TPM optional):
```bash
sudo insmod dsmil-104dev.ko
```

**With module parameters**:
```bash
# Require TPM (fail if TPM unavailable)
sudo insmod dsmil-104dev.ko require_tpm=1

# Disable automatic BIOS failover
sudo insmod dsmil-104dev.ko enable_bios_failover=0

# Custom health threshold
sudo insmod dsmil-104dev.ko bios_health_critical=40
```

### Verify Installation

```bash
# Check driver loaded
lsmod | grep dsmil

# Check device created
ls -l /dev/dsmil-104dev

# View driver info
dmesg | grep DSMIL
```

Expected output:
```
DSMIL: Driver loaded successfully
DSMIL: - 104 devices across 9 groups
DSMIL: - Active BIOS: A (Health: 90)
DSMIL: - Failover: Enabled
DSMIL: - Token count: 50
DSMIL: - SMBIOS backend: dell-smbios (kernel subsystem)
DSMIL: - TPM: Available
```

---

## Module Parameters

### Available Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `auto_discover_tokens` | bool | `true` | Auto-discover tokens on load |
| `enable_bios_failover` | bool | `true` | Enable automatic BIOS failover |
| `bios_health_critical` | uint | `30` | Health score for failover (0-100) |
| `thermal_threshold` | uint | `90` | Thermal shutdown threshold (°C) |
| `enable_protected_tokens` | bool | `true` | Enable protected token access |
| `require_tpm` | bool | `false` | Require TPM (fail if unavailable) |

### Examples

**High-security configuration**:
```bash
sudo insmod dsmil-104dev.ko require_tpm=1 bios_health_critical=50
```

**Development configuration**:
```bash
sudo insmod dsmil-104dev.ko require_tpm=0 enable_bios_failover=0
```

---

## IOCTL Interface

### Opening the Device

```c
#include <fcntl.h>
#include <sys/ioctl.h>

int fd = open("/dev/dsmil-104dev", O_RDWR);
if (fd < 0) {
    perror("Failed to open DSMIL device");
    return -1;
}
```

### IOCTL Commands

#### 1. Get Driver Version

```c
#define DSMIL_IOC_GET_VERSION  _IOR('D', 1, __u32)

__u32 version;
if (ioctl(fd, DSMIL_IOC_GET_VERSION, &version) == 0) {
    printf("Driver version: %u.%u.%u\n",
           (version >> 16) & 0xFF,
           (version >> 8) & 0xFF,
           version & 0xFF);
}
```

#### 2. Get System Status

```c
#define DSMIL_IOC_GET_STATUS   _IOR('D', 2, struct dsmil_system_status)

struct dsmil_system_status {
    __u32 driver_version;
    __u32 device_count;
    __u32 group_count;
    __u32 active_bios;
    __u32 bios_health_a;
    __u32 bios_health_b;
    __u32 bios_health_c;
    __u32 thermal_celsius;
    __u8  authenticated;
    __u8  failover_enabled;
};

struct dsmil_system_status status;
if (ioctl(fd, DSMIL_IOC_GET_STATUS, &status) == 0) {
    printf("Devices: %u, Active BIOS: %c, Health: %u\n",
           status.device_count,
           'A' + status.active_bios,
           status.bios_health_a);
}
```

#### 3. Read Token

```c
#define DSMIL_IOC_READ_TOKEN   _IOWR('D', 3, struct dsmil_token_op)

struct dsmil_token_op {
    __u16 token_id;
    __u32 value;
    __u32 result;
};

struct dsmil_token_op op = {
    .token_id = 0x8000,  // Device 0 status
};

if (ioctl(fd, DSMIL_IOC_READ_TOKEN, &op) == 0) {
    printf("Token 0x%04x = 0x%08x\n", op.token_id, op.value);
} else {
    printf("Read failed: %d\n", op.result);
}
```

#### 4. Write Token (Unprotected)

```c
#define DSMIL_IOC_WRITE_TOKEN  _IOWR('D', 4, struct dsmil_token_op)

struct dsmil_token_op op = {
    .token_id = 0x8001,  // Device 0 config
    .value = 0x00000001,
};

if (ioctl(fd, DSMIL_IOC_WRITE_TOKEN, &op) == 0) {
    printf("Token written successfully\n");
} else {
    printf("Write failed: %d\n", op.result);
}
```

#### 5. Get Device Info

```c
#define DSMIL_IOC_GET_DEVICE_INFO  _IOWR('D', 6, struct dsmil_device_info)

struct dsmil_device_info info = {
    .device_id = 5,  // Query device 5
};

if (ioctl(fd, DSMIL_IOC_GET_DEVICE_INFO, &info) == 0) {
    printf("Device %u: Group %u, Token base 0x%04x\n",
           info.device_id, info.group_id, info.token_base);
}
```

#### 6. Get BIOS Status

```c
#define DSMIL_IOC_GET_BIOS_STATUS  _IOR('D', 7, struct dsmil_bios_status)

struct dsmil_bios_status {
    __u32 bios_a_status;
    __u32 bios_b_status;
    __u32 bios_c_status;
    __u32 active_bios;
    __u32 boot_count_a;
    __u32 boot_count_b;
    __u32 boot_count_c;
    __u32 error_count_a;
    __u32 error_count_b;
    __u32 error_count_c;
};

struct dsmil_bios_status bios;
if (ioctl(fd, DSMIL_IOC_GET_BIOS_STATUS, &bios) == 0) {
    printf("Active: %c, Boots: A=%u B=%u C=%u\n",
           'A' + bios.active_bios,
           bios.boot_count_a,
           bios.boot_count_b,
           bios.boot_count_c);
}
```

#### 7. BIOS Failover (Admin only)

```c
#define DSMIL_IOC_BIOS_FAILOVER  _IOW('D', 8, enum dsmil_bios_id)

enum dsmil_bios_id {
    DSMIL_BIOS_A = 0,
    DSMIL_BIOS_B = 1,
    DSMIL_BIOS_C = 2,
};

enum dsmil_bios_id target = DSMIL_BIOS_B;

// Requires CAP_SYS_ADMIN
if (ioctl(fd, DSMIL_IOC_BIOS_FAILOVER, &target) == 0) {
    printf("Failed over to BIOS B\n");
} else {
    perror("Failover failed");
}
```

---

## Sysfs Interface

### Monitoring Attributes

All sysfs attributes are under: `/sys/class/dsmil-104dev/dsmil-104dev/`

#### Device Information

```bash
# Number of devices
cat device_count
# Output: 104

# Number of groups
cat group_count
# Output: 9

# Token count
cat token_count
# Output: 50
```

#### BIOS Status

```bash
# Active BIOS
cat active_bios
# Output: A

# BIOS health scores
cat bios_health
# Output: A:90 B:85 C:95

# Failover count
cat failover_count
# Output: 3
```

#### Statistics

```bash
# Token operations
cat token_reads
# Output: 1524

cat token_writes
# Output: 87
```

#### Error Statistics

```bash
cat error_stats
```

Output:
```
token_errors:      0
device_errors:     0
bios_errors:       1
auth_errors:       2
security_errors:   0
smbios_errors:     0
validation_errors: 3
thermal_errors:    0
total_errors:      6
last_error_code:   0x1003
```

#### Last Audit Entry

```bash
cat last_audit
```

Output:
```
timestamp:   1704067200000000000
event_type:  2
user_id:     1000
token_id:    0x8209
old_value:   0x00000000
new_value:   0x00000001
result:      0
message:     Protected token write authorized
```

#### SMBIOS Backend

```bash
cat smbios_backend
```

Output:
```
backend:         dell-smbios (kernel subsystem)
type:            2
token_read:      yes
token_write:     yes
token_discovery: yes
wmi_support:     yes
smm_support:     yes
buffer_size:     32
```

#### TPM Status

```bash
cat tpm_status
```

Output:
```
state:           ready
available:       yes
chip_present:    yes
auth_mode:       challenge
session_active:  yes
auth_attempts:   5
auth_successes:  4
auth_failures:   1
pcr_extends:     127
```

---

## TPM Authentication

### Overview

Protected tokens require TPM 2.0 hardware-backed authentication using a challenge-response protocol.

### Authentication Flow

1. **Get Challenge** - Request authentication challenge from driver
2. **Sign Challenge** - Sign challenge using TPM or external credential
3. **Submit Response** - Submit signed response for validation
4. **Access Protected Tokens** - Use authenticated session (5 min default)

### Step 1: Get Challenge

```c
#define DSMIL_IOC_TPM_GET_CHALLENGE  _IOR('D', 11, struct dsmil_tpm_challenge_data)

struct dsmil_tpm_challenge_data {
    __u8  challenge[32];
    __u32 challenge_id;
    __u8  tpm_available;
};

struct dsmil_tpm_challenge_data chal;
if (ioctl(fd, DSMIL_IOC_TPM_GET_CHALLENGE, &chal) == 0) {
    printf("Challenge ID: 0x%08x\n", chal.challenge_id);
    printf("TPM available: %s\n", chal.tpm_available ? "yes" : "no");

    // Print challenge (hex)
    printf("Challenge: ");
    for (int i = 0; i < 32; i++) {
        printf("%02x", chal.challenge[i]);
    }
    printf("\n");
}
```

### Step 2: Sign Challenge

**Option A: Using TPM Tools** (recommended):

```bash
# Save challenge to file
echo "challenge_data" > /tmp/challenge.bin

# Sign using TPM
tpm2_sign -c 0x81000001 -g sha256 -o /tmp/signature.bin /tmp/challenge.bin

# Or using tpm2-tools
tpm2_sign --key-context 0x81000001 --hash-algorithm sha256 \
          --signature /tmp/signature.bin /tmp/challenge.bin
```

**Option B: External Signing**:

```c
// Use your own signing implementation
// (OpenSSL, libcrypto, hardware HSM, etc.)

unsigned char signature[256];
int sig_len = sign_with_key(chal.challenge, 32, signature);
```

### Step 3: Submit Response

```c
#define DSMIL_IOC_AUTHENTICATE  _IOW('D', 10, struct dsmil_auth_request)

struct dsmil_auth_request {
    __u32 auth_method;  // 1 = challenge-response
    __u8  auth_data[256];
    __u32 auth_data_len;
};

struct dsmil_auth_request req;
req.auth_method = 1;  // DSMIL_TPM_AUTH_CHALLENGE

// Format: [challenge_id (4 bytes)][signature (remaining bytes)]
memcpy(req.auth_data, &chal.challenge_id, sizeof(chal.challenge_id));
memcpy(req.auth_data + 4, signature, sig_len);
req.auth_data_len = 4 + sig_len;

if (ioctl(fd, DSMIL_IOC_AUTHENTICATE, &req) == 0) {
    printf("Authentication successful!\n");
    printf("Session valid for 5 minutes\n");
} else {
    perror("Authentication failed");
}
```

### Step 4: Access Protected Tokens

```c
// Now you can write protected tokens
struct dsmil_token_op op = {
    .token_id = 0x8209,  // SYSTEM_RESET (protected)
    .value = 1,
};

if (ioctl(fd, DSMIL_IOC_WRITE_TOKEN, &op) == 0) {
    printf("Protected token write successful\n");
} else {
    printf("Write failed: %d\n", op.result);
}
```

### Logout

```c
#define DSMIL_IOC_TPM_INVALIDATE  _IO('D', 12)

// Invalidate session (logout)
ioctl(fd, DSMIL_IOC_TPM_INVALIDATE);
printf("Session invalidated\n");
```

### Fallback Mode (No TPM)

When TPM is unavailable, authentication falls back to **CAP_SYS_ADMIN only**:

```c
// Still need to authenticate, but validation is simplified
struct dsmil_auth_request req = {0};
req.auth_method = 1;
req.auth_data_len = 4;

// Validation will succeed if you have CAP_SYS_ADMIN
if (ioctl(fd, DSMIL_IOC_AUTHENTICATE, &req) == 0) {
    printf("Authenticated (no TPM, capability only)\n");
}
```

---

## Protected Tokens

### Protected Token List

These tokens require authentication and CAP_SYS_ADMIN:

| Token ID | Name | Description |
|----------|------|-------------|
| 0x8209 | SYSTEM_RESET | Full system reset command |
| 0x820A | SECURE_ERASE | Secure data erase |
| 0x820B | FACTORY_RESET | Factory reset command |
| 0x8401 | NETWORK_KILLSWITCH | Emergency network disable |
| 0x8605 | DATA_WIPE | Secure data wipe |
| 0x811F | BIOS_A_CONTROL | BIOS A control register |
| 0x812F | BIOS_B_CONTROL | BIOS B control register |
| 0x813F | BIOS_C_CONTROL | BIOS C control register |

### Example: System Reset

```c
// 1. Authenticate
authenticate_with_tpm(fd);

// 2. Write protected token
struct dsmil_token_op op = {
    .token_id = 0x8209,  // SYSTEM_RESET
    .value = 1,          // Trigger reset
};

int ret = ioctl(fd, DSMIL_IOC_WRITE_TOKEN, &op);
if (ret == 0) {
    printf("System reset initiated\n");
} else if (ret == -EPERM) {
    printf("Error: Insufficient privileges\n");
} else if (ret == -EACCES) {
    printf("Error: Authentication required\n");
}
```

---

## BIOS Management

### Query BIOS Health

```bash
# Via sysfs
cat /sys/class/dsmil-104dev/dsmil-104dev/bios_health
# Output: A:90 B:85 C:95
```

```c
// Via IOCTL
struct dsmil_bios_status bios;
ioctl(fd, DSMIL_IOC_GET_BIOS_STATUS, &bios);

printf("BIOS A health: %u/100\n", bios.bios_a_status & 0xFF);
printf("BIOS B health: %u/100\n", bios.bios_b_status & 0xFF);
printf("BIOS C health: %u/100\n", bios.bios_c_status & 0xFF);
```

### Manual Failover

```c
// Requires CAP_SYS_ADMIN
enum dsmil_bios_id target = DSMIL_BIOS_C;

if (ioctl(fd, DSMIL_IOC_BIOS_FAILOVER, &target) == 0) {
    printf("Successfully failed over to BIOS C\n");
} else {
    perror("Failover failed");
}
```

### BIOS Synchronization

```c
#define DSMIL_IOC_BIOS_SYNC  _IOW('D', 9, struct dsmil_bios_sync_request)

struct dsmil_bios_sync_request {
    enum dsmil_bios_id source;
    enum dsmil_bios_id target;
    __u32 flags;
};

// Sync BIOS A → BIOS B
struct dsmil_bios_sync_request req = {
    .source = DSMIL_BIOS_A,
    .target = DSMIL_BIOS_B,
    .flags = 0,
};

if (ioctl(fd, DSMIL_IOC_BIOS_SYNC, &req) == 0) {
    printf("BIOS sync initiated\n");
}
```

### Automatic Failover

Automatic failover triggers when BIOS health drops below threshold:

```bash
# Set low threshold for testing
sudo rmmod dsmil-104dev
sudo insmod dsmil-104dev.ko bios_health_critical=40

# Monitor failover events
dmesg -w | grep "BIOS failover"
```

---

## Example Programs

### Complete Example: Read Device Status

```c
#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>

#define DSMIL_IOC_READ_TOKEN  _IOWR('D', 3, struct dsmil_token_op)

struct dsmil_token_op {
    __u16 token_id;
    __u32 value;
    __u32 result;
};

int main() {
    int fd = open("/dev/dsmil-104dev", O_RDWR);
    if (fd < 0) {
        perror("Failed to open device");
        return 1;
    }

    // Read device 0 status (token 0x8000)
    struct dsmil_token_op op = {
        .token_id = 0x8000,
    };

    if (ioctl(fd, DSMIL_IOC_READ_TOKEN, &op) == 0) {
        printf("Device 0 status: 0x%08x\n", op.value);
        printf("  Online: %s\n", (op.value & 0x01) ? "yes" : "no");
        printf("  Ready:  %s\n", (op.value & 0x02) ? "yes" : "no");
    } else {
        printf("Read failed\n");
    }

    close(fd);
    return 0;
}
```

Compile and run:
```bash
gcc -o read_device read_device.c
sudo ./read_device
```

---

## Troubleshooting

### Driver Won't Load

**Problem**: `insmod: ERROR: could not insert module`

**Solutions**:
```bash
# Check kernel version
uname -r  # Must be >= 6.14.0

# Check dmesg for errors
dmesg | tail -20

# Verify kernel headers
ls /lib/modules/$(uname -r)/build

# Try verbose load
sudo insmod dsmil-104dev.ko
dmesg | grep DSMIL
```

### Device Not Created

**Problem**: `/dev/dsmil-104dev` doesn't exist

**Solutions**:
```bash
# Check if driver loaded
lsmod | grep dsmil

# Check device major/minor
ls -l /sys/class/dsmil-104dev/dsmil-104dev/dev

# Manually create device
sudo mknod /dev/dsmil-104dev c 240 0
sudo chmod 666 /dev/dsmil-104dev
```

### TPM Not Available

**Problem**: `tpm_status` shows `unavailable`

**Solutions**:
```bash
# Check TPM device
ls /dev/tpm*

# Check TPM version
cat /sys/class/tpm/tpm0/tpm_version_major
# Should be "2" for TPM 2.0

# Verify TPM tools
tpm2_getcap properties-fixed

# Try reload without requiring TPM
sudo rmmod dsmil-104dev
sudo insmod dsmil-104dev.ko require_tpm=0
```

### Authentication Fails

**Problem**: `DSMIL_IOC_AUTHENTICATE` returns error

**Solutions**:
```bash
# Check TPM status
cat /sys/class/dsmil-104dev/dsmil-104dev/tpm_status

# Verify capabilities
sudo -l  # Should show CAP_SYS_ADMIN

# Check audit log
cat /sys/class/dsmil-104dev/dsmil-104dev/last_audit

# Try simplified auth (no signature)
# Works in fallback mode
```

### Permission Denied

**Problem**: IOCTL returns `-EPERM` or `-EACCES`

**Solutions**:
```bash
# Check device permissions
ls -l /dev/dsmil-104dev
# Should be: crw-rw-rw-

# Fix permissions
sudo chmod 666 /dev/dsmil-104dev

# Run as root for protected tokens
sudo ./your_program

# Check if authenticated
cat /sys/class/dsmil-104dev/dsmil-104dev/error_stats
# Look for auth_errors
```

### High Error Count

**Problem**: `error_stats` shows many errors

**Solutions**:
```bash
# View detailed errors
cat /sys/class/dsmil-104dev/dsmil-104dev/error_stats

# Check system log
dmesg | grep DSMIL | grep -i error

# Check last audit
cat /sys/class/dsmil-104dev/dsmil-104dev/last_audit

# Reset driver
sudo rmmod dsmil-104dev
sudo insmod dsmil-104dev.ko
```

---

## Additional Resources

- **Driver Source**: `/path/to/LAT5150DRVMIL/01-source/kernel/`
- **Testing Guide**: `TESTING_GUIDE.md`
- **API Reference**: `API_REFERENCE.md`
- **TPM Guide**: `TPM_AUTHENTICATION_GUIDE.md`

For bug reports or questions, contact the DSMIL development team.

---

**Document Version**: 1.0
**Driver Version**: 5.2.0
**Last Updated**: 2025-01-13
