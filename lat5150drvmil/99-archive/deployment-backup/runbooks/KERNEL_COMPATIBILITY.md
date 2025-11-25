# KERNEL COMPATIBILITY RUNBOOK

**Dell MIL-SPEC Platform Operations**
**Document Type**: Maintenance Procedure
**Version**: 1.0.0
**Last Updated**: 2025-10-11
**Classification**: UNCLASSIFIED // FOR OFFICIAL USE ONLY

---

## PURPOSE

This runbook provides procedures for testing, validating, and maintaining Dell MIL-SPEC DKMS packages across Linux kernel versions. DKMS (Dynamic Kernel Module Support) automatically rebuilds kernel modules, but new kernels may require compatibility updates.

**GOAL**: Ensure seamless operation across kernel versions 6.14.0 through 7.0.0+

---

## KERNEL COMPATIBILITY MATRIX

### Currently Supported Kernels

| Kernel Version | DSMIL DKMS | TPM2 DKMS | Status | Notes |
|----------------|------------|-----------|--------|-------|
| 6.14.0 | ✅ 2.1.1 | ✅ 1.0.0 | STABLE | Initial release |
| 6.15.x | ✅ 2.1.1 | ✅ 1.0.0 | STABLE | Tested |
| 6.16.x | ✅ 2.1.1 | ✅ 1.0.0 | STABLE | Production |
| 6.17.x | ⏳ TBD | ⏳ TBD | TESTING | This runbook |
| 6.18.x | ❓ Unknown | ❓ Unknown | FUTURE | |
| 7.0.x | ❓ Unknown | ❓ Unknown | FUTURE | |

### Minimum Kernel Requirements

**dell-milspec-dsmil-dkms**:
- Minimum: Linux 6.14.0
- Required features:
  - `io_uring` support
  - `ioctl_ops` structure
  - `cdev` character device API
  - Dell SMBIOS infrastructure

**tpm2-accel-early-dkms**:
- Minimum: Linux 6.14.0
- Required features:
  - TPM 2.0 subsystem
  - `subsys_initcall_sync` early boot
  - Intel NPU/GNA drivers (optional)
  - Initramfs integration

---

## SCENARIO 1: New Kernel Version Released

### Trigger
A new Linux kernel version is released (e.g., 6.17.0) and needs testing with Dell MIL-SPEC packages.

### Timeline
**Total Time**: 2-4 hours (automated) or 1 day (manual fixes required)

---

## PHASE 1: DETECTION & PREPARATION (15 minutes)

### Step 1.1: Kernel Release Detection

**Automated Detection** (CI/CD):
```yaml
# .github/workflows/kernel-compatibility-check.yml
name: Kernel Compatibility Check

on:
  schedule:
    - cron: '0 6 * * *'  # Daily at 6 AM
  workflow_dispatch:

jobs:
  check-new-kernels:
    runs-on: ubuntu-latest
    steps:
      - name: Check for new kernel versions
        run: |
          # Query kernel.org
          LATEST_STABLE=$(curl -s https://www.kernel.org/ | \
            grep -oP 'stable:\s+\K\d+\.\d+\.\d+' | head -1)

          echo "Latest stable kernel: $LATEST_STABLE"

          # Check if we've tested this version
          if ! grep -q "$LATEST_STABLE" deployment/docs/kernel-compatibility.txt; then
            echo "NEW_KERNEL=$LATEST_STABLE" >> $GITHUB_ENV
            echo "New kernel detected: $LATEST_STABLE"
          fi
```

**Manual Detection**:
```bash
# Check kernel.org for latest stable
curl -s https://www.kernel.org/ | grep -oP 'stable:\s+\K\d+\.\d+\.\d+' | head -1

# Check your distribution
apt-cache policy linux-image-generic | grep Candidate

# Check what's installed
uname -r
```

- [ ] New kernel version detected: `________________`
- [ ] Release notes reviewed: `________________`

### Step 1.2: Gather Kernel Information

```bash
# Download kernel release notes
NEW_KERNEL="6.17.0"
mkdir -p /tmp/kernel-compat-${NEW_KERNEL}
cd /tmp/kernel-compat-${NEW_KERNEL}

# Get changelog
curl -o changelog-${NEW_KERNEL}.txt \
  https://cdn.kernel.org/pub/linux/kernel/v6.x/ChangeLog-${NEW_KERNEL}

# Check for API changes relevant to our modules
grep -iE "ioctl|cdev|tpm|dmi|smbios|character device" changelog-${NEW_KERNEL}.txt > relevant-changes.txt
```

**Key Areas to Check**:
- [ ] Character device API changes
- [ ] IOCTL interface changes
- [ ] TPM subsystem changes
- [ ] DMI/SMBIOS changes
- [ ] Early boot infrastructure changes
- [ ] DKMS compatibility changes

### Step 1.3: Create Test Environment

```bash
# Option A: Use Docker with specific kernel
docker run -it --privileged \
  -v /home/john/LAT5150DRVMIL:/workspace \
  ubuntu:latest /bin/bash

# Inside container:
apt-get update
apt-get install -y linux-headers-${NEW_KERNEL}-generic build-essential dkms

# Option B: Use VM with new kernel
# Create QEMU/KVM VM with new kernel

# Option C: Test system with new kernel installed
sudo apt-get update
sudo apt-get install -y linux-image-${NEW_KERNEL}-generic linux-headers-${NEW_KERNEL}-generic
# (Reboot to new kernel after testing)
```

- [ ] Test environment prepared
- [ ] Kernel headers installed: `________________`
- [ ] Environment type: Docker / VM / Physical

---

## PHASE 2: AUTOMATED COMPATIBILITY TEST (30 minutes)

### Step 2.1: DKMS Build Test

```bash
# Test DKMS build for dell-milspec-dsmil-dkms
cd /tmp/kernel-compat-${NEW_KERNEL}

# Install package in test environment
sudo dpkg -i /workspace/deployment/debian-packages/dell-milspec-dsmil-dkms_2.1.1-1_all.deb

# Check DKMS build status
dkms status

# Expected: dell-milspec-dsmil/2.1.1, 6.17.0-xxx, x86_64: installed
```

**Expected Output (SUCCESS)**:
```
Loading new dell-milspec-dsmil-2.1.1 DKMS files...
Building for 6.17.0-generic
Building initial module for 6.17.0-generic
Done.

dell-milspec-dsmil/2.1.1, 6.17.0-generic, x86_64: installed
```

**Expected Output (FAILURE)**:
```
Building for 6.17.0-generic
Error! Bad return status for module build on kernel: 6.17.0-generic (x86_64)
Consult /var/lib/dkms/dell-milspec-dsmil/2.1.1/build/make.log for more information.
```

- [ ] DSMIL DKMS build status: SUCCESS / FAILURE
- [ ] TPM2 DKMS build status: SUCCESS / FAILURE

### Step 2.2: Analyze Build Failures

If DKMS build fails:

```bash
# Check build log
sudo cat /var/lib/dkms/dell-milspec-dsmil/2.1.1/build/make.log

# Common error patterns:

# 1. Missing function/symbol
grep "undefined reference\|implicit declaration" /var/lib/dkms/dell-milspec-dsmil/2.1.1/build/make.log

# 2. API changes
grep "incompatible\|has no member named" /var/lib/dkms/dell-milspec-dsmil/2.1.1/build/make.log

# 3. Header changes
grep "No such file or directory" /var/lib/dkms/dell-milspec-dsmil/2.1.1/build/make.log
```

**Common Issues**:

| Error Pattern | Likely Cause | Solution |
|--------------|--------------|----------|
| `implicit declaration of function` | API removed/renamed | Update function calls |
| `incompatible pointer type` | Structure changed | Update structure usage |
| `has no member named` | Structure member removed | Use new member name |
| `undefined reference to` | Symbol moved/removed | Update Makefile or code |

- [ ] Build errors analyzed
- [ ] Root cause identified: `________________`

### Step 2.3: Compatibility Test Suite

If DKMS build succeeds, run automated tests:

```bash
# Create compatibility test script
cat > /tmp/kernel-compat-test.sh << 'EOF'
#!/bin/bash
set -euo pipefail

KERNEL_VERSION=$(uname -r)
echo "=== Dell MIL-SPEC Kernel Compatibility Test ==="
echo "Kernel: ${KERNEL_VERSION}"
echo "Date: $(date)"
echo ""

PASS=0
FAIL=0

# Test 1: Module loads
echo "Test 1: Module Loading"
if sudo modprobe dsmil-72dev; then
    echo "  PASS: dsmil-72dev loaded"
    ((PASS++))
else
    echo "  FAIL: dsmil-72dev failed to load"
    ((FAIL++))
fi

if sudo modprobe tpm2_accel_early; then
    echo "  PASS: tpm2_accel_early loaded"
    ((PASS++))
else
    echo "  FAIL: tpm2_accel_early failed to load"
    ((FAIL++))
fi

# Test 2: Device nodes created
echo "Test 2: Device Nodes"
if [ -c /dev/dsmil0 ]; then
    echo "  PASS: /dev/dsmil0 exists"
    ((PASS++))
else
    echo "  FAIL: /dev/dsmil0 missing"
    ((FAIL++))
fi

if [ -c /dev/tpm2_accel_early ]; then
    echo "  PASS: /dev/tpm2_accel_early exists"
    ((PASS++))
else
    echo "  FAIL: /dev/tpm2_accel_early missing"
    ((FAIL++))
fi

# Test 3: Basic IOCTL operations
echo "Test 3: IOCTL Operations"
if dsmil-status > /dev/null 2>&1; then
    echo "  PASS: DSMIL IOCTL working"
    ((PASS++))
else
    echo "  FAIL: DSMIL IOCTL failed"
    ((FAIL++))
fi

# Test 4: sysfs attributes
echo "Test 4: Sysfs Attributes"
if [ -d /sys/module/dsmil_72dev ]; then
    echo "  PASS: DSMIL sysfs present"
    ((PASS++))
else
    echo "  FAIL: DSMIL sysfs missing"
    ((FAIL++))
fi

# Test 5: dmesg for errors
echo "Test 5: Kernel Messages"
if dmesg | tail -100 | grep -i "dsmil\|tpm2_accel" | grep -iq "error\|fail\|warn"; then
    echo "  FAIL: Errors/warnings in dmesg"
    dmesg | tail -20 | grep -i "dsmil\|tpm2_accel"
    ((FAIL++))
else
    echo "  PASS: No errors in dmesg"
    ((PASS++))
fi

# Test 6: Performance sanity
echo "Test 6: Performance Sanity"
START=$(date +%s%N)
for i in {1..100}; do
    dsmil-status > /dev/null 2>&1 || break
done
END=$(date +%s%N)
ELAPSED=$((($END - $START) / 1000000))  # ms

if [ $ELAPSED -lt 1000 ]; then
    echo "  PASS: 100 operations in ${ELAPSED}ms"
    ((PASS++))
else
    echo "  FAIL: 100 operations took ${ELAPSED}ms (expected <1000ms)"
    ((FAIL++))
fi

echo ""
echo "=== Summary ==="
echo "PASSED: $PASS"
echo "FAILED: $FAIL"

if [ $FAIL -eq 0 ]; then
    echo "STATUS: COMPATIBLE ✅"
    exit 0
else
    echo "STATUS: INCOMPATIBLE ❌"
    exit 1
fi
EOF

chmod +x /tmp/kernel-compat-test.sh
sudo /tmp/kernel-compat-test.sh
```

- [ ] Compatibility tests run
- [ ] Tests passed: `______` / `______`
- [ ] Overall status: COMPATIBLE / INCOMPATIBLE

---

## PHASE 3: FIXING COMPATIBILITY ISSUES (varies)

### Step 3.1: Identify Required Code Changes

**Common Compatibility Patterns**:

#### Pattern 1: IOCTL Structure Changes
```c
// Old kernel (6.14-6.16)
static long dsmil_ioctl(struct file *file, unsigned int cmd, unsigned long arg)
{
    // Implementation
}

static struct file_operations dsmil_fops = {
    .owner = THIS_MODULE,
    .unlocked_ioctl = dsmil_ioctl,
};

// New kernel (6.17+) - hypothetical change
static long dsmil_ioctl(struct file *file, unsigned int cmd, unsigned long arg, void *private_data)
{
    // Implementation with new parameter
}
```

#### Pattern 2: Character Device Registration
```c
// Check kernel version and adapt
#include <linux/version.h>

#if LINUX_VERSION_CODE >= KERNEL_VERSION(6,17,0)
    // Use new API
    result = cdev_device_add(&dsmil_cdev, &dsmil_device);
#else
    // Use old API
    result = cdev_add(&dsmil_cdev, dev_num, 1);
    device_create(...);
#endif
```

#### Pattern 3: Header File Changes
```c
// Old header location
#include <linux/dmi.h>

// New kernel might change to:
#include <linux/firmware/dmi.h>

// Solution: Conditional include
#if LINUX_VERSION_CODE >= KERNEL_VERSION(6,17,0)
    #include <linux/firmware/dmi.h>
#else
    #include <linux/dmi.h>
#endif
```

### Step 3.2: Create Compatibility Patch

```bash
cd /home/john/LAT5150DRVMIL

# Create patch branch
git checkout -b compat/kernel-6.17.0

# Example: Fix IOCTL API change
vim dsmil/kernel_module/dsmil_ioctl.c

# Add kernel version checks
cat >> dsmil/kernel_module/dsmil_ioctl.c << 'EOF'

#include <linux/version.h>

#if LINUX_VERSION_CODE >= KERNEL_VERSION(6,17,0)
/* Kernel 6.17+ compatibility layer */
static long dsmil_ioctl_new(struct file *file, unsigned int cmd,
                             unsigned long arg, void *private_data)
{
    // Adapt to new API
    return dsmil_ioctl_common(file, cmd, arg);
}
#define DSMIL_IOCTL_HANDLER dsmil_ioctl_new
#else
/* Kernel 6.14-6.16 compatibility */
static long dsmil_ioctl(struct file *file, unsigned int cmd, unsigned long arg)
{
    return dsmil_ioctl_common(file, cmd, arg);
}
#define DSMIL_IOCTL_HANDLER dsmil_ioctl
#endif

static struct file_operations dsmil_fops = {
    .owner = THIS_MODULE,
    .unlocked_ioctl = DSMIL_IOCTL_HANDLER,
    .open = dsmil_open,
    .release = dsmil_release,
};
EOF
```

- [ ] Compatibility code added
- [ ] Files modified: `________________`

### Step 3.3: Test Patched Code

```bash
# Rebuild DKMS package with patches
cd /home/john/LAT5150DRVMIL/dsmil/kernel_module
make clean
make

# Test on both old and new kernels
# On kernel 6.17:
sudo insmod dsmil-72dev.ko
dmesg | tail -20
lsmod | grep dsmil

# On kernel 6.16 (if available):
# Boot to 6.16 and repeat test
```

- [ ] Code compiles on kernel 6.17: YES / NO
- [ ] Code compiles on kernel 6.16: YES / NO (backward compat)
- [ ] Module loads successfully: YES / NO

---

## PHASE 4: PACKAGE UPDATE (30 minutes)

### Step 4.1: Update DKMS Package

```bash
cd /home/john/LAT5150DRVMIL

# Update package version (minor bump for compatibility)
vim deployment/debian-packages/dell-milspec-dsmil-dkms/DEBIAN/control
# Version: 2.1.1-1 → 2.1.2-1

# Update changelog
cat >> deployment/debian-packages/dell-milspec-dsmil-dkms/DEBIAN/changelog << EOF

dell-milspec-dsmil-dkms (2.1.2-1) stable; urgency=medium

  * Add compatibility for Linux kernel 6.17.x
  * Update IOCTL handler for new kernel API
  * Maintain backward compatibility with 6.14-6.16

 -- Dell MIL-SPEC Development Team <milspec-dev@dell.com>  $(date -R)
EOF

# Copy patched source files to DKMS package
cp -r dsmil/kernel_module/* \
  deployment/debian-packages/dell-milspec-dsmil-dkms/usr/src/dell-milspec-dsmil-2.1.2/
```

- [ ] Package version updated: `________________`
- [ ] Changelog updated
- [ ] Patched source copied

### Step 4.2: Build and Test Updated Package

```bash
# Build updated package
cd /home/john/LAT5150DRVMIL/deployment/debian-packages/dell-milspec-dsmil-dkms
dpkg-deb --build . ../dell-milspec-dsmil-dkms_2.1.2-1_all.deb

# Test on kernel 6.17 system
sudo dpkg -i ../dell-milspec-dsmil-dkms_2.1.2-1_all.deb

# Verify DKMS build
dkms status
# Should show: dell-milspec-dsmil/2.1.2, 6.17.0-xxx, x86_64: installed

# Run compatibility tests
sudo /tmp/kernel-compat-test.sh
```

- [ ] Package builds successfully
- [ ] DKMS builds on kernel 6.17
- [ ] All compatibility tests pass

---

## PHASE 5: DOCUMENTATION & DEPLOYMENT (30 minutes)

### Step 5.1: Update Compatibility Matrix

```bash
# Update documentation
cat >> /home/john/LAT5150DRVMIL/deployment/docs/kernel-compatibility.txt << EOF

=== Kernel 6.17.x ===
Status: SUPPORTED
DSMIL Package: 2.1.2-1 or later
TPM2 Package: 1.0.0-1 or later
Tested: $(date +%Y-%m-%d)
Notes: Requires updated IOCTL API
EOF

# Update README
vim /home/john/LAT5150DRVMIL/README.md
# Add 6.17.x to supported kernels list
```

- [ ] Documentation updated
- [ ] Compatibility matrix current

### Step 5.2: Deploy to Repository

```bash
# Add updated package to testing repository first
cd /home/john/LAT5150DRVMIL/deployment/apt-repository
./scripts/add-package.sh \
  ../debian-packages/dell-milspec-dsmil-dkms_2.1.2-1_all.deb testing

# Notify testers
cat > KERNEL_COMPAT_UPDATE.txt << EOF
Dell MIL-SPEC Kernel Compatibility Update

A new package version is available for testing with kernel 6.17.x:

Package: dell-milspec-dsmil-dkms
Version: 2.1.2-1
Kernel Support: 6.14.0 - 6.17.x

Install from testing repository:
  sudo apt-get update
  sudo apt-get install -t testing dell-milspec-dsmil-dkms

Please test and report any issues.
EOF

# After successful testing (24-48 hours), promote to stable
./scripts/add-package.sh \
  ../debian-packages/dell-milspec-dsmil-dkms_2.1.2-1_all.deb stable
```

- [ ] Package deployed to testing
- [ ] Testing period: `________________`
- [ ] Package promoted to stable

### Step 5.3: Update CI/CD

```bash
# Update kernel version matrix in CI
vim /home/john/LAT5150DRVMIL/.github/workflows/build-packages.yml

# Add kernel 6.17 to test matrix
cat >> .github/workflows/build-packages.yml << 'EOF'

  test-kernel-compatibility:
    name: Test Kernel Compatibility
    runs-on: ubuntu-latest
    strategy:
      matrix:
        kernel:
          - 6.14.0
          - 6.15.0
          - 6.16.0
          - 6.17.0  # NEW
    steps:
      - name: Test on kernel ${{ matrix.kernel }}
        run: |
          # Install kernel headers
          sudo apt-get install linux-headers-${{ matrix.kernel }}-generic
          # Install DKMS package
          sudo dpkg -i dell-milspec-dsmil-dkms_2.1.2-1_all.deb
          # Verify build
          dkms status | grep "dell-milspec-dsmil"
EOF

# Commit CI updates
git add .github/workflows/build-packages.yml
git commit -m "ci: Add kernel 6.17.x to compatibility test matrix"
```

- [ ] CI/CD updated with new kernel
- [ ] Automated tests configured

---

## AUTOMATED CI/CD WORKFLOW

### GitHub Actions Integration

```yaml
# .github/workflows/kernel-compatibility.yml
name: Kernel Compatibility Monitor

on:
  schedule:
    - cron: '0 6 * * 1'  # Weekly on Monday
  workflow_dispatch:

jobs:
  detect-new-kernels:
    runs-on: ubuntu-latest
    outputs:
      new_kernel: ${{ steps.check.outputs.new_kernel }}
    steps:
      - name: Check for new kernels
        id: check
        run: |
          LATEST=$(curl -s https://www.kernel.org/ | grep -oP 'stable:\s+\K\d+\.\d+\.\d+')
          echo "new_kernel=$LATEST" >> $GITHUB_OUTPUT

  test-compatibility:
    needs: detect-new-kernels
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install kernel ${{ needs.detect-new-kernels.outputs.new_kernel }}
        run: |
          sudo apt-get update
          sudo apt-get install -y linux-headers-${{ needs.detect-new-kernels.outputs.new_kernel }}

      - name: Build DKMS packages
        run: |
          sudo dpkg -i deployment/debian-packages/dell-milspec-dsmil-dkms_*_all.deb
          dkms status

      - name: Run compatibility tests
        run: |
          # Run test suite
          ./deployment/scripts/kernel-compat-test.sh

      - name: Create issue if failed
        if: failure()
        uses: actions/github-script@v6
        with:
          script: |
            github.rest.issues.create({
              owner: context.repo.owner,
              repo: context.repo.repo,
              title: 'Kernel compatibility issue: ${{ needs.detect-new-kernels.outputs.new_kernel }}',
              body: 'DKMS build or tests failed on kernel ${{ needs.detect-new-kernels.outputs.new_kernel }}. See workflow run for details.',
              labels: ['kernel-compatibility', 'bug']
            })
```

- [ ] Automated workflow configured
- [ ] Failure notifications enabled

---

## ROLLBACK PROCEDURE

If new kernel causes issues:

```bash
# Option 1: Pin to older kernel
echo "linux-image-6.16.0-generic hold" | sudo dpkg --set-selections
echo "linux-headers-6.16.0-generic hold" | sudo dpkg --set-selections

# Option 2: Remove problematic package version
sudo apt-get remove dell-milspec-dsmil-dkms
sudo apt-get install dell-milspec-dsmil-dkms=2.1.1-1

# Option 3: Prevent kernel upgrade
cat >> /etc/apt/preferences.d/dell-milspec-kernel << EOF
# Pin kernel version for Dell MIL-SPEC compatibility
Package: linux-image-*
Pin: version 6.16.*
Pin-Priority: 1001
EOF
```

---

## VERIFICATION CHECKLIST

- [ ] New kernel version detected: `________________`
- [ ] DKMS builds successfully on new kernel
- [ ] All compatibility tests pass
- [ ] Backward compatibility maintained (older kernels still work)
- [ ] Package version updated and built
- [ ] Documentation updated
- [ ] Changes deployed to testing repository
- [ ] Testing completed (24-48 hours)
- [ ] Changes deployed to stable repository
- [ ] CI/CD updated with new kernel
- [ ] Compatibility matrix updated

---

## RELATED PROCEDURES

- [HOTFIX_DEPLOYMENT.md](./HOTFIX_DEPLOYMENT.md) - Rapid patch deployment
- [REPOSITORY_MAINTENANCE.md](./REPOSITORY_MAINTENANCE.md) - Repository operations

---

## REVISION HISTORY

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2025-10-11 | Operations Team | Initial release |

---

**Classification**: UNCLASSIFIED // FOR OFFICIAL USE ONLY
**Review Date**: 2025-11-11
**Owner**: Engineering Operations Team
