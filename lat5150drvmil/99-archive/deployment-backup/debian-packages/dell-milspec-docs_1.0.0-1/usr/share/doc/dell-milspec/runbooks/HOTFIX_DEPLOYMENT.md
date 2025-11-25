# HOTFIX DEPLOYMENT RUNBOOK

**Dell MIL-SPEC Platform Operations**
**Document Type**: Rapid Response Procedure
**Version**: 1.0.0
**Last Updated**: 2025-10-11
**Classification**: UNCLASSIFIED // FOR OFFICIAL USE ONLY

---

## PURPOSE

This runbook provides accelerated procedures for rapid deployment of security hotfixes to Dell MIL-SPEC packages when critical vulnerabilities require immediate patching.

**TARGET**: Hotfix deployment from discovery to production in < 4 hours.

---

## SCENARIO

A critical security vulnerability has been discovered that requires immediate patching:
- CVE with CVSS score ≥ 8.0
- Active exploitation detected
- Privilege escalation vulnerability
- Remote code execution vulnerability
- Data leakage vulnerability

**Example**: CVE-2025-XXXX in dell-milspec-dsmil-dkms allows privilege escalation via IOCTL

---

## HOTFIX TIMELINE

**Total Time**: 4 hours maximum

| Phase | Duration | Activity |
|-------|----------|----------|
| **0:00-0:15** | 15 min | Issue assessment and hotfix planning |
| **0:15-0:45** | 30 min | Hotfix code development |
| **0:45-1:30** | 45 min | Critical testing (fast-track) |
| **1:30-2:00** | 30 min | Package building and signing |
| **2:00-2:30** | 30 min | Repository deployment |
| **2:30-3:00** | 30 min | User notification and rollout |
| **3:00-4:00** | 60 min | Monitoring and verification |

---

## PREREQUISITES

### Required Access
- [ ] GitHub repository write access
- [ ] GPG signing key for packages
- [ ] Repository server sudo access
- [ ] CI/CD pipeline admin access
- [ ] Communication channels access

### Required Tools
```bash
# Verify tools
which git dpkg-deb reprepro gpg

# Verify build environment
gcc --version
make --version
dkms --version
```

### Environment Setup
```bash
# Set environment variables
export HOTFIX_WORKSPACE="/tmp/hotfix-$(date +%Y%m%d-%H%M%S)"
export REPO_BASE="/home/john/LAT5150DRVMIL"
export APT_REPO="${REPO_BASE}/deployment/apt-repository"

mkdir -p ${HOTFIX_WORKSPACE}
cd ${HOTFIX_WORKSPACE}

# Clone repository
git clone ${REPO_BASE} dell-milspec-hotfix
cd dell-milspec-hotfix
```

- [ ] Environment configured
- [ ] Repository cloned to: `________________`

---

## PHASE 1: ASSESSMENT & PLANNING (15 minutes)

### Step 1.1: Vulnerability Assessment

```bash
# Document vulnerability details
cat > ${HOTFIX_WORKSPACE}/VULNERABILITY_ASSESSMENT.txt << EOF
====================================================================
VULNERABILITY ASSESSMENT
====================================================================

CVE ID: CVE-2025-XXXX
Discovery Date: $(date +%Y-%m-%d)
Reporter: [NAME/ORGANIZATION]
Severity: CRITICAL (CVSS 9.1)

AFFECTED PACKAGES:
- dell-milspec-dsmil-dkms 2.1.0-1
- [Additional packages]

VULNERABILITY DESCRIPTION:
[Detailed description of the vulnerability]

EXPLOIT DETAILS:
[How the vulnerability can be exploited - LIMITED DISTRIBUTION]

IMPACT:
- Privilege escalation to root
- Unauthorized access to military tokens
- [Other impacts]

PATCH REQUIREMENTS:
[What needs to be fixed]

====================================================================
EOF
```

**Vulnerability Classification**:
- [ ] CVE ID obtained: `________________`
- [ ] CVSS score: `________________`
- [ ] Exploitability: PUBLIC / LIMITED / PRIVATE
- [ ] Patch complexity: SIMPLE / MODERATE / COMPLEX

### Step 1.2: Identify Affected Code

```bash
# Locate vulnerable code
cd ${HOTFIX_WORKSPACE}/dell-milspec-hotfix

# Example: Find vulnerable IOCTL handler
grep -rn "DSMIL_IOCTL_" dsmil/kernel_module/

# Example: Specific function
grep -rn "dsmil_ioctl" dsmil/kernel_module/*.c
```

**Vulnerable Files**:
- [ ] File 1: `________________`
- [ ] File 2: `________________`

### Step 1.3: Plan Minimal Patch

**Hotfix Principles**:
1. **Minimal changes** - Only fix the vulnerability
2. **No features** - No enhancements or refactoring
3. **Backward compatible** - Maintain API/ABI compatibility
4. **Well-tested** - Focus on critical paths only

**Patch Plan**:
```markdown
## Patch Strategy

### Changes Required:
1. [Specific change 1]
2. [Specific change 2]

### Files to Modify:
- dsmil/kernel_module/dsmil_ioctl.c (add bounds checking)
- dsmil/kernel_module/dsmil_security.c (validate token permissions)

### Testing Requirements:
- IOCTL security test
- Privilege escalation attempt (should fail)
- Normal functionality verification

### Estimated Effort: [TIME]
```

- [ ] Patch plan documented and approved
- [ ] Estimated completion time: `________________`

---

## PHASE 2: HOTFIX DEVELOPMENT (30 minutes)

### Step 2.1: Create Hotfix Branch

```bash
cd ${HOTFIX_WORKSPACE}/dell-milspec-hotfix

# Create hotfix branch from production tag
git checkout tags/v2.1.0
git checkout -b hotfix/cve-2025-xxxx

# Verify starting point
git log --oneline -1
```

**Branch Strategy**:
```
main (v2.1.0) ──┐
                ├─→ hotfix/cve-2025-xxxx ─→ v2.1.1 (hotfix)
                │
develop ────────┘
```

- [ ] Hotfix branch created: `________________`
- [ ] Starting commit verified: `________________`

### Step 2.2: Apply Security Patch

**Example: Add Bounds Checking to IOCTL**

```bash
# Edit vulnerable file
cd dsmil/kernel_module

# Apply patch (example)
cat > /tmp/security_patch.diff << 'EOF'
--- a/dsmil/kernel_module/dsmil_ioctl.c
+++ b/dsmil/kernel_module/dsmil_ioctl.c
@@ -156,6 +156,12 @@ static long dsmil_ioctl(struct file *file, unsigned int cmd, unsigned long arg)
     struct dsmil_device *dev = file->private_data;
     int ret = 0;

+    /* CVE-2025-XXXX: Validate device index bounds */
+    if (dev->device_id >= DSMIL_MAX_DEVICES) {
+        pr_err("dsmil: Invalid device ID %d\n", dev->device_id);
+        return -EINVAL;
+    }
+
     switch (cmd) {
         case DSMIL_IOCTL_READ_TOKEN:
             ret = dsmil_read_token(dev, arg);
EOF

# Apply the patch
patch -p1 < /tmp/security_patch.diff

# Or edit manually
vim dsmil_ioctl.c
```

**Patch Verification**:
```bash
# Review changes
git diff

# Compile test (quick check)
cd ${HOTFIX_WORKSPACE}/dell-milspec-hotfix/dsmil/kernel_module
make clean
make

# Check for warnings
echo "Exit code: $?"  # Should be 0
```

- [ ] Security patch applied
- [ ] Code compiles without errors
- [ ] Changes reviewed: `________________`

### Step 2.3: Update Version Numbers

```bash
cd ${HOTFIX_WORKSPACE}/dell-milspec-hotfix

# Update DKMS version
vim packaging/dkms/dell-milspec-dsmil.dkms.conf
# Change: PACKAGE_VERSION="2.1.0" → PACKAGE_VERSION="2.1.1"

# Update Debian package version
vim deployment/debian-packages/dell-milspec-dsmil-dkms/DEBIAN/control
# Change: Version: 2.1.0-1 → Version: 2.1.1-1

# Update changelog
cat >> deployment/debian-packages/dell-milspec-dsmil-dkms/DEBIAN/changelog << EOF

dell-milspec-dsmil-dkms (2.1.1-1) stable; urgency=critical

  * SECURITY: Fix CVE-2025-XXXX privilege escalation vulnerability
  * Add bounds checking to IOCTL handlers
  * Add device ID validation

 -- Dell MIL-SPEC Development Team <milspec-dev@dell.com>  $(date -R)
EOF
```

- [ ] Version numbers updated to 2.1.1-1
- [ ] Changelog updated with CVE reference

### Step 2.4: Commit Changes

```bash
# Commit the hotfix
git add -A
git commit -m "$(cat <<'EOFCOMMIT'
security: Fix CVE-2025-XXXX privilege escalation vulnerability

Add bounds checking to IOCTL handlers to prevent out-of-bounds
access that could lead to privilege escalation.

Security Impact:
- Prevents privilege escalation via malicious IOCTL calls
- Adds device ID validation
- Maintains backward compatibility

CVE: CVE-2025-XXXX
Severity: CRITICAL (CVSS 9.1)
Affected Versions: 2.1.0-1 and earlier

Testing:
- IOCTL security tests pass
- Privilege escalation attempts blocked
- Normal functionality verified
EOFCOMMIT
)"

# Create tag
git tag -a v2.1.1 -m "Hotfix for CVE-2025-XXXX"
```

- [ ] Changes committed
- [ ] Tag created: v2.1.1

---

## PHASE 3: CRITICAL TESTING (45 minutes)

**FAST-TRACK TESTING**: Only critical security and functionality tests

### Step 3.1: Security Test - Exploit Attempt

```bash
# Create exploit test
cat > ${HOTFIX_WORKSPACE}/exploit_test.c << 'EOF'
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <unistd.h>

#define DSMIL_IOCTL_READ_TOKEN _IOWR('D', 1, struct dsmil_token_request)

struct dsmil_token_request {
    unsigned int device_id;
    unsigned int token;
    unsigned int value;
};

int main() {
    int fd;
    struct dsmil_token_request req;

    printf("=== CVE-2025-XXXX Exploit Test ===\n");

    fd = open("/dev/dsmil0", O_RDWR);
    if (fd < 0) {
        perror("open");
        return 1;
    }

    // Attempt exploit with out-of-bounds device ID
    req.device_id = 999;  // Invalid - should be < 84
    req.token = 0x8000;
    req.value = 0;

    printf("Attempting exploit (device_id=999)...\n");
    if (ioctl(fd, DSMIL_IOCTL_READ_TOKEN, &req) < 0) {
        printf("PASS: Exploit blocked (expected behavior)\n");
        close(fd);
        return 0;
    } else {
        printf("FAIL: Exploit succeeded (VULNERABILITY STILL PRESENT!)\n");
        close(fd);
        return 1;
    }
}
EOF

# Compile exploit test
gcc -o ${HOTFIX_WORKSPACE}/exploit_test ${HOTFIX_WORKSPACE}/exploit_test.c

# Load hotfix module
cd ${HOTFIX_WORKSPACE}/dell-milspec-hotfix/dsmil/kernel_module
sudo insmod dsmil-72dev.ko

# Run exploit test
sudo ${HOTFIX_WORKSPACE}/exploit_test
```

**Expected Output**:
```
=== CVE-2025-XXXX Exploit Test ===
Attempting exploit (device_id=999)...
PASS: Exploit blocked (expected behavior)
```

- [ ] Exploit test PASSED (exploit blocked)
- [ ] Test output: `________________`

### Step 3.2: Functionality Test - Normal Operations

```bash
# Test normal IOCTL operations
cd ${HOTFIX_WORKSPACE}/dell-milspec-hotfix

# Use existing test suite (if available)
if [ -f dsmil/kernel_module/test_dsmil.sh ]; then
    sudo ./dsmil/kernel_module/test_dsmil.sh
fi

# Or manual functionality test
cat > ${HOTFIX_WORKSPACE}/functionality_test.sh << 'EOF'
#!/bin/bash
set -euo pipefail

echo "=== DSMIL Functionality Test ==="

# Test 1: Module loaded
if lsmod | grep -q "dsmil"; then
    echo "PASS: Module loaded"
else
    echo "FAIL: Module not loaded"
    exit 1
fi

# Test 2: Device node exists
if [ -c /dev/dsmil0 ]; then
    echo "PASS: Device node exists"
else
    echo "FAIL: Device node missing"
    exit 1
fi

# Test 3: Read valid token (example)
# Use milspec-control if available
if which dsmil-status >/dev/null 2>&1; then
    dsmil-status
    echo "PASS: DSMIL status readable"
fi

echo "=== All Tests Passed ==="
EOF

chmod +x ${HOTFIX_WORKSPACE}/functionality_test.sh
sudo ${HOTFIX_WORKSPACE}/functionality_test.sh
```

**Expected Output**:
```
=== DSMIL Functionality Test ===
PASS: Module loaded
PASS: Device node exists
PASS: DSMIL status readable
=== All Tests Passed ===
```

- [ ] Functionality tests PASSED
- [ ] No regressions detected

### Step 3.3: Performance Sanity Check

```bash
# Quick performance test - ensure no major degradation
cat > ${HOTFIX_WORKSPACE}/perf_test.sh << 'EOF'
#!/bin/bash

echo "=== Performance Sanity Check ==="

# Time 1000 IOCTL operations
time for i in {1..1000}; do
    # Example IOCTL test (use actual tool)
    echo -n > /dev/null
done

echo "If time < 5 seconds: PASS"
EOF

bash ${HOTFIX_WORKSPACE}/perf_test.sh
```

- [ ] Performance acceptable (no major degradation)
- [ ] Average latency: `________________`

---

## PHASE 4: PACKAGE BUILD & SIGN (30 minutes)

### Step 4.1: Build Hotfix Package

```bash
cd ${HOTFIX_WORKSPACE}/dell-milspec-hotfix

# Build DKMS package
cd deployment/debian-packages/dell-milspec-dsmil-dkms

# Clean any previous builds
rm -f ../*.deb

# Build package
dpkg-deb --build . ../dell-milspec-dsmil-dkms_2.1.1-1_all.deb

# Verify package
dpkg-deb --info ../dell-milspec-dsmil-dkms_2.1.1-1_all.deb
dpkg-deb --contents ../dell-milspec-dsmil-dkms_2.1.1-1_all.deb | head -20
```

**Expected Output**:
```
 new Debian package, version 2.0.
 size 12345 bytes: control archive=1234 bytes.
     345 bytes,    12 lines      control
     ...
 Package: dell-milspec-dsmil-dkms
 Version: 2.1.1-1
 ...
```

- [ ] Package built successfully
- [ ] Package size: `________________` bytes
- [ ] Package location: `________________`

### Step 4.2: Sign Package with GPG

```bash
# Sign the package
cd ${HOTFIX_WORKSPACE}/dell-milspec-hotfix/deployment/debian-packages

# List available GPG keys
gpg --list-secret-keys

# Sign package
dpkg-sig --sign builder dell-milspec-dsmil-dkms_2.1.1-1_all.deb

# Verify signature
dpkg-sig --verify dell-milspec-dsmil-dkms_2.1.1-1_all.deb
```

**Expected Output**:
```
Processing dell-milspec-dsmil-dkms_2.1.1-1_all.deb...
GOODSIG _gpgbuilder ...
```

- [ ] Package signed successfully
- [ ] Signature verified
- [ ] GPG key ID: `________________`

### Step 4.3: Run Final Package Validation

```bash
# Lintian checks
lintian dell-milspec-dsmil-dkms_2.1.1-1_all.deb

# Test installation (in clean environment or VM)
sudo dpkg -i dell-milspec-dsmil-dkms_2.1.1-1_all.deb

# Verify installation
dpkg -l | grep dell-milspec-dsmil-dkms

# Verify DKMS build
dkms status | grep dell-milspec-dsmil

# Should show: dell-milspec-dsmil, 2.1.1, 6.x.x-xxx, x86_64: installed
```

- [ ] Lintian checks passed (or warnings acceptable)
- [ ] Package installs cleanly
- [ ] DKMS build successful

---

## PHASE 5: REPOSITORY DEPLOYMENT (30 minutes)

### Step 5.1: Test Repository Addition

```bash
# Copy package to temporary test repository location
cp ${HOTFIX_WORKSPACE}/dell-milspec-hotfix/deployment/debian-packages/dell-milspec-dsmil-dkms_2.1.1-1_all.deb \
   /tmp/hotfix_package.deb

# Test adding to repository (dry run)
cd ${APT_REPO}

# Check current repository state
reprepro -b . list stable | grep dell-milspec-dsmil-dkms
```

**Current State**:
```
stable|main|amd64: dell-milspec-dsmil-dkms 2.1.0-1
```

- [ ] Current repository state documented

### Step 5.2: Deploy to Testing Repository First

```bash
# Add to testing repository first (canary deployment)
cd ${APT_REPO}

reprepro -b . includedeb testing /tmp/hotfix_package.deb

# Verify addition
reprepro -b . list testing | grep dell-milspec-dsmil-dkms

# Update repository metadata
./scripts/update-repository.sh
```

**Expected Output**:
```
testing|main|amd64: dell-milspec-dsmil-dkms 2.1.1-1
```

- [ ] Package added to testing repository
- [ ] Version verified: 2.1.1-1

### Step 5.3: Canary Testing

```bash
# Install on test system from testing repository
# (Assuming test system has testing repo configured)

# On test system:
sudo apt-get update
sudo apt-get install -t testing dell-milspec-dsmil-dkms

# Verify installation
dpkg -l | grep dell-milspec-dsmil-dkms
lsmod | grep dsmil

# Run exploit test again
sudo ${HOTFIX_WORKSPACE}/exploit_test
# Should show: PASS

# Run functionality tests
sudo ${HOTFIX_WORKSPACE}/functionality_test.sh
# Should show: All Tests Passed
```

- [ ] Canary installation successful
- [ ] Security fix verified on canary
- [ ] Functionality verified on canary

### Step 5.4: Deploy to Stable Repository

```bash
# Promote to stable repository
cd ${APT_REPO}

# Remove vulnerable version
reprepro -b . remove stable dell-milspec-dsmil-dkms

# Add hotfix version
reprepro -b . includedeb stable /tmp/hotfix_package.deb

# Verify
reprepro -b . list stable | grep dell-milspec-dsmil-dkms

# Update repository metadata
./scripts/update-repository.sh
```

**Expected Output**:
```
stable|main|amd64: dell-milspec-dsmil-dkms 2.1.1-1
```

- [ ] Hotfix deployed to stable repository
- [ ] Old version removed
- [ ] Repository metadata updated

---

## PHASE 6: USER NOTIFICATION & ROLLOUT (30 minutes)

### Step 6.1: Create Security Advisory

```bash
cat > ${HOTFIX_WORKSPACE}/SECURITY_ADVISORY.txt << EOF
====================================================================
DELL MIL-SPEC PLATFORM - CRITICAL SECURITY UPDATE
====================================================================

Advisory ID: DELLMS-2025-001
CVE ID: CVE-2025-XXXX
Severity: CRITICAL (CVSS 9.1)
Date: $(date +%Y-%m-%d)

AFFECTED PACKAGE:
  Package: dell-milspec-dsmil-dkms
  Affected Versions: 2.1.0-1 and earlier
  Fixed Version: 2.1.1-1

VULNERABILITY DESCRIPTION:
A privilege escalation vulnerability has been discovered in the
dell-milspec-dsmil-dkms kernel module that could allow a local
unprivileged user to gain root access through malicious IOCTL calls.

IMPACT:
- Privilege escalation to root
- Unauthorized access to military security tokens
- Potential compromise of classified data

EXPLOITATION:
Currently no public exploits, but proof-of-concept exists.
Active exploitation has NOT been observed in the wild.

URGENCY: IMMEDIATE - Install within 24 hours

INSTALLATION INSTRUCTIONS:

1. Update package list:
   sudo apt-get update

2. Upgrade to fixed version:
   sudo apt-get install dell-milspec-dsmil-dkms

3. Verify installation:
   dpkg -l | grep dell-milspec-dsmil-dkms
   # Should show: 2.1.1-1

4. Reboot system (recommended):
   sudo reboot

VERIFICATION:

After installation and reboot:
  lsmod | grep dsmil
  dmesg | grep dsmil | tail -5

Should show module version 2.1.1 loaded successfully.

TECHNICAL DETAILS:
[Limited distribution - available to security teams only]

TIMELINE:
- Vulnerability discovered: [DATE]
- Patch developed: [DATE]
- Testing completed: [DATE]
- Released to repository: $(date +%Y-%m-%d)

CONTACT:
  Security Team: security@dell-milspec.local
  Support: support@dell-milspec.local

REFERENCES:
- CVE-2025-XXXX: https://cve.mitre.org/cgi-bin/cvename.cgi?name=CVE-2025-XXXX
- Advisory: https://security.dell-milspec.local/advisories/DELLMS-2025-001

====================================================================
Classification: UNCLASSIFIED // FOR OFFICIAL USE ONLY
====================================================================
EOF
```

- [ ] Security advisory created
- [ ] Reviewed by security team

### Step 6.2: Distribute Advisory

```bash
# Email notification
mail -s "CRITICAL SECURITY UPDATE: dell-milspec-dsmil-dkms CVE-2025-XXXX" \
     -r security@dell-milspec.local \
     admins@example.com < ${HOTFIX_WORKSPACE}/SECURITY_ADVISORY.txt

# Slack notification
curl -X POST -H 'Content-type: application/json' \
  --data "{\"text\":\"CRITICAL SECURITY UPDATE: CVE-2025-XXXX hotfix available. Update dell-milspec-dsmil-dkms immediately. See advisory DELLMS-2025-001.\"}" \
  ${SLACK_WEBHOOK_URL}

# Post to repository MOTD
cp ${HOTFIX_WORKSPACE}/SECURITY_ADVISORY.txt ${APT_REPO}/SECURITY_ADVISORY.txt
```

- [ ] Email sent to administrators
- [ ] Slack notification posted
- [ ] Advisory posted to repository
- [ ] Incident system updated

### Step 6.3: Coordinate Rollout

**Rollout Strategy** (staged deployment):

**Phase 1: Non-Production (0-2 hours)**
- Development systems
- Test systems
- Staging environments

**Phase 2: Production Low-Risk (2-8 hours)**
- Non-critical production systems
- Systems with maintenance windows

**Phase 3: Production Critical (8-24 hours)**
- Mission-critical systems
- Systems requiring change approval

```bash
# Create rollout tracking
cat > ${HOTFIX_WORKSPACE}/rollout_tracking.txt << EOF
System Group | Status | Updated Time | Notes
-------------|--------|--------------|-------
Development  | PENDING |              |
Testing      | PENDING |              |
Staging      | PENDING |              |
Prod-Low     | PENDING |              |
Prod-Critical| PENDING |              |
EOF
```

- [ ] Rollout plan created
- [ ] System groups identified
- [ ] Rollout schedule published

---

## PHASE 7: MONITORING & VERIFICATION (60 minutes)

### Step 7.1: Monitor Installations

```bash
# Monitor APT repository access logs
tail -f /var/log/apache2/access.log | grep "dell-milspec-dsmil-dkms_2.1.1"

# Or for file-based repository
# Monitor package downloads
find ${APT_REPO}/pool/main -name "dell-milspec-dsmil-dkms_2.1.1-1_all.deb" -exec stat {} \;
```

- [ ] Monitoring started
- [ ] Initial installations observed

### Step 7.2: Verify Successful Updates

```bash
# Query systems for successful updates
# (Using configuration management or SSH)

cat > ${HOTFIX_WORKSPACE}/verify_updates.sh << 'EOF'
#!/bin/bash

# List of systems to check
SYSTEMS="prod-server-01 prod-server-02 prod-server-03"

echo "=== Hotfix Verification Report ==="
echo "Generated: $(date)"
echo ""

for host in ${SYSTEMS}; do
    echo "=== $host ==="

    # Check package version
    VERSION=$(ssh $host "dpkg -l | grep dell-milspec-dsmil-dkms | awk '{print \$3}'")

    if [ "$VERSION" == "2.1.1-1" ]; then
        echo "  Status: UPDATED ✓"
    else
        echo "  Status: NOT UPDATED (version: $VERSION) ✗"
    fi

    # Check module loaded
    if ssh $host "lsmod | grep -q dsmil"; then
        echo "  Module: LOADED ✓"
    else
        echo "  Module: NOT LOADED ✗"
    fi

    echo ""
done
EOF

chmod +x ${HOTFIX_WORKSPACE}/verify_updates.sh
${HOTFIX_WORKSPACE}/verify_updates.sh
```

- [ ] Verification script executed
- [ ] Update status: `____%` complete

### Step 7.3: Monitor for Issues

```bash
# Monitor system logs for errors
# (On systems with hotfix installed)

ssh prod-server-01 "dmesg | grep -i 'dsmil\|error' | tail -20"

# Check for DKMS build failures
ssh prod-server-01 "dkms status | grep dell-milspec-dsmil"

# Monitor for security events
ssh prod-server-01 "grep -i 'dsmil' /var/log/audit/audit.log | tail -10"
```

- [ ] No critical errors detected
- [ ] All systems stable
- [ ] Any issues documented: `________________`

---

## SUCCESS CRITERIA

Before closing hotfix deployment:

- [ ] Security vulnerability patched (exploit blocked)
- [ ] Functionality tests passed (no regressions)
- [ ] Package built, signed, and verified
- [ ] Deployed to testing repository and validated
- [ ] Deployed to stable repository
- [ ] Security advisory published
- [ ] Users notified via all channels
- [ ] ≥90% of systems updated within 24 hours
- [ ] No critical issues reported
- [ ] Documentation updated

**Timeline Achieved**:
- [ ] Total time from start to deployment: `______` hours (target: <4 hours)

---

## ROLLBACK PROCEDURE

If hotfix causes critical issues:

```bash
# Emergency rollback to previous version
cd ${APT_REPO}

# Remove hotfix version
reprepro -b . remove stable dell-milspec-dsmil-dkms

# Re-add previous version (from backup)
reprepro -b . includedeb stable pool/main/dell-milspec-dsmil-dkms_2.1.0-1_all.deb

# Update repository
./scripts/update-repository.sh

# Notify users
cat > ROLLBACK_NOTICE.txt << EOF
HOTFIX ROLLBACK NOTICE

The security hotfix 2.1.1-1 has been rolled back to 2.1.0-1
due to critical issues.

Reason: [DESCRIBE ISSUE]

Users should:
1. Do NOT upgrade if on 2.1.0-1
2. If on 2.1.1-1, downgrade:
   sudo apt-get install dell-milspec-dsmil-dkms=2.1.0-1

Original vulnerability remains present. Alternative mitigation:
[DESCRIBE WORKAROUND]

Updated hotfix ETA: [TIME]
EOF

mail -s "URGENT: Hotfix Rollback" admins@example.com < ROLLBACK_NOTICE.txt
```

- [ ] Rollback reason documented
- [ ] Previous version restored
- [ ] Users notified of rollback

---

## POST-DEPLOYMENT REVIEW

### Hotfix Metrics

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| Time to Patch | <4 hours | _____ hours | _____ |
| Testing Duration | <45 min | _____ min | _____ |
| Build Time | <30 min | _____ min | _____ |
| Deployment Time | <30 min | _____ min | _____ |
| Systems Updated (24h) | >90% | _____ % | _____ |
| Issues Reported | 0 | _____ | _____ |

### Lessons Learned

**What Went Well**:
- [Item]
- [Item]

**What Could Be Improved**:
- [Item]
- [Item]

**Process Improvements**:
- [ ] Update runbook based on experience
- [ ] Improve testing automation
- [ ] Enhance monitoring capabilities
- [ ] Review security disclosure process

---

## RELATED PROCEDURES

- [EMERGENCY_PACKAGE_REMOVAL.md](./EMERGENCY_PACKAGE_REMOVAL.md) - Emergency package removal
- [KERNEL_COMPATIBILITY.md](./KERNEL_COMPATIBILITY.md) - Kernel version handling
- [INCIDENT_RESPONSE.md](./INCIDENT_RESPONSE.md) - Security incident handling

---

## REVISION HISTORY

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2025-10-11 | Operations Team | Initial release |

---

**Classification**: UNCLASSIFIED // FOR OFFICIAL USE ONLY
**Review Date**: 2025-11-11
**Owner**: Security Operations Team
