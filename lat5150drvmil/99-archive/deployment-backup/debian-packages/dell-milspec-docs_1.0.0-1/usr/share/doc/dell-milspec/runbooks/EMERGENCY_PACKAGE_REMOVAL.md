# EMERGENCY PACKAGE REMOVAL RUNBOOK

**Dell MIL-SPEC Platform Operations**
**Document Type**: Emergency Procedure
**Version**: 1.0.0
**Last Updated**: 2025-10-11
**Classification**: UNCLASSIFIED // FOR OFFICIAL USE ONLY

---

## PURPOSE

This runbook provides step-by-step procedures for emergency removal of Dell MIL-SPEC packages from the APT repository and user systems when critical bugs or security vulnerabilities are discovered.

**CRITICAL**: This procedure is for emergency use only. Standard package updates should follow normal deprecation procedures.

---

## SCENARIO

A critical bug has been discovered in a deployed package that:
- Causes system instability or crashes
- Contains a security vulnerability
- Results in data corruption
- Violates security compliance requirements

**Example**: `dell-milspec-dsmil-dkms 2.1.0-1` causes kernel panic on kernel 6.17.0

---

## PREREQUISITES

### Required Access
- [ ] Root/sudo access to repository server
- [ ] GPG signing key access
- [ ] GitHub repository write access
- [ ] Communication channels (email, Slack, incident system)

### Required Tools
```bash
# Verify tools are available
which reprepro dpkg-sig apt-ftparchive
```

### Required Information
- [ ] Package name and version (e.g., `dell-milspec-dsmil-dkms 2.1.0-1`)
- [ ] Severity level (CRITICAL, HIGH, MEDIUM)
- [ ] Affected systems list
- [ ] Incident ticket number

---

## EMERGENCY RESPONSE TIMELINE

**Total Time**: 30-60 minutes

| Phase | Duration | Description |
|-------|----------|-------------|
| **Detection** | 0-5 min | Issue identified and confirmed |
| **Assessment** | 5-10 min | Impact analysis and severity determination |
| **Repository Removal** | 10-20 min | Package removed from APT repository |
| **User Notification** | 20-30 min | Affected users notified |
| **Verification** | 30-45 min | Confirm removal and user compliance |
| **Post-Incident** | 45-60 min | Documentation and review |

---

## PHASE 1: IMMEDIATE ASSESSMENT (5 minutes)

### Step 1.1: Confirm Critical Issue

```bash
# Document the issue
PACKAGE_NAME="dell-milspec-dsmil-dkms"
PACKAGE_VERSION="2.1.0-1"
ISSUE_TICKET="INC-2025-1234"
SEVERITY="CRITICAL"  # CRITICAL, HIGH, MEDIUM

# Confirm package exists in repository
cd /home/john/LAT5150DRVMIL/deployment/apt-repository
reprepro -b . list stable | grep "${PACKAGE_NAME}"
```

**Expected Output**:
```
stable|main|amd64: dell-milspec-dsmil-dkms 2.1.0-1
```

- [ ] Issue confirmed and documented
- [ ] Severity level assigned
- [ ] Incident ticket created: `________________`

### Step 1.2: Identify Affected Systems

```bash
# Query APT logs to find systems that downloaded the package
# (This requires centralized logging or repository access logs)

# Check local installations (if available)
grep "${PACKAGE_NAME}" /var/log/apt/history.log 2>/dev/null
```

- [ ] Number of affected systems estimated: `________________`
- [ ] Critical production systems identified: `________________`

### Step 1.3: Escalation Decision

**Severity Criteria**:
- **CRITICAL**: Immediate removal required, no approval needed
- **HIGH**: Security team lead approval required
- **MEDIUM**: Normal change control procedures apply

- [ ] Severity confirmed: `________________`
- [ ] Approver notified (if required): `________________`

---

## PHASE 2: REPOSITORY REMOVAL (10 minutes)

### Step 2.1: Backup Current Repository State

```bash
# Create backup of repository metadata
BACKUP_DIR="/home/john/LAT5150DRVMIL/deployment/apt-repository/backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p ${BACKUP_DIR}
cd /home/john/LAT5150DRVMIL/deployment/apt-repository

# Backup repository database
cp -r db/ ${BACKUP_DIR}/db_${TIMESTAMP}/

# Backup distributions file
cp -r dists/ ${BACKUP_DIR}/dists_${TIMESTAMP}/

# Backup pool
cp -r pool/main/${PACKAGE_NAME}* ${BACKUP_DIR}/pool_${TIMESTAMP}/ 2>/dev/null

echo "Backup created: ${BACKUP_DIR}/*_${TIMESTAMP}"
```

**Expected Output**:
```
Backup created: /home/john/LAT5150DRVMIL/deployment/apt-repository/backups/*_20251011_143022
```

- [ ] Repository backup completed
- [ ] Backup location verified: `________________`

### Step 2.2: Remove Package from Repository

```bash
# Navigate to repository
cd /home/john/LAT5150DRVMIL/deployment/apt-repository

# Remove from stable distribution
reprepro -b . remove stable ${PACKAGE_NAME}

# Remove from testing distribution (if present)
reprepro -b . remove testing ${PACKAGE_NAME} 2>/dev/null || true

# Remove from unstable distribution (if present)
reprepro -b . remove unstable ${PACKAGE_NAME} 2>/dev/null || true
```

**Expected Output**:
```
Exporting indices...
Successfully removed dell-milspec-dsmil-dkms from stable
```

- [ ] Package removed from stable
- [ ] Package removed from testing (if applicable)
- [ ] Package removed from unstable (if applicable)

### Step 2.3: Verify Removal

```bash
# Confirm package is no longer listed
reprepro -b . list stable | grep "${PACKAGE_NAME}"
reprepro -b . list testing | grep "${PACKAGE_NAME}"
reprepro -b . list unstable | grep "${PACKAGE_NAME}"

# Should return no results
echo "Exit code: $?"  # Should be 1 (not found)
```

**Expected Output**: (empty - no results)

- [ ] Package verified removed from all distributions
- [ ] Repository metadata regenerated

### Step 2.4: Update Repository Metadata

```bash
# Regenerate repository metadata
cd /home/john/LAT5150DRVMIL/deployment/apt-repository
./scripts/update-repository.sh

# Verify Packages files no longer contain the package
zgrep -l "${PACKAGE_NAME}" dists/stable/main/binary-amd64/Packages.gz
# Should return no results
```

- [ ] Repository metadata updated
- [ ] Package absence verified in Packages files

---

## PHASE 3: USER NOTIFICATION (10 minutes)

### Step 3.1: Create Emergency Advisory

**Template: SECURITY_ADVISORY.txt**

```text
====================================================================
DELL MIL-SPEC PLATFORM - EMERGENCY SECURITY ADVISORY
====================================================================

Advisory ID: ${ISSUE_TICKET}
Date: $(date +%Y-%m-%d)
Severity: ${SEVERITY}

AFFECTED PACKAGE:
  Package: ${PACKAGE_NAME}
  Version: ${PACKAGE_VERSION}
  Status: REMOVED FROM REPOSITORY

ISSUE DESCRIPTION:
[Describe the critical issue - be specific but avoid exposing exploit details]

IMPACT:
[Describe potential impact on affected systems]

IMMEDIATE ACTIONS REQUIRED:

1. DO NOT INSTALL affected package version
2. If installed, remove immediately:

   sudo apt-get remove ${PACKAGE_NAME}
   sudo apt-get autoremove

3. For DKMS packages, verify module is unloaded:

   lsmod | grep dsmil
   # If present, reboot system

4. Update to patched version when available:

   sudo apt-get update
   sudo apt-get install ${PACKAGE_NAME}

VERIFICATION:

Check installed version:
  dpkg -l | grep ${PACKAGE_NAME}

Ensure version is NOT 2.1.0-1

TIMELINE:
- Issue discovered: [TIME]
- Repository removal: [TIME]
- Hotfix estimated: [TIMEFRAME]

CONTACT:
  Security Team: security@dell-milspec.local
  Incident: ${ISSUE_TICKET}

====================================================================
Classification: UNCLASSIFIED // FOR OFFICIAL USE ONLY
====================================================================
```

- [ ] Advisory created with accurate information
- [ ] Advisory reviewed by security team

### Step 3.2: Distribute Advisory

**Communication Channels**:

1. **Email Distribution**:
```bash
# Send to all administrators
mail -s "URGENT: Dell MIL-SPEC Security Advisory ${ISSUE_TICKET}" \
     admins@example.com < SECURITY_ADVISORY.txt
```

2. **Slack/Teams Notification**:
```bash
# Post to #dell-milspec-alerts channel
curl -X POST -H 'Content-type: application/json' \
  --data '{"text":"URGENT SECURITY ADVISORY: Package ${PACKAGE_NAME} removed. See ${ISSUE_TICKET}"}' \
  ${SLACK_WEBHOOK_URL}
```

3. **Repository Message of the Day**:
```bash
# Add warning to repository access
cat > /home/john/LAT5150DRVMIL/deployment/apt-repository/URGENT.txt << EOF
WARNING: Package ${PACKAGE_NAME} version ${PACKAGE_VERSION} has been
removed due to critical issue. DO NOT INSTALL. See ${ISSUE_TICKET}.
EOF
```

- [ ] Email notification sent
- [ ] Slack/Teams notification posted
- [ ] Repository warning added
- [ ] Incident system updated

---

## PHASE 4: SYSTEM REMEDIATION (15 minutes)

### Step 4.1: Identify Installed Systems

**For systems with centralized management**:
```bash
# Query configuration management system
# (Puppet/Ansible/Chef example)
puppet query "Package['${PACKAGE_NAME}']"

# Or use SSH to query multiple systems
for host in $(cat affected_systems.txt); do
  echo "=== $host ==="
  ssh $host "dpkg -l | grep ${PACKAGE_NAME}"
done
```

- [ ] Affected systems identified
- [ ] List saved to: `________________`

### Step 4.2: Remote Package Removal

**CRITICAL**: Test on non-production system first!

```bash
# Create removal script
cat > /tmp/emergency_removal.sh << 'EOF'
#!/bin/bash
set -euo pipefail

PACKAGE="dell-milspec-dsmil-dkms"
VERSION="2.1.0-1"

echo "=== Emergency Package Removal ==="
echo "Package: ${PACKAGE} ${VERSION}"
echo "Host: $(hostname)"
echo ""

# Check if package is installed
if dpkg -l | grep -q "${PACKAGE}"; then
    echo "Package found. Removing..."

    # For DKMS packages, unload module first
    if lsmod | grep -q "dsmil"; then
        echo "Unloading kernel module..."
        sudo rmmod dsmil-72dev || true
    fi

    # Remove package
    sudo apt-get remove -y ${PACKAGE}
    sudo apt-get autoremove -y

    echo "Package removed successfully."

    # Verify removal
    if ! dpkg -l | grep -q "${PACKAGE}"; then
        echo "VERIFICATION: Package successfully removed"
        exit 0
    else
        echo "ERROR: Package still present!"
        exit 1
    fi
else
    echo "Package not installed. No action required."
    exit 0
fi
EOF

chmod +x /tmp/emergency_removal.sh

# Deploy and execute on affected systems
for host in $(cat affected_systems.txt); do
  echo "=== Removing from $host ==="
  scp /tmp/emergency_removal.sh $host:/tmp/
  ssh $host "sudo /tmp/emergency_removal.sh"

  # Log result
  if [ $? -eq 0 ]; then
    echo "$host: SUCCESS" >> removal_log.txt
  else
    echo "$host: FAILED - MANUAL INTERVENTION REQUIRED" >> removal_log.txt
  fi
done
```

- [ ] Removal script tested on non-production system
- [ ] Script deployed to affected systems
- [ ] Removal results logged

### Step 4.3: Verification

```bash
# Verify removal on all systems
for host in $(cat affected_systems.txt); do
  echo "=== Verifying $host ==="
  ssh $host "dpkg -l | grep ${PACKAGE_NAME} || echo 'VERIFIED: Not installed'"
  ssh $host "lsmod | grep dsmil || echo 'VERIFIED: Module not loaded'"
done
```

- [ ] All systems verified clean
- [ ] Any failures documented: `________________`

---

## PHASE 5: HOTFIX DEPLOYMENT (if available)

### Step 5.1: Add Patched Package

```bash
# If hotfix is ready (see HOTFIX_DEPLOYMENT.md for details)
cd /home/john/LAT5150DRVMIL/deployment/apt-repository

# Add patched version
./scripts/add-package.sh dell-milspec-dsmil-dkms_2.1.1-1_all.deb stable

# Update repository
./scripts/update-repository.sh

# Verify new version is available
reprepro -b . list stable | grep "${PACKAGE_NAME}"
```

**Expected Output**:
```
stable|main|amd64: dell-milspec-dsmil-dkms 2.1.1-1
```

- [ ] Patched package added to repository
- [ ] Version verified: `________________`

### Step 5.2: Notify Users of Fix

```bash
# Send update notification
cat > HOTFIX_AVAILABLE.txt << EOF
====================================================================
DELL MIL-SPEC PLATFORM - HOTFIX AVAILABLE
====================================================================

Advisory ID: ${ISSUE_TICKET}
Status: RESOLVED

PATCHED PACKAGE AVAILABLE:
  Package: ${PACKAGE_NAME}
  Version: 2.1.1-1
  Status: AVAILABLE IN REPOSITORY

INSTALLATION INSTRUCTIONS:

1. Update package list:
   sudo apt-get update

2. Install patched version:
   sudo apt-get install ${PACKAGE_NAME}

3. Verify installation:
   dpkg -l | grep ${PACKAGE_NAME}
   # Should show version 2.1.1-1

4. Verify functionality:
   lsmod | grep dsmil
   ls -l /dev/dsmil0

TESTING:
[Include any specific testing procedures]

====================================================================
EOF

# Distribute notification
mail -s "Dell MIL-SPEC Hotfix Available: ${ISSUE_TICKET}" \
     admins@example.com < HOTFIX_AVAILABLE.txt
```

- [ ] Hotfix notification sent
- [ ] Installation instructions provided
- [ ] Support team briefed

---

## PHASE 6: POST-INCIDENT REVIEW (30 minutes)

### Step 6.1: Incident Timeline

**Document complete timeline**:

```markdown
## Incident Timeline: ${ISSUE_TICKET}

### Detection Phase
- **[TIME]**: Issue first reported by [SOURCE]
- **[TIME]**: Issue confirmed by [PERSON]
- **[TIME]**: Severity assessed as ${SEVERITY}

### Response Phase
- **[TIME]**: Emergency procedure initiated
- **[TIME]**: Package removed from repository
- **[TIME]**: Users notified
- **[TIME]**: [N] systems remediated

### Resolution Phase
- **[TIME]**: Hotfix developed
- **[TIME]**: Hotfix tested and validated
- **[TIME]**: Hotfix deployed to repository
- **[TIME]**: All systems patched

### Metrics
- Time to detect: [DURATION]
- Time to contain: [DURATION]
- Time to resolve: [DURATION]
- Systems affected: [NUMBER]
- Downtime: [DURATION]
```

- [ ] Timeline completed and accurate
- [ ] All events documented with times

### Step 6.2: Root Cause Analysis

**Five Whys Analysis**:

1. **Why did the issue occur?**
   - [Answer]

2. **Why wasn't it caught in testing?**
   - [Answer]

3. **Why did testing procedures miss it?**
   - [Answer]

4. **Why were testing procedures inadequate?**
   - [Answer]

5. **Why was the process not reviewed?**
   - [Answer]

**Root Cause**: [IDENTIFIED ROOT CAUSE]

- [ ] Root cause identified
- [ ] Contributing factors documented

### Step 6.3: Corrective Actions

**Immediate Actions** (within 24 hours):
- [ ] Action 1: `________________`
- [ ] Action 2: `________________`
- [ ] Action 3: `________________`

**Short-term Actions** (within 1 week):
- [ ] Action 1: `________________`
- [ ] Action 2: `________________`
- [ ] Action 3: `________________`

**Long-term Actions** (within 1 month):
- [ ] Action 1: `________________`
- [ ] Action 2: `________________`
- [ ] Action 3: `________________`

### Step 6.4: Lessons Learned

**What Went Well**:
- [Item 1]
- [Item 2]
- [Item 3]

**What Could Be Improved**:
- [Item 1]
- [Item 2]
- [Item 3]

**Action Items for Process Improvement**:
- [ ] Update testing procedures
- [ ] Enhance monitoring/detection
- [ ] Improve communication templates
- [ ] Update this runbook based on experience

---

## VERIFICATION CHECKLIST

Before closing the incident:

- [ ] Package removed from all repository distributions
- [ ] All affected systems identified
- [ ] All affected systems remediated and verified
- [ ] Users notified of issue and resolution
- [ ] Hotfix deployed (if applicable)
- [ ] Documentation updated
- [ ] Post-incident review completed
- [ ] Corrective actions assigned and tracked
- [ ] Incident ticket closed: `________________`

---

## ROLLBACK PROCEDURE

If emergency removal causes worse issues:

```bash
# Restore from backup
cd /home/john/LAT5150DRVMIL/deployment/apt-repository/backups

# Identify backup
ls -ltr db_* | tail -1

# Restore repository database
BACKUP_TIME="20251011_143022"  # From Step 2.1
cd /home/john/LAT5150DRVMIL/deployment/apt-repository
rm -rf db/
cp -r backups/db_${BACKUP_TIME}/ db/

# Regenerate metadata
./scripts/update-repository.sh

# Verify package is back
reprepro -b . list stable | grep "${PACKAGE_NAME}"
```

- [ ] Rollback reason documented
- [ ] Approval obtained for rollback
- [ ] Repository restored from backup
- [ ] Users notified of rollback

---

## ESCALATION CONTACTS

**Security Team**:
- On-call: [PHONE NUMBER]
- Email: security@dell-milspec.local
- Slack: #security-oncall

**Engineering Lead**:
- Primary: [NAME] - [CONTACT]
- Secondary: [NAME] - [CONTACT]

**Management**:
- Director: [NAME] - [CONTACT]
- VP Engineering: [NAME] - [CONTACT]

**External Contacts** (for severe incidents):
- Dell Security: [CONTACT]
- CERT/CC: [CONTACT] (for vulnerabilities)

---

## RELATED PROCEDURES

- [HOTFIX_DEPLOYMENT.md](./HOTFIX_DEPLOYMENT.md) - Rapid security patch deployment
- [INCIDENT_RESPONSE.md](./INCIDENT_RESPONSE.md) - Security incident handling
- [REPOSITORY_MAINTENANCE.md](./REPOSITORY_MAINTENANCE.md) - Standard repository operations

---

## REVISION HISTORY

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2025-10-11 | Operations Team | Initial release |

---

**Classification**: UNCLASSIFIED // FOR OFFICIAL USE ONLY
**Review Date**: 2025-11-11
**Owner**: Operations Team
