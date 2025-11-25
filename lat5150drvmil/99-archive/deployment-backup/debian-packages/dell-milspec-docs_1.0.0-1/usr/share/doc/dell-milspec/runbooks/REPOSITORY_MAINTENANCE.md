# REPOSITORY MAINTENANCE RUNBOOK

**Dell MIL-SPEC Platform Operations**
**Document Type**: Maintenance Procedure
**Version**: 1.0.0
**Last Updated**: 2025-10-11
**Classification**: UNCLASSIFIED // FOR OFFICIAL USE ONLY

---

## PURPOSE

This runbook provides comprehensive procedures for maintaining the Dell MIL-SPEC APT repository, including package management, cleanup, GPG key operations, and backup procedures.

**Maintenance Schedule**: Weekly cleanup, Monthly audits, Quarterly GPG key review

---

## REPOSITORY OVERVIEW

### Repository Structure

```
/home/john/LAT5150DRVMIL/deployment/apt-repository/
├── pool/main/                    # Package files
│   ├── dell-milspec-dsmil-dkms_*_all.deb
│   ├── tpm2-accel-early-dkms_*_all.deb
│   └── dell-milspec-tools_*_amd64.deb
├── dists/                        # Distribution metadata
│   ├── stable/main/binary-amd64/
│   │   ├── Packages             # Package index
│   │   ├── Packages.gz          # Compressed index
│   │   └── Release              # Release file
│   ├── testing/
│   └── unstable/
├── conf/                         # Repository configuration
│   ├── distributions            # reprepro distributions config
│   └── options                  # reprepro options
├── db/                          # reprepro database
├── gpg/                         # GPG keys
│   ├── public-key.asc          # Public signing key
│   └── private-key.gpg         # Private signing key (SECURED)
└── scripts/                     # Management scripts
    ├── setup-repository.sh
    ├── add-package.sh
    ├── update-repository.sh
    └── list-packages.sh
```

### Repository Distributions

| Distribution | Purpose | Update Frequency | Stability |
|--------------|---------|------------------|-----------|
| **stable** | Production releases | Monthly | High |
| **testing** | Pre-release testing | Weekly | Medium |
| **unstable** | Development builds | Daily | Low |

---

## ROUTINE MAINTENANCE TASKS

### Daily Tasks (Automated)

- [ ] Monitor repository access logs
- [ ] Check disk space usage
- [ ] Verify repository integrity
- [ ] Monitor download metrics

### Weekly Tasks (Manual)

- [ ] Review and clean old package versions
- [ ] Audit package signatures
- [ ] Check for repository corruption
- [ ] Review access logs for anomalies

### Monthly Tasks (Manual)

- [ ] Full repository backup
- [ ] Security audit
- [ ] GPG key status review
- [ ] Capacity planning review

### Quarterly Tasks (Manual)

- [ ] GPG key rotation consideration
- [ ] Repository structure optimization
- [ ] Access policy review
- [ ] Disaster recovery test

---

## SECTION 1: PACKAGE MANAGEMENT

### 1.1 Adding New Packages

```bash
cd /home/john/LAT5150DRVMIL/deployment/apt-repository

# Add package to repository
./scripts/add-package.sh <package.deb> [distribution]

# Example: Add to stable
./scripts/add-package.sh /path/to/dell-milspec-dsmil-dkms_2.1.2-1_all.deb stable

# Verify addition
reprepro -b . list stable | grep dell-milspec-dsmil-dkms

# Update repository metadata
./scripts/update-repository.sh
```

**Verification**:
```bash
# Check package is in pool
ls -lh pool/main/dell-milspec-dsmil-dkms*

# Verify in Packages file
zgrep "Package: dell-milspec-dsmil-dkms" dists/stable/main/binary-amd64/Packages.gz

# Test installation
sudo apt-get update
apt-cache policy dell-milspec-dsmil-dkms
```

- [ ] Package added successfully
- [ ] Repository metadata updated
- [ ] Installation test passed

### 1.2 Removing Packages

**WARNING**: Removing packages from stable should be rare and well-documented.

```bash
cd /home/john/LAT5150DRVMIL/deployment/apt-repository

# List all versions of a package
reprepro -b . list stable | grep dell-milspec-dsmil-dkms

# Remove specific version
reprepro -b . remove stable dell-milspec-dsmil-dkms

# For removing specific version (if multiple exist)
# Edit conf/distributions to allow multiple versions, then:
reprepro -b . removesrc stable dell-milspec-dsmil-dkms 2.1.0-1

# Update repository
./scripts/update-repository.sh
```

**Documentation Required**:
```bash
# Log removal
cat >> /home/john/LAT5150DRVMIL/deployment/apt-repository/PACKAGE_REMOVAL_LOG.txt << EOF
Date: $(date +%Y-%m-%d)
Package: dell-milspec-dsmil-dkms
Version: 2.1.0-1
Distribution: stable
Reason: Security vulnerability CVE-2025-XXXX
Removed by: [YOUR NAME]
Ticket: [TICKET NUMBER]
EOF
```

- [ ] Removal reason documented
- [ ] Approval obtained (if required)
- [ ] Package removed
- [ ] Removal logged

### 1.3 Updating Packages (Version Upgrade)

```bash
cd /home/john/LAT5150DRVMIL/deployment/apt-repository

# Method 1: Replace existing version
# reprepro automatically replaces if same package name
./scripts/add-package.sh dell-milspec-dsmil-dkms_2.1.3-1_all.deb stable

# Method 2: Keep old version (requires config change)
# Edit conf/distributions to allow multiple versions:
# Add line: Allow: multiple-versions yes

# Then add new version
reprepro -b . includedeb stable dell-milspec-dsmil-dkms_2.1.3-1_all.deb

# List all versions
reprepro -b . list stable | grep dell-milspec-dsmil-dkms
```

- [ ] New version added
- [ ] Old version handled (removed/kept)
- [ ] Users notified of update

### 1.4 Copying Packages Between Distributions

```bash
# Promote package from testing to stable
cd /home/john/LAT5150DRVMIL/deployment/apt-repository

# Method 1: Copy package
reprepro -b . copy stable testing dell-milspec-dsmil-dkms

# Method 2: Copy and remove from source
reprepro -b . copysrc stable testing dell-milspec-dsmil-dkms
reprepro -b . removesrc testing dell-milspec-dsmil-dkms

# Verify
reprepro -b . list stable | grep dell-milspec-dsmil-dkms
reprepro -b . list testing | grep dell-milspec-dsmil-dkms
```

- [ ] Package copied successfully
- [ ] Source distribution updated (if removed)

---

## SECTION 2: REPOSITORY CLEANUP

### 2.1 Disk Space Monitoring

```bash
# Check repository size
du -sh /home/john/LAT5150DRVMIL/deployment/apt-repository

# Break down by directory
du -h --max-depth=1 /home/john/LAT5150DRVMIL/deployment/apt-repository | sort -h

# Check for large packages
find /home/john/LAT5150DRVMIL/deployment/apt-repository/pool -name "*.deb" -exec ls -lh {} \; | sort -k5 -h

# Monitor pool directory size
du -sh /home/john/LAT5150DRVMIL/deployment/apt-repository/pool/main
```

**Thresholds**:
- Warning: Repository > 5 GB
- Critical: Repository > 10 GB

- [ ] Current size: `________________` GB
- [ ] Status: OK / WARNING / CRITICAL

### 2.2 Removing Old Package Versions

**Policy**: Keep last 3 versions in stable, last 5 in testing, last 2 in unstable

```bash
cd /home/john/LAT5150DRVMIL/deployment/apt-repository

# List all packages with versions
reprepro -b . list stable

# Create cleanup plan
cat > /tmp/cleanup_plan.txt << EOF
# Dell MIL-SPEC Repository Cleanup Plan
# Date: $(date +%Y-%m-%d)

Packages to Remove:
EOF

# Identify old versions (example for dell-milspec-dsmil-dkms)
reprepro -b . list stable | grep dell-milspec-dsmil-dkms | sort -V

# Remove old versions (manual confirmation required)
read -p "Remove dell-milspec-dsmil-dkms 2.1.0-1? (y/N) " -n 1 -r
if [[ $REPLY =~ ^[Yy]$ ]]; then
    reprepro -b . removesrc stable dell-milspec-dsmil-dkms 2.1.0-1
    echo "Removed: dell-milspec-dsmil-dkms 2.1.0-1" >> /tmp/cleanup_plan.txt
fi
```

**Automated Cleanup Script**:
```bash
cat > /home/john/LAT5150DRVMIL/deployment/apt-repository/scripts/cleanup-old-versions.sh << 'EOF'
#!/bin/bash
# Automated repository cleanup
# Keeps last 3 versions in stable

set -euo pipefail

REPO_BASE="/home/john/LAT5150DRVMIL/deployment/apt-repository"
KEEP_VERSIONS=3

cd ${REPO_BASE}

# Get all packages
PACKAGES=$(reprepro -b . list stable | awk '{print $2}' | sort -u)

for pkg in ${PACKAGES}; do
    echo "Checking $pkg..."

    # Get all versions, sorted newest first
    VERSIONS=$(reprepro -b . list stable | grep "^stable.*${pkg}" | awk '{print $3}' | sort -Vr)

    # Count versions
    VERSION_COUNT=$(echo "$VERSIONS" | wc -l)

    if [ $VERSION_COUNT -gt $KEEP_VERSIONS ]; then
        echo "  Found $VERSION_COUNT versions, keeping $KEEP_VERSIONS"

        # Remove old versions (keep first KEEP_VERSIONS)
        echo "$VERSIONS" | tail -n +$((KEEP_VERSIONS + 1)) | while read ver; do
            echo "  Removing $pkg $ver"
            reprepro -b . removesrc stable $pkg $ver
        done
    else
        echo "  OK ($VERSION_COUNT versions)"
    fi
done

echo "Cleanup complete"
EOF

chmod +x /home/john/LAT5150DRVMIL/deployment/apt-repository/scripts/cleanup-old-versions.sh
```

- [ ] Old versions identified
- [ ] Cleanup script executed
- [ ] Disk space reclaimed: `________________` GB

### 2.3 Database Cleanup

```bash
cd /home/john/LAT5150DRVMIL/deployment/apt-repository

# reprepro database can accumulate cruft over time
# Clean unreferenced packages
reprepro -b . deleteunreferenced

# Check database integrity
reprepro -b . check

# Rebuild database if corrupted
# BACKUP FIRST!
mv db db.backup
reprepro -b . export

# Verify rebuild
reprepro -b . list stable
```

- [ ] Database checked
- [ ] Unreferenced packages cleaned
- [ ] Status: OK / REBUILT

---

## SECTION 3: GPG KEY MANAGEMENT

### 3.1 GPG Key Status Check

```bash
# List GPG keys
gpg --list-keys

# Check key expiration
gpg --list-keys --with-colons | grep -A1 "pub:" | grep -A1 "dell-milspec"

# Detailed key info
gpg --list-secret-keys --keyid-format LONG

# Check key signature status
gpg --check-sigs
```

**Key Information**:
- [ ] Key ID: `________________`
- [ ] Creation Date: `________________`
- [ ] Expiration Date: `________________`
- [ ] Days until expiration: `________________`

### 3.2 Extending GPG Key Expiration

```bash
# Edit key
gpg --edit-key <KEY-ID>

# At gpg> prompt:
# gpg> expire
# Enter new expiration (e.g., 2y for 2 years)
# gpg> save

# Export updated public key
gpg --armor --export <KEY-ID> > /home/john/LAT5150DRVMIL/deployment/apt-repository/gpg/public-key.asc

# Distribute updated key
# Users need to import updated key:
# sudo apt-key add /path/to/public-key.asc
```

- [ ] Key expiration extended
- [ ] New expiration date: `________________`
- [ ] Updated key exported
- [ ] Users notified

### 3.3 GPG Key Rotation (Advanced)

**WARNING**: Key rotation requires careful coordination with all users.

**Timeline**: 4-6 weeks

**Phase 1: Generate New Key**
```bash
# Generate new GPG key
gpg --full-generate-key

# Select:
# (1) RSA and RSA
# Key size: 4096
# Expiration: 2 years
# Real name: Dell MIL-SPEC Repository 2026
# Email: milspec-dev@dell.com

# Export new public key
NEW_KEY_ID="<new-key-id>"
gpg --armor --export ${NEW_KEY_ID} > /home/john/LAT5150DRVMIL/deployment/apt-repository/gpg/public-key-2026.asc
```

- [ ] New key generated
- [ ] New key ID: `________________`
- [ ] Public key exported

**Phase 2: Dual Signing (Weeks 1-4)**
```bash
# Update conf/distributions to sign with both keys
vim /home/john/LAT5150DRVMIL/deployment/apt-repository/conf/distributions

# Change:
# SignWith: yes
# To:
# SignWith: <OLD-KEY-ID> <NEW-KEY-ID>

# Re-export all distributions
reprepro -b . export
```

- [ ] Dual signing configured
- [ ] Users notified of key transition
- [ ] New key published

**Phase 3: Transition to New Key (Week 4-6)**
```bash
# Update to sign with new key only
vim /home/john/LAT5150DRVMIL/deployment/apt-repository/conf/distributions

# Change:
# SignWith: <OLD-KEY-ID> <NEW-KEY-ID>
# To:
# SignWith: <NEW-KEY-ID>

# Re-export
reprepro -b . export

# Notify users to update
cat > KEY_ROTATION_NOTICE.txt << EOF
Dell MIL-SPEC Repository Key Rotation

The repository signing key has been rotated.

OLD KEY: <OLD-KEY-ID>
NEW KEY: <NEW-KEY-ID>

ACTION REQUIRED:
Import the new key:
  wget -qO - https://your-repo/public-key-2026.asc | sudo apt-key add -

Or:
  curl -fsSL https://your-repo/public-key-2026.asc | sudo gpg --dearmor -o /usr/share/keyrings/dell-milspec.gpg

Update sources.list:
  deb [signed-by=/usr/share/keyrings/dell-milspec.gpg] https://your-repo stable main

Timeline:
- Old key deprecated: [DATE]
- Old key will be revoked: [DATE + 6 months]
EOF
```

- [ ] New key activated
- [ ] Users notified with timeline
- [ ] Old key deprecation scheduled

### 3.4 Emergency Key Revocation

**CRITICAL**: Only in case of key compromise

```bash
# Revoke compromised key
gpg --output revoke.asc --gen-revoke <COMPROMISED-KEY-ID>

# Import revocation certificate
gpg --import revoke.asc

# Publish revocation
gpg --keyserver keyserver.ubuntu.com --send-keys <COMPROMISED-KEY-ID>

# IMMEDIATELY generate and activate new key (see 3.3)

# Notify all users URGENTLY
cat > SECURITY_INCIDENT_KEY_COMPROMISE.txt << EOF
URGENT SECURITY NOTICE

The Dell MIL-SPEC repository signing key has been compromised
and has been REVOKED immediately.

COMPROMISED KEY: <KEY-ID>
STATUS: REVOKED

IMMEDIATE ACTION REQUIRED:
1. Stop using packages signed with old key
2. Import new emergency key:
   wget -qO - https://your-repo/emergency-key.asc | sudo apt-key add -
3. Verify all installed packages
4. Report any suspicious activity

New packages will be signed with emergency key: <NEW-KEY-ID>

Incident: [TICKET]
Date: $(date)
EOF
```

- [ ] Compromised key revoked
- [ ] Emergency key activated
- [ ] Security incident opened: `________________`
- [ ] All users notified URGENTLY

---

## SECTION 4: REPOSITORY BACKUP & RECOVERY

### 4.1 Full Repository Backup

```bash
# Create backup directory
BACKUP_DIR="/backup/dell-milspec-repo"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_PATH="${BACKUP_DIR}/repo_backup_${TIMESTAMP}"

mkdir -p ${BACKUP_PATH}

# Backup repository
cd /home/john/LAT5150DRVMIL/deployment/apt-repository

# Backup pool (package files)
tar czf ${BACKUP_PATH}/pool.tar.gz pool/

# Backup database
tar czf ${BACKUP_PATH}/db.tar.gz db/

# Backup configuration
tar czf ${BACKUP_PATH}/conf.tar.gz conf/

# Backup GPG keys (CRITICAL - secure this!)
tar czf ${BACKUP_PATH}/gpg.tar.gz gpg/

# Backup dists (can be regenerated, but good to have)
tar czf ${BACKUP_PATH}/dists.tar.gz dists/

# Create backup manifest
cat > ${BACKUP_PATH}/MANIFEST.txt << EOF
Dell MIL-SPEC Repository Backup
Date: $(date)
Backup Path: ${BACKUP_PATH}

Contents:
- pool.tar.gz: $(du -h ${BACKUP_PATH}/pool.tar.gz | cut -f1)
- db.tar.gz: $(du -h ${BACKUP_PATH}/db.tar.gz | cut -f1)
- conf.tar.gz: $(du -h ${BACKUP_PATH}/conf.tar.gz | cut -f1)
- gpg.tar.gz: $(du -h ${BACKUP_PATH}/gpg.tar.gz | cut -f1)
- dists.tar.gz: $(du -h ${BACKUP_PATH}/dists.tar.gz | cut -f1)

Total Size: $(du -sh ${BACKUP_PATH} | cut -f1)

Package Count:
$(reprepro -b . list stable | wc -l) packages in stable
$(reprepro -b . list testing | wc -l) packages in testing
EOF

# Verify backup integrity
cd ${BACKUP_PATH}
sha256sum *.tar.gz > checksums.sha256

# Secure GPG backup (encrypt it!)
gpg --symmetric --cipher-algo AES256 gpg.tar.gz
rm gpg.tar.gz  # Remove unencrypted version

echo "Backup complete: ${BACKUP_PATH}"
```

- [ ] Backup created: `________________`
- [ ] Backup size: `________________` GB
- [ ] Checksums verified
- [ ] GPG keys encrypted

### 4.2 Restore from Backup

```bash
BACKUP_PATH="/backup/dell-milspec-repo/repo_backup_20251011_120000"
RESTORE_DIR="/home/john/LAT5150DRVMIL/deployment/apt-repository"

# CAUTION: This will overwrite existing repository!

# Verify backup integrity
cd ${BACKUP_PATH}
sha256sum -c checksums.sha256

# Decrypt GPG backup
gpg -d gpg.tar.gz.gpg > gpg.tar.gz

# Stop any services using repository
# sudo systemctl stop apache2

# Backup current state (just in case)
mv ${RESTORE_DIR} ${RESTORE_DIR}.backup.$(date +%Y%m%d_%H%M%S)

# Create new directory
mkdir -p ${RESTORE_DIR}
cd ${RESTORE_DIR}

# Restore files
tar xzf ${BACKUP_PATH}/pool.tar.gz
tar xzf ${BACKUP_PATH}/db.tar.gz
tar xzf ${BACKUP_PATH}/conf.tar.gz
tar xzf ${BACKUP_PATH}/gpg.tar.gz
tar xzf ${BACKUP_PATH}/dists.tar.gz

# Verify restoration
reprepro -b . check
reprepro -b . list stable

# Restart services
# sudo systemctl start apache2

echo "Restore complete"
```

- [ ] Backup integrity verified
- [ ] Repository restored
- [ ] Verification checks passed

### 4.3 Disaster Recovery Test

**Schedule**: Quarterly

```bash
# Test restoration in isolated environment
TEST_DIR="/tmp/repo_dr_test_$(date +%Y%m%d)"
mkdir -p ${TEST_DIR}

# Use most recent backup
LATEST_BACKUP=$(ls -td /backup/dell-milspec-repo/repo_backup_* | head -1)

# Perform test restore
cd ${TEST_DIR}
tar xzf ${LATEST_BACKUP}/pool.tar.gz
tar xzf ${LATEST_BACKUP}/db.tar.gz
tar xzf ${LATEST_BACKUP}/conf.tar.gz
tar xzf ${LATEST_BACKUP}/dists.tar.gz

# Verify
reprepro -b ${TEST_DIR} check
reprepro -b ${TEST_DIR} list stable

# Test package installation
mkdir -p ${TEST_DIR}/test-install
cd ${TEST_DIR}/test-install
apt-get download -o Dir::Etc::SourceList=${TEST_DIR}/sources.list dell-milspec-dsmil-dkms

# Cleanup
rm -rf ${TEST_DIR}

echo "DR test complete: $(date)" >> /home/john/LAT5150DRVMIL/deployment/apt-repository/DR_TEST_LOG.txt
```

- [ ] DR test completed
- [ ] Backup integrity confirmed
- [ ] Restore process validated
- [ ] Test results documented

---

## SECTION 5: MONITORING & LOGGING

### 5.1 Access Log Analysis

```bash
# For file-based repository (if using web server)
REPO_BASE="/home/john/LAT5150DRVMIL/deployment/apt-repository"

# Parse access logs (assuming Apache/Nginx)
# Count downloads by package
grep "\.deb" /var/log/apache2/access.log | awk '{print $7}' | sort | uniq -c | sort -rn | head -10

# Downloads by IP
grep "\.deb" /var/log/apache2/access.log | awk '{print $1}' | sort | uniq -c | sort -rn | head -10

# Downloads over time
grep "\.deb" /var/log/apache2/access.log | awk '{print $4}' | cut -d: -f1 | sort | uniq -c

# Create access report
cat > ${REPO_BASE}/ACCESS_REPORT_$(date +%Y%m%d).txt << EOF
Dell MIL-SPEC Repository Access Report
Date: $(date)

Top 10 Downloaded Packages:
$(grep "\.deb" /var/log/apache2/access.log | awk '{print $7}' | sort | uniq -c | sort -rn | head -10)

Top 10 Accessing IPs:
$(grep "\.deb" /var/log/apache2/access.log | awk '{print $1}' | sort | uniq -c | sort -rn | head -10)

Total Downloads Today:
$(grep "$(date +%d/%b/%Y)" /var/log/apache2/access.log | grep "\.deb" | wc -l)
EOF
```

- [ ] Access logs analyzed
- [ ] Report generated
- [ ] Anomalies identified: `________________`

### 5.2 Integrity Monitoring

```bash
# Monitor repository files for unauthorized changes
REPO_BASE="/home/john/LAT5150DRVMIL/deployment/apt-repository"

# Create baseline (first time only)
if [ ! -f ${REPO_BASE}/integrity_baseline.sha256 ]; then
    find ${REPO_BASE}/pool -name "*.deb" -exec sha256sum {} \; > ${REPO_BASE}/integrity_baseline.sha256
fi

# Check integrity
find ${REPO_BASE}/pool -name "*.deb" -exec sha256sum {} \; > /tmp/current_checksums.txt

# Compare
if diff ${REPO_BASE}/integrity_baseline.sha256 /tmp/current_checksums.txt > /dev/null; then
    echo "Repository integrity: OK"
else
    echo "WARNING: Repository integrity check FAILED!"
    echo "Changes detected:"
    diff ${REPO_BASE}/integrity_baseline.sha256 /tmp/current_checksums.txt
fi
```

- [ ] Integrity check completed
- [ ] Status: OK / FAILED
- [ ] Action taken (if failed): `________________`

---

## SECTION 6: TROUBLESHOOTING

### 6.1 Common Issues

#### Issue: Repository metadata corrupted

**Symptoms**: `apt-get update` fails, reprepro errors

**Solution**:
```bash
cd /home/john/LAT5150DRVMIL/deployment/apt-repository

# Regenerate metadata
reprepro -b . export

# If that fails, rebuild from scratch
rm -rf dists/
reprepro -b . export stable
reprepro -b . export testing
reprepro -b . export unstable
```

#### Issue: GPG signature verification fails

**Symptoms**: `GPG error: ... NO_PUBKEY`

**Solution**:
```bash
# Re-sign repository
cd /home/john/LAT5150DRVMIL/deployment/apt-repository
reprepro -b . export

# Ensure users have correct public key
gpg --armor --export <KEY-ID> > gpg/public-key.asc
# Distribute to users
```

#### Issue: Disk space exhausted

**Symptoms**: Repository operations fail

**Solution**:
```bash
# Run cleanup
./scripts/cleanup-old-versions.sh

# Remove unnecessary files
reprepro -b . deleteunreferenced

# If still full, move old packages to archive
mkdir -p /archive/dell-milspec-packages
mv pool/main/*-old.deb /archive/dell-milspec-packages/
```

### 6.2 Emergency Contacts

- Repository Admin: [NAME] - [CONTACT]
- Security Team: security@dell-milspec.local
- Infrastructure: [NAME] - [CONTACT]

---

## VERIFICATION CHECKLIST

### Weekly Maintenance
- [ ] Disk space checked (< 80% used)
- [ ] Old package versions reviewed
- [ ] Access logs reviewed for anomalies
- [ ] Repository integrity verified

### Monthly Maintenance
- [ ] Full backup completed
- [ ] Backup integrity tested
- [ ] Security audit performed
- [ ] GPG key status checked
- [ ] Documentation updated

### Quarterly Maintenance
- [ ] Disaster recovery test performed
- [ ] GPG key rotation reviewed
- [ ] Access policies reviewed
- [ ] Capacity planning updated

---

## RELATED PROCEDURES

- [EMERGENCY_PACKAGE_REMOVAL.md](./EMERGENCY_PACKAGE_REMOVAL.md) - Emergency package removal
- [HOTFIX_DEPLOYMENT.md](./HOTFIX_DEPLOYMENT.md) - Rapid security patching
- [INCIDENT_RESPONSE.md](./INCIDENT_RESPONSE.md) - Security incident handling

---

## REVISION HISTORY

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2025-10-11 | Operations Team | Initial release |

---

**Classification**: UNCLASSIFIED // FOR OFFICIAL USE ONLY
**Review Date**: 2025-11-11
**Owner**: Repository Operations Team
