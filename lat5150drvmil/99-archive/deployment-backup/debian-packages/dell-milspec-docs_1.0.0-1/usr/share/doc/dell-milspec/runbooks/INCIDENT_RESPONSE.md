# INCIDENT RESPONSE RUNBOOK

**Dell MIL-SPEC Platform Operations**
**Document Type**: Security Procedure
**Version**: 1.0.0
**Last Updated**: 2025-10-11
**Classification**: UNCLASSIFIED // FOR OFFICIAL USE ONLY

---

## PURPOSE

This runbook provides comprehensive incident response procedures for security events involving the Dell MIL-SPEC platform, including compromised packages, unauthorized access, quarantine violations, and emergency stops.

**CRITICAL**: This runbook contains time-sensitive procedures. Familiarize yourself with it BEFORE an incident occurs.

---

## INCIDENT SEVERITY LEVELS

| Level | Description | Response Time | Notification |
|-------|-------------|---------------|--------------|
| **P0 - CRITICAL** | Active exploit, data breach, system compromise | Immediate (< 15 min) | All teams + Management |
| **P1 - HIGH** | Potential compromise, quarantine violation | < 1 hour | Security + Ops teams |
| **P2 - MEDIUM** | Suspicious activity, failed authentication | < 4 hours | Security team |
| **P3 - LOW** | Policy violation, informational | < 24 hours | On-call engineer |

---

## INCIDENT RESPONSE TEAM

### Roles & Responsibilities

**Incident Commander (IC)**:
- Overall incident coordination
- Decision authority
- Communication with management
- Contact: [NAME] - [PHONE]

**Security Lead**:
- Security analysis
- Forensics coordination
- Threat assessment
- Contact: [NAME] - [PHONE]

**Operations Lead**:
- System containment
- Service restoration
- Log collection
- Contact: [NAME] - [PHONE]

**Communications Lead**:
- User notifications
- Stakeholder updates
- External coordination
- Contact: [NAME] - [PHONE]

---

## GENERAL INCIDENT RESPONSE PROCESS

### Phase 1: DETECTION (Minutes 0-5)
1. Incident detected via monitoring/alert/report
2. Initial assessment and severity determination
3. Incident Commander notified
4. War room established (physical or virtual)

### Phase 2: CONTAINMENT (Minutes 5-30)
1. Immediate threat containment
2. Affected systems isolated
3. Evidence preservation
4. Attack vector blocked

### Phase 3: INVESTIGATION (Hours 1-4)
1. Root cause analysis
2. Scope determination
3. Forensic data collection
4. Timeline reconstruction

### Phase 4: ERADICATION (Hours 4-8)
1. Threat removal
2. Vulnerability patching
3. System hardening
4. Verification

### Phase 5: RECOVERY (Hours 8-24)
1. System restoration
2. Service resumption
3. Monitoring enhancement
4. User notification

### Phase 6: POST-INCIDENT (Days 1-7)
1. Incident report
2. Lessons learned
3. Process improvements
4. Policy updates

---

## SCENARIO 1: COMPROMISED PACKAGE DETECTED

### Trigger
A Dell MIL-SPEC package in the repository has been compromised (malware, backdoor, unauthorized modification).

**Severity**: P0 - CRITICAL

### Detection Methods
- Integrity check failure (SHA256 mismatch)
- Antivirus/EDR alert
- User report of suspicious behavior
- Security audit finding

---

### IMMEDIATE ACTIONS (< 15 minutes)

#### Step 1.1: Verify Compromise

```bash
# Get package details
COMPROMISED_PACKAGE="dell-milspec-dsmil-dkms"
COMPROMISED_VERSION="2.1.2-1"

cd /home/john/LAT5150DRVMIL/deployment/apt-repository

# Check package integrity
sha256sum pool/main/${COMPROMISED_PACKAGE}_${COMPROMISED_VERSION}_all.deb

# Compare with known-good hash
# (From build logs, CI/CD artifacts, or backup)
echo "[KNOWN-GOOD-HASH] pool/main/${COMPROMISED_PACKAGE}_${COMPROMISED_VERSION}_all.deb" | sha256sum -c

# Extract and examine package
mkdir -p /tmp/forensic/${COMPROMISED_PACKAGE}
dpkg-deb --raw-extract pool/main/${COMPROMISED_PACKAGE}_${COMPROMISED_VERSION}_all.deb \
  /tmp/forensic/${COMPROMISED_PACKAGE}/

# Check for suspicious files
find /tmp/forensic/${COMPROMISED_PACKAGE}/ -type f -name "*.sh" -exec cat {} \;
find /tmp/forensic/${COMPROMISED_PACKAGE}/ -type f -exec grep -l "bash -i\|nc -e\|eval\|base64" {} \;
```

**Compromise Indicators**:
- [ ] SHA256 mismatch confirmed
- [ ] Suspicious scripts found: `________________`
- [ ] Unauthorized files detected: `________________`
- [ ] Backdoor evidence: `________________`

#### Step 1.2: Immediate Containment (< 5 minutes)

```bash
# EMERGENCY: Remove package from repository IMMEDIATELY
cd /home/john/LAT5150DRVMIL/deployment/apt-repository

# Remove from all distributions
reprepro -b . remove stable ${COMPROMISED_PACKAGE}
reprepro -b . remove testing ${COMPROMISED_PACKAGE}
reprepro -b . remove unstable ${COMPROMISED_PACKAGE}

# Verify removal
reprepro -b . list stable | grep ${COMPROMISED_PACKAGE}
# Should return nothing

# Update repository immediately
./scripts/update-repository.sh

# Take repository offline (if web-served)
# sudo systemctl stop apache2
# Or set maintenance page:
# echo "Repository offline for emergency maintenance" > /var/www/html/index.html
```

**Containment Timestamp**: `________________`

- [ ] Package removed from repository (< 5 min)
- [ ] Repository updated
- [ ] Repository taken offline (if applicable)

#### Step 1.3: Emergency Notification (< 10 minutes)

```bash
# Create emergency alert
cat > /tmp/SECURITY_INCIDENT_ALERT.txt << EOF
====================================================================
CRITICAL SECURITY INCIDENT - IMMEDIATE ACTION REQUIRED
====================================================================

Incident ID: SEC-$(date +%Y%m%d-%H%M%S)
Severity: P0 - CRITICAL
Time: $(date)

COMPROMISED PACKAGE DETECTED:
Package: ${COMPROMISED_PACKAGE}
Version: ${COMPROMISED_VERSION}
Status: REMOVED FROM REPOSITORY

THREAT:
[Describe threat - malware, backdoor, etc.]

IMMEDIATE ACTIONS REQUIRED:

1. DO NOT INSTALL this package version
2. If installed, REMOVE IMMEDIATELY and ISOLATE SYSTEM:

   sudo systemctl stop tpm2-acceleration-early
   sudo rmmod dsmil-72dev tpm2_accel_early
   sudo apt-get remove --purge ${COMPROMISED_PACKAGE}
   sudo reboot

3. After removal, run security scan:

   sudo rkhunter --check
   sudo chkrootkit

4. Report any suspicious activity to security@dell-milspec.local

AFFECTED SYSTEMS:
Any system that installed ${COMPROMISED_PACKAGE} ${COMPROMISED_VERSION}
between [START_DATE] and $(date +%Y-%m-%d)

NEXT STEPS:
- Full investigation in progress
- Clean replacement package will be provided
- Follow-up forensics may be required on affected systems

INCIDENT COMMANDER: [NAME] - [CONTACT]
SECURITY HOTLINE: [PHONE]

====================================================================
Classification: UNCLASSIFIED // FOR OFFICIAL USE ONLY
Distribution: IMMEDIATE - ALL ADMINISTRATORS
====================================================================
EOF

# Send emergency notification
mail -s "CRITICAL SECURITY INCIDENT: Compromised Package ${COMPROMISED_PACKAGE}" \
     -r security@dell-milspec.local \
     -c management@dell-milspec.local \
     all-admins@dell-milspec.local < /tmp/SECURITY_INCIDENT_ALERT.txt

# Slack emergency notification
curl -X POST -H 'Content-type: application/json' \
  --data "{\"text\":\"@channel CRITICAL SECURITY INCIDENT: Package ${COMPROMISED_PACKAGE} compromised. REMOVE IMMEDIATELY. Check email for details.\"}" \
  ${SLACK_WEBHOOK_URL}
```

- [ ] Emergency notification sent (< 10 min)
- [ ] All teams notified
- [ ] Management informed

---

### INVESTIGATION PHASE (Hours 1-4)

#### Step 2.1: Identify Affected Systems

```bash
# Query systems that downloaded the compromised package
# (Requires access logs or centralized management)

# Check repository access logs
grep "${COMPROMISED_PACKAGE}_${COMPROMISED_VERSION}" /var/log/apache2/access.log | \
  awk '{print $1, $4, $7}' | sort -u > /tmp/affected_systems.txt

# Or query configuration management
# puppet query "Package['${COMPROMISED_PACKAGE}']"
# ansible all -m shell -a "dpkg -l | grep ${COMPROMISED_PACKAGE}"

# Create affected systems list
cat > /tmp/AFFECTED_SYSTEMS.txt << EOF
Affected Systems List
Incident: SEC-$(date +%Y%m%d-%H%M%S)
Package: ${COMPROMISED_PACKAGE} ${COMPROMISED_VERSION}

IP Address          Hostname                Download Time           Status
================================================================================
EOF

# For each IP in access logs, resolve hostname and check installation status
cat /tmp/affected_systems.txt | while read ip timestamp file; do
    hostname=$(dig +short -x $ip | head -1)
    echo "$ip $hostname $timestamp UNKNOWN" >> /tmp/AFFECTED_SYSTEMS.txt
done

echo ""
echo "Total affected systems: $(wc -l < /tmp/affected_systems.txt)"
```

- [ ] Access logs analyzed
- [ ] Affected systems identified: `______` systems
- [ ] Systems list saved to: `________________`

#### Step 2.2: Forensic Analysis

```bash
# Establish forensic workspace
FORENSIC_DIR="/tmp/incident_forensics_$(date +%Y%m%d_%H%M%S)"
mkdir -p ${FORENSIC_DIR}/{package,logs,timeline,evidence}

# Collect evidence from compromised package
cd ${FORENSIC_DIR}/package
cp /home/john/LAT5150DRVMIL/deployment/apt-repository/pool/main/${COMPROMISED_PACKAGE}_${COMPROMISED_VERSION}_all.deb ./ || \
  cp /home/john/LAT5150DRVMIL/deployment/apt-repository/backups/pool_*/main/${COMPROMISED_PACKAGE}_${COMPROMISED_VERSION}_all.deb ./

# Extract and analyze
dpkg-deb --raw-extract ${COMPROMISED_PACKAGE}_${COMPROMISED_VERSION}_all.deb extracted/

# Generate file listing with hashes
find extracted/ -type f -exec sha256sum {} \; > file_hashes.txt

# Check for suspicious patterns
echo "=== Suspicious Scripts ===" > ${FORENSIC_DIR}/evidence/suspicious_findings.txt
find extracted/ -type f \( -name "*.sh" -o -name "postinst" -o -name "prerm" \) -exec grep -Hn "eval\|base64\|wget\|curl.*sh\|nc -e\|bash -i" {} \; >> ${FORENSIC_DIR}/evidence/suspicious_findings.txt

# Check for unexpected binaries
echo "=== Unexpected Binaries ===" >> ${FORENSIC_DIR}/evidence/suspicious_findings.txt
find extracted/ -type f -executable >> ${FORENSIC_DIR}/evidence/suspicious_findings.txt

# Network connections (if analyzing on live system)
echo "=== Network Connections ===" >> ${FORENSIC_DIR}/evidence/suspicious_findings.txt
file extracted/DEBIAN/postinst >> ${FORENSIC_DIR}/evidence/suspicious_findings.txt
strings extracted/DEBIAN/postinst | grep -E "http://|https://|ftp://|[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+" >> ${FORENSIC_DIR}/evidence/suspicious_findings.txt

# Timeline reconstruction
cat > ${FORENSIC_DIR}/timeline/TIMELINE.txt << EOF
Incident Timeline Reconstruction
Incident: SEC-$(date +%Y%m%d-%H%M%S)

=== Package Build Timeline ===
$(dpkg-deb --info ${COMPROMISED_PACKAGE}_${COMPROMISED_VERSION}_all.deb | grep -A5 "Package:")

=== Repository Addition ===
$(grep "${COMPROMISED_PACKAGE}_${COMPROMISED_VERSION}" /home/john/LAT5150DRVMIL/deployment/apt-repository/db/*.db 2>/dev/null | head -5)

=== First Download ===
$(grep "${COMPROMISED_PACKAGE}_${COMPROMISED_VERSION}" /var/log/apache2/access.log 2>/dev/null | head -1)

=== Last Known Good Version ===
[Identify from backups or CI/CD]

=== Detection ===
Detection Time: $(date)
Detection Method: [HOW DETECTED]
Reporter: [WHO REPORTED]
EOF
```

- [ ] Forensic data collected
- [ ] Suspicious indicators documented
- [ ] Timeline reconstructed
- [ ] Evidence preserved at: `________________`

#### Step 2.3: Determine Attack Vector

**Common Attack Vectors**:

1. **Compromised Build System**
   - Check CI/CD logs
   - Verify GitHub Actions workflow integrity
   - Review recent commits to build scripts

2. **Compromised Developer Account**
   - Review git commit signatures
   - Check for unauthorized repository access
   - Audit GPG key usage

3. **Repository Server Compromise**
   - Check server access logs
   - Review sudo/authentication logs
   - Verify file modification times

4. **Supply Chain Attack**
   - Review dependencies
   - Check for compromised upstream packages
   - Verify third-party sources

```bash
# Check git history for suspicious commits
cd /home/john/LAT5150DRVMIL
git log --all --pretty=format:"%h %an %ae %ad %s" --since="30 days ago" | \
  grep -v "noreply@anthropic.com\|known-developer@dell.com"

# Check for unsigned commits
git log --show-signature --since="30 days ago" | grep -B5 "No signature"

# Check repository file modifications
find /home/john/LAT5150DRVMIL/deployment/apt-repository -type f -mtime -7 -ls

# Check server authentication logs
sudo grep "Accepted\|Failed" /var/log/auth.log | tail -100

# Check for privilege escalation
sudo grep "sudo:" /var/log/auth.log | tail -50
```

**Attack Vector Identified**: `________________`

- [ ] Attack vector determined
- [ ] Initial access point identified
- [ ] Lateral movement tracked (if applicable)

---

### ERADICATION PHASE (Hours 4-8)

#### Step 3.1: Remove Threat

```bash
# If repository server compromised
# 1. Change all passwords
# 2. Rotate GPG keys (see REPOSITORY_MAINTENANCE.md Section 3.4)
# 3. Rebuild server from clean backup

# If build system compromised
# 1. Rebuild build environment
# 2. Audit all CI/CD workflows
# 3. Re-sign all packages with new key

# If developer account compromised
# 1. Revoke access immediately
# 2. Force password reset for all developers
# 3. Enable 2FA if not already enabled

# Clean and rebuild compromised package
cd /home/john/LAT5150DRVMIL

# Revert to last known good version
git checkout tags/v2.1.1  # Or appropriate clean version

# Rebuild package from clean source
cd deployment/debian-packages/dell-milspec-dsmil-dkms
dpkg-deb --build . ../${COMPROMISED_PACKAGE}_2.1.3-1_all.deb

# Sign with NEW key (if key compromised)
dpkg-sig --sign builder ../${COMPROMISED_PACKAGE}_2.1.3-1_all.deb

# Verify clean
sha256sum ../${COMPROMISED_PACKAGE}_2.1.3-1_all.deb
```

- [ ] Threat removed from source
- [ ] Clean package rebuilt
- [ ] Package verified clean

#### Step 3.2: System Hardening

```bash
# Implement additional security controls

# 1. Enable package integrity monitoring
cat > /home/john/LAT5150DRVMIL/deployment/apt-repository/scripts/integrity-monitor.sh << 'EOF'
#!/bin/bash
# Continuous integrity monitoring
while true; do
    find pool/main -name "*.deb" -exec sha256sum {} \; > /tmp/current.sha256
    if ! diff baseline.sha256 /tmp/current.sha256 >/dev/null; then
        mail -s "ALERT: Repository integrity violation" security@dell-milspec.local < /tmp/diff.txt
    fi
    sleep 300  # Check every 5 minutes
done
EOF

# 2. Enable audit logging
sudo auditctl -w /home/john/LAT5150DRVMIL/deployment/apt-repository -p wa -k repo_changes

# 3. Restrict repository access
chmod 750 /home/john/LAT5150DRVMIL/deployment/apt-repository
chown root:repo-admins /home/john/LAT5150DRVMIL/deployment/apt-repository

# 4. Enable GPG signature verification (mandatory)
echo "APT::Get::AllowUnauthenticated \"false\";" | sudo tee /etc/apt/apt.conf.d/99-require-signatures
```

- [ ] Integrity monitoring enabled
- [ ] Audit logging configured
- [ ] Access controls hardened
- [ ] Signature verification enforced

---

### RECOVERY PHASE (Hours 8-24)

#### Step 4.1: Deploy Clean Package

```bash
# Add clean package to repository
cd /home/john/LAT5150DRVMIL/deployment/apt-repository

./scripts/add-package.sh ../debian-packages/${COMPROMISED_PACKAGE}_2.1.3-1_all.deb stable

# Update repository
./scripts/update-repository.sh

# Verify
reprepro -b . list stable | grep ${COMPROMISED_PACKAGE}
```

- [ ] Clean package deployed
- [ ] Repository online
- [ ] Package availability verified

#### Step 4.2: Coordinate User Remediation

```bash
# Create remediation instructions
cat > /tmp/REMEDIATION_INSTRUCTIONS.txt << EOF
====================================================================
SECURITY INCIDENT REMEDIATION INSTRUCTIONS
====================================================================

Incident: SEC-$(date +%Y%m%d-%H%M%S)
Package: ${COMPROMISED_PACKAGE}
Compromised Version: ${COMPROMISED_VERSION}
Clean Version: 2.1.3-1

REMEDIATION STEPS:

1. Verify if compromised version is installed:
   dpkg -l | grep ${COMPROMISED_PACKAGE}

2. If version ${COMPROMISED_VERSION} is installed:

   a. Remove compromised package:
      sudo apt-get remove --purge ${COMPROMISED_PACKAGE}

   b. Run security scan:
      sudo rkhunter --check
      sudo chkrootkit

   c. Review system logs:
      sudo grep "dsmil\|tpm2" /var/log/syslog
      sudo journalctl -u tpm2-acceleration-early

   d. Check for indicators of compromise:
      - Unusual cron jobs: crontab -l
      - Suspicious network connections: netstat -antp
      - Unknown processes: ps aux | grep -v "grep"
      - Modified system files: sudo rpm -Va  (or debsums)

3. Install clean version:
   sudo apt-get update
   sudo apt-get install ${COMPROMISED_PACKAGE}

4. Verify clean installation:
   dpkg -l | grep ${COMPROMISED_PACKAGE}
   # Should show version 2.1.3-1

   lsmod | grep dsmil
   dsmil-status

5. Report results to security team:
   - Hostname: $(hostname)
   - Compromised version was installed: YES / NO
   - Remediation completed: YES / NO
   - Indicators of compromise found: YES / NO / UNKNOWN
   - Clean version installed: YES / NO

Submit report to: security@dell-milspec.local
Subject: Remediation Report - SEC-$(date +%Y%m%d-%H%M%S)

====================================================================
Support: [PHONE] | [EMAIL]
====================================================================
EOF

# Distribute remediation instructions
mail -s "SECURITY REMEDIATION REQUIRED: ${COMPROMISED_PACKAGE}" \
     all-admins@dell-milspec.local < /tmp/REMEDIATION_INSTRUCTIONS.txt
```

- [ ] Remediation instructions distributed
- [ ] User support established
- [ ] Remediation tracking system setup

#### Step 4.3: Monitor Remediation Progress

```bash
# Create tracking spreadsheet
cat > /tmp/remediation_tracking.csv << EOF
Hostname,IP Address,Compromised Version Installed,Remediation Started,Remediation Completed,IOCs Found,Verification Status
EOF

# Monitor remediation (update as reports come in)
# Set up dashboard or tracking system
```

- [ ] Remediation tracking active
- [ ] Progress: `_____%` complete
- [ ] IOCs found on: `______` systems

---

### POST-INCIDENT PHASE (Days 1-7)

#### Step 5.1: Incident Report

```bash
cat > /tmp/INCIDENT_REPORT_FINAL.md << 'EOF'
# Security Incident Report

**Incident ID**: SEC-[DATE]-[TIME]
**Severity**: P0 - CRITICAL
**Status**: CLOSED
**Date Range**: [START] - [END]
**Incident Commander**: [NAME]

## Executive Summary
[2-3 paragraph summary of incident]

## Timeline
| Time | Event |
|------|-------|
| [TIME] | Incident detected |
| [TIME] | Containment initiated |
| [TIME] | Package removed from repository |
| [TIME] | Users notified |
| [TIME] | Investigation completed |
| [TIME] | Clean package deployed |
| [TIME] | Remediation 100% complete |

## Impact Assessment
- **Systems Affected**: [NUMBER]
- **Data Compromised**: [YES/NO/UNKNOWN]
- **Downtime**: [DURATION]
- **Users Impacted**: [NUMBER]
- **Financial Impact**: $[AMOUNT]

## Root Cause
[Detailed root cause analysis]

## Attack Vector
[How the compromise occurred]

## Indicators of Compromise
1. [IOC 1]
2. [IOC 2]
3. [IOC 3]

## Response Actions
1. [Action 1]
2. [Action 2]
3. [Action 3]

## Lessons Learned

### What Went Well
- [Item 1]
- [Item 2]

### What Could Be Improved
- [Item 1]
- [Item 2]

### Action Items
- [ ] Action 1 - Owner: [NAME] - Due: [DATE]
- [ ] Action 2 - Owner: [NAME] - Due: [DATE]
- [ ] Action 3 - Owner: [NAME] - Due: [DATE]

## Recommendations
1. [Recommendation 1]
2. [Recommendation 2]
3. [Recommendation 3]

---
**Classification**: UNCLASSIFIED // FOR OFFICIAL USE ONLY
**Distribution**: Management + Security Team
EOF
```

- [ ] Incident report completed
- [ ] Management briefing conducted
- [ ] Action items assigned

#### Step 5.2: Process Improvements

**Security Enhancements**:
- [ ] Implement code signing for all commits
- [ ] Enable 2FA for all developers
- [ ] Add package integrity monitoring
- [ ] Increase audit logging
- [ ] Implement SIEM integration
- [ ] Schedule regular security audits

**Operational Improvements**:
- [ ] Update runbooks based on experience
- [ ] Conduct incident response training
- [ ] Improve detection capabilities
- [ ] Enhance communication procedures

---

## SCENARIO 2: UNAUTHORIZED REPOSITORY ACCESS

### Detection
Suspicious access patterns detected in repository logs or failed authentication attempts.

**Severity**: P1 - HIGH

### Immediate Actions

```bash
# 1. Review authentication logs
sudo grep "Failed\|Accepted" /var/log/auth.log | tail -100

# 2. Identify suspicious IPs
grep "Failed" /var/log/auth.log | awk '{print $(NF-3)}' | sort | uniq -c | sort -rn | head -10

# 3. Block suspicious IPs immediately
SUSPICIOUS_IP="192.168.1.100"
sudo iptables -A INPUT -s ${SUSPICIOUS_IP} -j DROP

# 4. Review repository access
grep ${SUSPICIOUS_IP} /var/log/apache2/access.log

# 5. Check for unauthorized package additions
cd /home/john/LAT5150DRVMIL/deployment/apt-repository
find pool/main -mtime -1 -ls

# 6. Verify package integrity
sha256sum pool/main/*.deb > /tmp/current_hashes.txt
diff baseline_hashes.txt /tmp/current_hashes.txt

# 7. If packages modified, follow SCENARIO 1 procedures

# 8. Rotate access credentials
# Change passwords for all repository admins
# Rotate GPG keys if necessary

# 9. Enable additional logging
sudo auditctl -w /home/john/LAT5150DRVMIL/deployment/apt-repository -p rwa -k repo_access
```

- [ ] Suspicious access blocked
- [ ] No unauthorized packages found / Compromised packages removed
- [ ] Credentials rotated
- [ ] Enhanced logging enabled

---

## SCENARIO 3: DSMIL QUARANTINE VIOLATION ATTEMPT

### Detection
System attempts to access quarantined DSMIL devices (0x8055, 0x8056, 0x8057, 0x8058, 0x8066).

**Severity**: P1 - HIGH

### Immediate Actions

```bash
# 1. Check quarantine logs
sudo dmesg | grep -i "quarantine\|access denied"
sudo journalctl -u dsmil-monitor | grep -i "quarantine"

# 2. Identify process attempting access
sudo audit search -m AVC -ts recent | grep dsmil
ps aux | grep [offending-pid]

# 3. Terminate offending process
sudo kill -9 [offending-pid]

# 4. Block process from restarting
PROCESS_PATH="/path/to/offending/process"
chmod 000 ${PROCESS_PATH}

# 5. Review DSMIL access logs
sudo /usr/share/dell-milspec/monitoring/dsmil_comprehensive_monitor.py --audit

# 6. Check if quarantine was breached
sudo dsmil-status | grep -A10 "Quarantined Devices"

# 7. If breach detected, trigger emergency stop
sudo /usr/sbin/milspec-emergency-stop

# 8. Secure the system
sudo systemctl stop dsmil-72dev
sudo modprobe -r dsmil-72dev

# 9. Collect forensics
mkdir -p /tmp/quarantine-violation-forensics
cp /var/log/dsmil.log /tmp/quarantine-violation-forensics/
dmesg > /tmp/quarantine-violation-forensics/dmesg.log
ps aux > /tmp/quarantine-violation-forensics/processes.txt

# 10. Notify security team
mail -s "ALERT: DSMIL Quarantine Violation Attempt" \
     security@dell-milspec.local < /tmp/quarantine-violation-forensics/summary.txt
```

- [ ] Violation detected and logged
- [ ] Offending process terminated
- [ ] Quarantine integrity verified
- [ ] Security team notified
- [ ] Forensics collected

---

## SCENARIO 4: THERMAL RUNAWAY / EMERGENCY STOP

### Detection
System temperature exceeds 100°C or emergency stop triggered.

**Severity**: P0 - CRITICAL (hardware safety)

### Immediate Actions

```bash
# 1. Verify emergency stop status
systemctl status milspec-emergency-stop

# 2. Check temperature
sensors
cat /sys/class/thermal/thermal_zone*/temp

# 3. If emergency stop triggered:
# - All DSMIL operations are suspended
# - System is in safe mode

# 4. Investigate cause
dmesg | grep -i "thermal\|temperature\|emergency"
journalctl -u milspec-monitor | grep -i "thermal"

# 5. Check workload
uptime
ps aux | sort -nrk 3,3 | head -10  # Top CPU users

# 6. If thermal issue, reduce load
# Move workload to E-cores only
sudo taskset -a -cp 12-21 [pid]

# 7. Check cooling system
sudo ipmitool sensor list | grep -i temp
# Verify fans operational

# 8. Document incident
cat > /tmp/thermal_incident.txt << EOF
Thermal Incident Report
Date: $(date)
Peak Temperature: $(sensors | grep "Core" | sort -k3 -rn | head -1)
Emergency Stop Triggered: YES/NO
Workload: [DESCRIPTION]
Resolution: [DESCRIPTION]
EOF

# 9. Once cooled down, resume operations
sudo systemctl start milspec-monitor
sudo systemctl start tpm2-acceleration-early

# 10. Implement preventive measures
# - Review power management settings
# - Check for cooling system issues
# - Adjust workload distribution
```

- [ ] Emergency stop executed successfully
- [ ] System temperature normalized
- [ ] Cause identified: `________________`
- [ ] Preventive measures implemented

---

## SCENARIO 5: GPG KEY COMPROMISE

**See**: [REPOSITORY_MAINTENANCE.md Section 3.4](./REPOSITORY_MAINTENANCE.md#34-emergency-key-revocation)

**Severity**: P0 - CRITICAL

---

## INCIDENT COMMUNICATION TEMPLATES

### Initial Alert Template

```
SECURITY INCIDENT ALERT

Incident: [ID]
Severity: [P0/P1/P2/P3]
Time: [TIMESTAMP]
Status: ACTIVE

Summary:
[2-3 sentences describing incident]

Impact:
[Affected systems/services]

Actions Required:
1. [Action 1]
2. [Action 2]

Next Update: [TIME]
Incident Commander: [NAME]
Contact: [EMAIL/PHONE]
```

### Status Update Template

```
SECURITY INCIDENT UPDATE #[N]

Incident: [ID]
Time: [TIMESTAMP]
Status: [CONTAINED/INVESTIGATING/RECOVERING]

Progress:
- [Update 1]
- [Update 2]

Current Actions:
- [Action 1]
- [Action 2]

Next Update: [TIME]
```

### Resolution Template

```
SECURITY INCIDENT RESOLVED

Incident: [ID]
Resolution Time: [TIMESTAMP]
Duration: [TIME]

Summary:
[Brief incident summary]

Resolution:
[How it was resolved]

Impact:
[Final impact assessment]

Follow-up Actions:
- [Action 1]
- [Action 2]

Full Report: [ETA]
```

---

## VERIFICATION CHECKLIST

### During Incident
- [ ] Incident Commander assigned
- [ ] War room established
- [ ] Initial notification sent (< 15 min for P0)
- [ ] Containment actions executed
- [ ] Evidence preserved
- [ ] Status updates every [FREQUENCY]

### Post-Incident
- [ ] All affected systems remediated
- [ ] Root cause identified
- [ ] Incident report completed
- [ ] Lessons learned session held
- [ ] Process improvements identified
- [ ] Action items assigned and tracked
- [ ] Stakeholders briefed

---

## ESCALATION PATHS

**P0 - CRITICAL**:
1. On-call Engineer (< 5 min)
2. Security Lead (< 10 min)
3. Incident Commander (< 15 min)
4. Engineering Director (< 30 min)
5. VP Engineering / CISO (< 1 hour)
6. CEO (if major breach)

**P1 - HIGH**:
1. On-call Engineer (< 30 min)
2. Security Lead (< 1 hour)
3. Incident Commander (< 2 hours)

---

## RELATED PROCEDURES

- [EMERGENCY_PACKAGE_REMOVAL.md](./EMERGENCY_PACKAGE_REMOVAL.md) - Emergency package removal
- [HOTFIX_DEPLOYMENT.md](./HOTFIX_DEPLOYMENT.md) - Rapid security patching
- [REPOSITORY_MAINTENANCE.md](./REPOSITORY_MAINTENANCE.md) - Repository operations

---

## REVISION HISTORY

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2025-10-11 | Security Operations Team | Initial release |

---

**Classification**: UNCLASSIFIED // FOR OFFICIAL USE ONLY
**Review Date**: 2025-11-11
**Owner**: Security Operations Team

---

## APPENDIX: QUICK REFERENCE

### Emergency Contacts
- Security Hotline: [PHONE]
- On-Call Engineer: [PHONE]
- Incident Commander: [PHONE]

### Critical Commands
```bash
# Emergency stop
sudo /usr/sbin/milspec-emergency-stop

# Remove package from repository
cd /home/john/LAT5150DRVMIL/deployment/apt-repository
reprepro -b . remove stable [PACKAGE]

# Block IP
sudo iptables -A INPUT -s [IP] -j DROP

# Check system integrity
sha256sum pool/main/*.deb | diff - baseline.sha256
```

### Severity Decision Matrix
| Factor | P0 | P1 | P2 |
|--------|----|----|-------|
| Data Breach | ✓ | | |
| Active Exploit | ✓ | | |
| Quarantine Violation | | ✓ | |
| Failed Auth | | | ✓ |
| Policy Violation | | | ✓ |
