# Dell MIL-SPEC Platform Operational Runbooks

**Version**: 1.0.0
**Last Updated**: 2025-10-11
**Classification**: UNCLASSIFIED // FOR OFFICIAL USE ONLY

---

## Overview

This directory contains comprehensive operational runbooks for managing the Dell MIL-SPEC platform .deb package deployment system. These runbooks provide step-by-step procedures for emergency response, maintenance, and routine operations.

**Total Runbooks**: 5
**Total Pages**: ~110 pages
**Total Lines**: 4,338 lines
**Total Size**: 110 KB

---

## Runbook Index

### 1. EMERGENCY_PACKAGE_REMOVAL.md
**Size**: 17 KB | **Lines**: 678 | **Time to Execute**: 30-60 minutes

Emergency procedures for immediate removal of compromised or critically buggy packages from the APT repository.

**Use When**:
- Critical bug discovered in deployed package
- Security vulnerability requires immediate action
- Package causes system instability or crashes
- Compliance violation detected

**Key Sections**:
- Immediate Assessment (5 minutes)
- Repository Removal (10 minutes)
- User Notification (10 minutes)
- System Remediation (15 minutes)
- Post-Incident Review (30 minutes)

**Target Timeline**: Package removed from repository within 20 minutes

---

### 2. HOTFIX_DEPLOYMENT.md
**Size**: 24 KB | **Lines**: 1,015 | **Time to Execute**: 2-4 hours

Accelerated procedures for rapid deployment of security hotfixes when critical vulnerabilities require immediate patching.

**Use When**:
- CVE with CVSS score ≥ 8.0
- Active exploitation detected
- Privilege escalation vulnerability
- Remote code execution vulnerability
- Data leakage vulnerability

**Key Sections**:
- Assessment & Planning (15 minutes)
- Hotfix Development (30 minutes)
- Critical Testing (45 minutes) - Fast-track only
- Package Build & Sign (30 minutes)
- Repository Deployment (30 minutes)
- User Notification & Rollout (30 minutes)
- Monitoring & Verification (60 minutes)

**Target Timeline**: Hotfix deployed from discovery to production in < 4 hours

**Phases**:
```
0:00-0:15  →  Assessment
0:15-0:45  →  Development
0:45-1:30  →  Testing
1:30-2:00  →  Build & Sign
2:00-2:30  →  Deploy
2:30-3:00  →  Rollout
3:00-4:00  →  Monitor
```

---

### 3. KERNEL_COMPATIBILITY.md
**Size**: 19 KB | **Lines**: 760 | **Time to Execute**: 2-4 hours (automated) or 1 day (manual)

Procedures for testing, validating, and maintaining Dell MIL-SPEC DKMS packages across Linux kernel versions.

**Use When**:
- New Linux kernel version released (e.g., 6.17.0)
- Kernel API changes detected
- DKMS build failures occur
- Testing new kernel compatibility

**Key Sections**:
- Detection & Preparation (15 minutes)
- Automated Compatibility Test (30 minutes)
- Fixing Compatibility Issues (varies)
- Package Update (30 minutes)
- Documentation & Deployment (30 minutes)

**Kernel Support Matrix**:
| Kernel | DSMIL | TPM2 | Status |
|--------|-------|------|--------|
| 6.14.0 | ✅ | ✅ | STABLE |
| 6.15.x | ✅ | ✅ | STABLE |
| 6.16.x | ✅ | ✅ | STABLE |
| 6.17.x | ⏳ | ⏳ | TESTING |

**Features**:
- Automated CI/CD detection of new kernels
- Compatibility testing suite
- Version-specific compatibility layers
- Backward compatibility verification

---

### 4. REPOSITORY_MAINTENANCE.md
**Size**: 22 KB | **Lines**: 852 | **Maintenance Schedule**: Daily/Weekly/Monthly

Comprehensive procedures for maintaining the Dell MIL-SPEC APT repository, including package management, cleanup, GPG key operations, and backup.

**Maintenance Schedule**:

**Daily Tasks** (Automated):
- Monitor repository access logs
- Check disk space usage
- Verify repository integrity
- Monitor download metrics

**Weekly Tasks** (Manual):
- Review and clean old package versions
- Audit package signatures
- Check for repository corruption
- Review access logs for anomalies

**Monthly Tasks** (Manual):
- Full repository backup
- Security audit
- GPG key status review
- Capacity planning review

**Quarterly Tasks** (Manual):
- GPG key rotation consideration
- Repository structure optimization
- Access policy review
- Disaster recovery test

**Key Sections**:
1. Package Management (add, remove, update, copy)
2. Repository Cleanup (disk space, old versions, database)
3. GPG Key Management (status, extension, rotation, revocation)
4. Repository Backup & Recovery
5. Monitoring & Logging
6. Troubleshooting

**Repository Structure**:
```
apt-repository/
├── pool/main/              # Package files
├── dists/                  # Distribution metadata
│   ├── stable/
│   ├── testing/
│   └── unstable/
├── conf/                   # Configuration
├── db/                     # reprepro database
├── gpg/                    # GPG keys
└── scripts/                # Management scripts
```

---

### 5. INCIDENT_RESPONSE.md
**Size**: 28 KB | **Lines**: 1,033 | **Response Time**: 15 min - 7 days

Comprehensive incident response procedures for security events involving the Dell MIL-SPEC platform.

**Severity Levels**:
| Level | Description | Response Time |
|-------|-------------|---------------|
| **P0** | Active exploit, data breach | < 15 minutes |
| **P1** | Potential compromise | < 1 hour |
| **P2** | Suspicious activity | < 4 hours |
| **P3** | Policy violation | < 24 hours |

**Covered Scenarios**:

1. **Compromised Package Detected** (P0)
   - Package contains malware, backdoor, or unauthorized modification
   - Timeline: Detection → Containment (5 min) → Investigation (4 hours)

2. **Unauthorized Repository Access** (P1)
   - Suspicious access patterns or failed authentication attempts
   - Timeline: Detection → Blocking (15 min) → Audit (2 hours)

3. **DSMIL Quarantine Violation Attempt** (P1)
   - Attempts to access quarantined military devices
   - Timeline: Detection → Termination (5 min) → Forensics (1 hour)

4. **Thermal Runaway / Emergency Stop** (P0)
   - System temperature exceeds 100°C or emergency stop triggered
   - Timeline: Detection → Emergency Stop (< 85 ms) → Recovery (1 hour)

5. **GPG Key Compromise** (P0)
   - Repository signing key compromised
   - Timeline: Detection → Revocation (15 min) → Key Rotation (4 hours)

**Incident Response Phases**:
1. **Detection** (0-5 min) - Alert received, severity assessed
2. **Containment** (5-30 min) - Threat isolated, evidence preserved
3. **Investigation** (1-4 hours) - Root cause analysis, scope determination
4. **Eradication** (4-8 hours) - Threat removal, vulnerability patching
5. **Recovery** (8-24 hours) - System restoration, service resumption
6. **Post-Incident** (1-7 days) - Report, lessons learned, improvements

---

## Quick Reference Guide

### When to Use Which Runbook

```
┌─────────────────────────────────────────────────────────────┐
│ START: What's the situation?                                │
└─────────────────────────────────────────────────────────────┘
                           ↓
    ┌──────────────────────┴──────────────────────┐
    │                                              │
    ↓ Bug/Issue                          Security ↓
┌─────────────────┐                    ┌──────────────────┐
│ Package has     │                    │ Package          │
│ critical bug    │                    │ compromised      │
└────────┬────────┘                    └────────┬─────────┘
         │                                      │
         ↓                                      ↓
┌─────────────────┐                    ┌──────────────────┐
│ EMERGENCY       │                    │ INCIDENT         │
│ PACKAGE REMOVAL │                    │ RESPONSE         │
│ (30-60 min)     │                    │ (hours-days)     │
└─────────────────┘                    └──────────────────┘

┌─────────────────┐                    ┌──────────────────┐
│ Need security   │                    │ New kernel       │
│ patch NOW       │                    │ released         │
└────────┬────────┘                    └────────┬─────────┘
         │                                      │
         ↓                                      ↓
┌─────────────────┐                    ┌──────────────────┐
│ HOTFIX          │                    │ KERNEL           │
│ DEPLOYMENT      │                    │ COMPATIBILITY    │
│ (<4 hours)      │                    │ (2-4 hours)      │
└─────────────────┘                    └──────────────────┘

┌─────────────────┐
│ Routine         │
│ maintenance     │
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│ REPOSITORY      │
│ MAINTENANCE     │
│ (scheduled)     │
└─────────────────┘
```

---

## Emergency Contact Information

**Security Team**:
- On-call: [PHONE NUMBER]
- Email: security@dell-milspec.local
- Slack: #security-oncall

**Operations Team**:
- On-call: [PHONE NUMBER]
- Email: ops@dell-milspec.local
- Slack: #ops-oncall

**Incident Commander**:
- Primary: [NAME] - [CONTACT]
- Secondary: [NAME] - [CONTACT]

**Management Escalation**:
- Engineering Director: [NAME] - [CONTACT]
- VP Engineering: [NAME] - [CONTACT]
- CISO: [NAME] - [CONTACT]

---

## Critical Commands Quick Reference

### Emergency Package Removal
```bash
cd /home/john/LAT5150DRVMIL/deployment/apt-repository
reprepro -b . remove stable [PACKAGE_NAME]
./scripts/update-repository.sh
```

### Emergency Stop (Thermal/Security)
```bash
sudo /usr/sbin/milspec-emergency-stop
```

### Repository Integrity Check
```bash
cd /home/john/LAT5150DRVMIL/deployment/apt-repository
sha256sum pool/main/*.deb | diff - baseline.sha256
```

### GPG Key Revocation (Emergency)
```bash
gpg --output revoke.asc --gen-revoke [KEY-ID]
gpg --import revoke.asc
gpg --keyserver keyserver.ubuntu.com --send-keys [KEY-ID]
```

### Block Suspicious IP
```bash
sudo iptables -A INPUT -s [SUSPICIOUS_IP] -j DROP
```

### Check DSMIL Quarantine Status
```bash
sudo dsmil-status | grep -A10 "Quarantined Devices"
dmesg | grep -i "quarantine\|access denied"
```

---

## Runbook Dependencies

```
INCIDENT_RESPONSE.md (master runbook)
    ├─→ EMERGENCY_PACKAGE_REMOVAL.md (Scenario 1: Compromised Package)
    ├─→ HOTFIX_DEPLOYMENT.md (Resolution: Deploy Clean Package)
    └─→ REPOSITORY_MAINTENANCE.md (Section 3.4: GPG Key Revocation)

HOTFIX_DEPLOYMENT.md
    └─→ KERNEL_COMPATIBILITY.md (If kernel-specific fix required)

REPOSITORY_MAINTENANCE.md
    ├─→ EMERGENCY_PACKAGE_REMOVAL.md (Referenced for emergency procedures)
    └─→ KERNEL_COMPATIBILITY.md (Package compatibility tracking)

KERNEL_COMPATIBILITY.md
    └─→ HOTFIX_DEPLOYMENT.md (Deploy compatibility patches)

EMERGENCY_PACKAGE_REMOVAL.md
    ├─→ HOTFIX_DEPLOYMENT.md (Deploy replacement package)
    └─→ INCIDENT_RESPONSE.md (Security incident procedures)
```

---

## Training Requirements

All operations team members should be trained on:

**Required (All Team Members)**:
- [ ] REPOSITORY_MAINTENANCE.md (Sections 1-2)
- [ ] EMERGENCY_PACKAGE_REMOVAL.md (Full)
- [ ] KERNEL_COMPATIBILITY.md (Sections 1-2)

**Security Team Only**:
- [ ] INCIDENT_RESPONSE.md (Full)
- [ ] REPOSITORY_MAINTENANCE.md (Section 3: GPG)
- [ ] HOTFIX_DEPLOYMENT.md (Full)

**Senior Engineers Only**:
- [ ] All runbooks (Full)
- [ ] Cross-runbook scenario training
- [ ] Incident Commander role training

**Recommended Training Schedule**:
- Initial training: 8 hours (2 days)
- Refresher training: Quarterly (2 hours)
- Tabletop exercises: Semi-annually
- DR test participation: Quarterly

---

## Runbook Versioning

All runbooks follow semantic versioning:
- **Major** (x.0.0): Significant procedure changes
- **Minor** (0.x.0): New sections or enhancements
- **Patch** (0.0.x): Corrections, clarifications

**Current Version**: 1.0.0

**Review Schedule**:
- Security runbooks: Monthly
- Maintenance runbooks: Quarterly
- All runbooks: After each major incident

---

## Testing & Validation

**Runbook Testing Schedule**:

| Runbook | Test Frequency | Last Tested | Next Test |
|---------|----------------|-------------|-----------|
| EMERGENCY_PACKAGE_REMOVAL | Quarterly | TBD | TBD |
| HOTFIX_DEPLOYMENT | Semi-annually | TBD | TBD |
| KERNEL_COMPATIBILITY | Per kernel release | TBD | TBD |
| REPOSITORY_MAINTENANCE | Quarterly (DR) | TBD | TBD |
| INCIDENT_RESPONSE | Annually (tabletop) | TBD | TBD |

---

## Metrics & KPIs

### Response Time Metrics
- **Emergency Package Removal**: < 20 minutes (target)
- **Hotfix Deployment**: < 4 hours (target)
- **Kernel Compatibility Fix**: < 1 day (target)
- **P0 Incident Response**: < 15 minutes (target)
- **P1 Incident Response**: < 1 hour (target)

### Quality Metrics
- **Runbook Accuracy**: > 95% (procedures work as documented)
- **Team Preparedness**: > 90% (training completion rate)
- **Incident Resolution Rate**: > 95% (successful incident closures)
- **User Satisfaction**: > 85% (post-incident surveys)

---

## Document Standards

All runbooks follow these standards:

**Structure**:
1. Header (title, version, classification)
2. Purpose statement
3. Scenario description
4. Prerequisites
5. Step-by-step procedures with checkboxes
6. Verification checklist
7. Rollback procedures
8. Related procedures
9. Revision history

**Formatting**:
- Clear hierarchical sections
- Checkboxes for tracking progress
- Code blocks for all commands
- Expected outputs documented
- Time estimates for each phase
- Escalation paths clearly defined

**Content Requirements**:
- Real examples using actual package names
- Actual file paths from the project
- Actual script names
- Actual commands (no placeholders)
- Clear success/failure criteria

---

## Contributing to Runbooks

### Suggesting Changes
1. Create issue: `runbook-improvement-[name]`
2. Describe proposed change
3. Provide justification (incident, near-miss, improvement)
4. Submit for review

### Updating After Incidents
After every major incident:
1. Document what worked / didn't work
2. Update relevant runbook sections
3. Increment version number
4. Update "Last Tested" date
5. Notify team of changes

### Approval Process
- **Minor changes**: Ops Lead approval
- **Major changes**: Security Lead + Ops Lead approval
- **Security runbooks**: CISO approval required

---

## File Locations

All runbooks are located at:
```
/home/john/LAT5150DRVMIL/deployment/runbooks/
```

**Repository Structure**:
```
deployment/
├── runbooks/
│   ├── README.md                        # This file
│   ├── EMERGENCY_PACKAGE_REMOVAL.md     # Emergency procedures
│   ├── HOTFIX_DEPLOYMENT.md             # Rapid security patching
│   ├── KERNEL_COMPATIBILITY.md          # Kernel version handling
│   ├── REPOSITORY_MAINTENANCE.md        # Repository operations
│   └── INCIDENT_RESPONSE.md             # Security incident handling
├── scripts/                             # Automation scripts
├── apt-repository/                      # APT repository
└── docs/                                # Additional documentation
```

---

## Revision History

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 1.0.0 | 2025-10-11 | Initial release of all 5 runbooks | Operations Team |

---

## Additional Resources

### Internal Documentation
- [FOUNDATION_COMPLETE.md](../FOUNDATION_COMPLETE.md) - Platform overview
- [PACKAGE_SUMMARY.md](../../packaging/PACKAGE_SUMMARY.md) - Package details
- [BUILD_REPORT.md](../../packaging/BUILD_REPORT.md) - Build procedures

### External Resources
- Debian Package Management: https://wiki.debian.org/DebianRepository
- reprepro Documentation: https://wiki.debian.org/DebianRepository/SetupWithReprepro
- DKMS Documentation: https://github.com/dell/dkms
- Incident Response Best Practices: NIST SP 800-61r2

---

**Classification**: UNCLASSIFIED // FOR OFFICIAL USE ONLY
**Maintained By**: Operations Team
**Last Review**: 2025-10-11
**Next Review**: 2025-11-11
