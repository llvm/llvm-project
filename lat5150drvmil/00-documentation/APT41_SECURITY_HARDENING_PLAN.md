# APT-41 THREAT MITIGATION PLAN
**Target**: Dell Latitude 5450 - Custom Kernel 6.16.9-milspec
**Threat Actor**: APT-41 (Chinese State-Sponsored)
**Attack Vectors Experienced**:
- Keylogger attacks (KEYPLUGGED)
- Image-based malware exploitation
- PDF-based malware exploitation
- VM escape vulnerabilities
- DMA attacks via Thunderbolt (PDF-to-Arc GPU)

---

## IMMEDIATE KERNEL HARDENING

### 1. DMA Attack Protection (Thunderbolt/USB)
**Attack**: PDF triggered DMA attack to Intel Arc GPU via Thunderbolt

**Kernel Boot Parameters** (CRITICAL):
```bash
# /etc/default/grub - GRUB_CMDLINE_LINUX_DEFAULT:
intel_iommu=on iommu=pt
thunderbolt.security=user
pci=noaer
module.sig_enforce=1
lockdown=confidentiality
```

**Purpose**:
- `intel_iommu=on`: Enable IOMMU for DMA protection
- `iommu=pt`: Passthrough mode for performance + security
- `thunderbolt.security=user`: Require user authorization for TB devices
- `module.sig_enforce=1`: Only signed modules can load
- `lockdown=confidentiality`: Maximum kernel lockdown mode

### 2. Keylogger Mitigation (KEYPLUGGED)
**Attack**: APT-41 keylogger implant

**Kernel Features**:
```
CONFIG_SECURITY_DMESG_RESTRICT=y     # Restrict kernel logs
CONFIG_SECURITY_YAMA=y                # Ptrace restrictions
CONFIG_SECURITY_LOADPIN=y             # Pin module loading location
CONFIG_MODULE_SIG_FORCE=y             # Force module signatures
```

**Additional Protections**:
- TPM2-backed module verification (using our custom tpm2_accel_npu.o)
- DSMIL hardware monitoring (79/84 devices tracked)
- Kernel lockdown mode (prevents runtime kernel modification)

### 3. Image/PDF Exploit Mitigation
**Attack**: Malicious image/PDF files exploiting parsers

**Kernel Hardening**:
```
CONFIG_HARDENED_USERCOPY=y           # Protect kernel from user data
CONFIG_FORTIFY_SOURCE=y              # Buffer overflow protection
CONFIG_SLAB_FREELIST_RANDOM=y        # Randomize heap allocations
CONFIG_INIT_ON_ALLOC_DEFAULT_ON=y    # Zero memory on allocation
CONFIG_INIT_ON_FREE_DEFAULT_ON=y     # Zero memory on free
```

**Userspace Requirements** (Post-Install):
- Firejail sandboxing for PDF readers
- AppArmor profiles for image viewers
- SELinux confinement for media applications

### 4. VM Escape Prevention
**Attack**: Hypervisor escape exploit

**Kernel Features**:
```
CONFIG_PAGE_TABLE_ISOLATION=y        # Meltdown mitigation
CONFIG_RETPOLINE=y                   # Spectre v2 mitigation
CONFIG_X86_KERNEL_IBT=y              # Indirect branch tracking
CONFIG_VMAP_STACK=y                  # Stack overflow protection
CONFIG_STACKPROTECTOR_STRONG=y       # Enhanced stack protection
```

**Intel-Specific Mitigations**:
- Hardware-backed CFI (Control Flow Integrity)
- Shadow stack support for Intel Arc GPU isolation
- IOMMU isolation for GPU memory access

---

## POST-INSTALL SECURITY CONFIGURATION

### 1. DSMIL Security Monitoring
**Capability**: 79/84 military-grade devices accessible

**Active Monitoring**:
```bash
# Enable DSMIL security monitoring daemon
systemctl enable dsmil-security-monitor.service

# Monitor SMI interface for unauthorized access
/opt/dsmil-framework/bin/smi-monitor --realtime
```

### 2. TPM2 Attestation
**Hardware**: STMicroelectronics ST33TPHF2XSP

**Boot Attestation**:
```bash
# Verify boot integrity with TPM2
tpm2_pcrread sha256:0,1,2,3,4,5,6,7

# Expected PCRs for secure boot:
# PCR 0: BIOS/UEFI firmware
# PCR 1: BIOS/UEFI configuration
# PCR 2: Option ROM code
# PCR 3: Option ROM configuration
# PCR 4: Boot loader (GRUB)
# PCR 5: Boot loader configuration
# PCR 7: Secure boot state
```

**Continuous Monitoring**:
- TPM-based remote attestation
- NPU-accelerated crypto verification
- Hardware-backed key storage

### 3. Network Security Isolation
**Protection**: Prevent command & control exfiltration

```bash
# Firewall rules (nftables)
nft add rule inet filter input ct state established,related accept
nft add rule inet filter input iif lo accept
nft add rule inet filter input drop  # Default deny

# DNS over HTTPS (prevent DNS hijacking)
systemd-resolved --set-dns-over-tls=opportunistic

# VPN enforcement (if applicable)
# Only allow traffic through VPN tunnel
```

### 4. File System Security
**Protection**: Prevent malicious file execution

```bash
# Mount /tmp with noexec (prevent execution from tmp)
mount -o remount,noexec,nosuid,nodev /tmp

# Enable file integrity monitoring (IMA/EVM)
echo 1 > /sys/kernel/security/ima/policy

# AppArmor enforcement mode
aa-enforce /etc/apparmor.d/*
```

### 5. Sandboxing Critical Applications
**Applications at Risk**: PDF readers, image viewers, browsers

```bash
# Firejail profiles
firejail --profile=/etc/firejail/evince.profile evince document.pdf
firejail --profile=/etc/firejail/firefox.profile firefox

# AppArmor confinement
aa-enforce /etc/apparmor.d/usr.bin.evince
aa-enforce /etc/apparmor.d/usr.bin.eog  # Image viewer
```

---

## KERNEL FEATURES VERIFICATION CHECKLIST

### Memory Protection
- [ ] `CONFIG_HARDENED_USERCOPY=y` - Protect kernel from user data
- [ ] `CONFIG_SLAB_FREELIST_RANDOM=y` - Heap randomization
- [ ] `CONFIG_INIT_ON_ALLOC_DEFAULT_ON=y` - Zero on alloc
- [ ] `CONFIG_INIT_ON_FREE_DEFAULT_ON=y` - Zero on free
- [ ] `CONFIG_PAGE_TABLE_ISOLATION=y` - Meltdown protection

### Control Flow Integrity
- [ ] `CONFIG_RETPOLINE=y` - Spectre v2 mitigation
- [ ] `CONFIG_X86_KERNEL_IBT=y` - Indirect branch tracking
- [ ] `CONFIG_X86_USER_SHADOW_STACK=y` - Hardware shadow stack
- [ ] `CONFIG_VMAP_STACK=y` - Stack overflow protection
- [ ] `CONFIG_STACKPROTECTOR_STRONG=y` - Stack canaries

### Module Security
- [ ] `CONFIG_MODULE_SIG=y` - Module signing
- [ ] `CONFIG_MODULE_SIG_FORCE=y` - Force signature verification
- [ ] `CONFIG_SECURITY_LOADPIN=y` - Pin module load location
- [ ] `CONFIG_MODULE_SIG_HASH="sha256"` - SHA-256 signatures

### DMA Protection
- [ ] `CONFIG_INTEL_IOMMU=y` - IOMMU support
- [ ] `CONFIG_INTEL_IOMMU_SVM=y` - Shared virtual memory
- [ ] Boot param: `intel_iommu=on` - Enable at boot
- [ ] Boot param: `iommu=pt` - Passthrough mode
- [ ] Boot param: `thunderbolt.security=user` - TB authorization

### Lockdown Features
- [ ] `CONFIG_SECURITY_LOCKDOWN_LSM=y` - Lockdown LSM
- [ ] `CONFIG_SECURITY_LOCKDOWN_LSM_EARLY=y` - Early lockdown
- [ ] Boot param: `lockdown=confidentiality` - Max lockdown

---

## APT-41 SPECIFIC COUNTERMEASURES

### 1. Keylogger Detection
**Tools**:
```bash
# Hardware keylogger detection via DSMIL
/opt/dsmil-framework/bin/hardware-scan --usb-devices

# Software keylogger detection
rkhunter --check --enable all
chkrootkit

# Monitor for kernel module injection
lsmod | grep -v "^Module" | awk '{print $1}' | sort > /tmp/modules.txt
# Compare against known-good baseline
```

### 2. Image/PDF Quarantine
**Workflow**:
```bash
# Quarantine directory for untrusted files
mkdir -p /quarantine/{images,pdfs}
chmod 1777 /quarantine/*

# Scan with multiple engines before opening
clamav /quarantine/pdfs/suspicious.pdf
yara-scan /quarantine/pdfs/suspicious.pdf

# Open in isolated VM if available
firejail --net=none --private evince /quarantine/pdfs/suspicious.pdf
```

### 3. Network Traffic Analysis
**Monitoring**:
```bash
# Capture suspicious network patterns
tcpdump -i any -w /var/log/network-$(date +%Y%m%d).pcap

# Real-time IDS
suricata -c /etc/suricata/suricata.yaml -i eth0

# DNS monitoring for C2 domains
pihole or unbound with blocklists
```

### 4. GPU Memory Isolation
**Intel Arc Protection**:
```bash
# IOMMU groups for GPU
find /sys/kernel/iommu_groups/ -type l

# Verify GPU is in isolated IOMMU group
lspci -vv | grep -A 20 "VGA compatible controller"

# Check IOMMU protection active
dmesg | grep -i iommu
```

---

## INCIDENT RESPONSE AUTOMATION

### 1. TPM-Based Tamper Detection
```bash
#!/bin/bash
# /usr/local/bin/tpm-integrity-check.sh

# Get current PCR values
CURRENT_PCRS=$(tpm2_pcrread sha256:0,1,2,3,4,5,6,7 | sha256sum)

# Compare against known-good baseline
BASELINE=$(cat /etc/tpm-baseline.sha256)

if [ "$CURRENT_PCRS" != "$BASELINE" ]; then
    echo "ALERT: Boot integrity compromised!" | logger -p security.crit
    # Trigger incident response
    /usr/local/bin/incident-response.sh --boot-tamper
fi
```

### 2. DSMIL Anomaly Detection
```bash
#!/bin/bash
# /usr/local/bin/dsmil-anomaly-check.sh

# Check for unauthorized device access
CURRENT_DEVICES=$(/opt/dsmil-framework/bin/device-enum | wc -l)
BASELINE=79  # Known good: 79/84 devices

if [ "$CURRENT_DEVICES" -lt "$BASELINE" ]; then
    echo "ALERT: DSMIL device count anomaly!" | logger -p security.warn
    # Possible hardware tampering
fi

# Monitor SMI interface for unusual activity
SMI_CALLS=$(cat /proc/dsmil_stats | grep smi_calls | awk '{print $2}')
if [ "$SMI_CALLS" -gt 10000 ]; then
    echo "ALERT: Excessive SMI calls detected!" | logger -p security.warn
fi
```

### 3. NPU Crypto Validation
```bash
#!/bin/bash
# /usr/local/bin/npu-crypto-verify.sh

# Verify NPU is handling crypto correctly
# Use TPM2 NPU acceleration module (tpm2_accel_npu.o)

# Test NPU crypto performance
PERF=$(/opt/dsmil-framework/bin/npu-benchmark --crypto)

# If NPU performance degraded, possible tampering
if [ "$PERF" -lt 30 ]; then  # 30+ TOPS expected
    echo "ALERT: NPU performance anomaly!" | logger -p security.warn
    # May indicate NPU compromise
fi
```

---

## RECOMMENDED ADDITIONAL TOOLS

### 1. Sandboxing & Isolation
```bash
apt-get install firejail bubblewrap
apt-get install apparmor-profiles apparmor-utils
apt-get install selinux-policy-default
```

### 2. Security Scanning
```bash
apt-get install clamav clamav-daemon
apt-get install rkhunter chkrootkit
apt-get install aide tripwire
apt-get install lynis
```

### 3. Network Security
```bash
apt-get install suricata snort
apt-get install wireshark tshark
apt-get install nftables iptables-persistent
```

### 4. Forensics & Monitoring
```bash
apt-get install sysstat auditd
apt-get install osquery  # Facebook's security monitoring
apt-get install volatility3  # Memory forensics
```

---

## BOOT CONFIGURATION - MAXIMUM SECURITY

### GRUB Configuration
**/etc/default/grub**:
```bash
GRUB_CMDLINE_LINUX_DEFAULT="quiet splash \
  intel_iommu=on iommu=pt \
  thunderbolt.security=user \
  pci=noaer \
  module.sig_enforce=1 \
  lockdown=confidentiality \
  page_alloc.shuffle=1 \
  init_on_alloc=1 init_on_free=1 \
  slab_nomerge \
  mce=0 \
  pti=on \
  spec_store_bypass_disable=on \
  tsx=off \
  vsyscall=none \
  kptr_restrict=2 \
  slub_debug=FZ \
  debugfs=off"
```

**Parameter Explanations**:
- `intel_iommu=on iommu=pt`: DMA protection (Thunderbolt attacks)
- `thunderbolt.security=user`: Manual TB device authorization
- `module.sig_enforce=1`: Only signed kernel modules
- `lockdown=confidentiality`: Prevent kernel runtime modification
- `init_on_alloc=1 init_on_free=1`: Zero memory (prevent info leaks)
- `pti=on`: Page table isolation (Meltdown)
- `spec_store_bypass_disable=on`: Spectre v4 mitigation
- `tsx=off`: Disable TSX (side-channel attacks)
- `vsyscall=none`: Disable legacy vsyscall (ROP mitigation)
- `kptr_restrict=2`: Hide kernel pointers
- `debugfs=off`: Disable debug filesystem

---

## MONITORING DASHBOARD

### Systemd Services to Enable
```bash
# TPM integrity monitoring (every 5 minutes)
systemctl enable tpm-integrity-check.timer

# DSMIL anomaly detection (every 1 minute)
systemctl enable dsmil-anomaly-check.timer

# NPU crypto validation (every 10 minutes)
systemctl enable npu-crypto-verify.timer

# Audit daemon (real-time)
systemctl enable auditd.service

# IMA/EVM integrity (boot-time)
systemctl enable ima-evm-initialize.service
```

### Log Aggregation
```bash
# Central security logging
journalctl -f -u auditd -u tpm-integrity-check -u dsmil-anomaly-check \
  | grep -i "alert\|critical\|security"
```

---

## SUMMARY

**Threat Level**: APT-41 (EXTREME - State-Sponsored)

**Attack Vectors Addressed**:
1. ✅ Keylogger (KEYPLUGGED) - Lockdown, module signing, Yama
2. ✅ Image exploits - Hardened usercopy, heap randomization, sandboxing
3. ✅ PDF exploits - Memory zeroing, AppArmor, Firejail isolation
4. ✅ VM escape - PTI, Retpoline, IBT, shadow stack
5. ✅ DMA attacks - IOMMU, Thunderbolt security, PCI isolation

**Hardware-Backed Security**:
- TPM2 boot attestation (ST33TPHF2XSP)
- DSMIL 84-device monitoring framework
- NPU crypto acceleration (34 TOPS)
- Intel Arc GPU IOMMU isolation

**Post-Install Priority**:
1. Configure GRUB with maximum security boot parameters
2. Enable TPM2 attestation monitoring
3. Deploy AppArmor/SELinux profiles for all user applications
4. Setup network IDS/IPS (Suricata)
5. Create TPM baseline and automated integrity checks

**CRITICAL**: This kernel + hardening will provide defense-in-depth against APT-41 tactics. However, continuous monitoring and rapid incident response are essential.

---

**Last Updated**: 2025-10-15 04:00 GMT
**Security Level**: MAXIMUM HARDENING
**Threat Actor**: APT-41 (Chinese State-Sponsored)
