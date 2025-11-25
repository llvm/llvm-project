# DBXForensics AI-Guided Installation Guide

**Complete guide for AI to assist users with automated installation and setup**

This document provides step-by-step instructions for the AI assistant to guide users through installing and configuring the DBXForensics integration with Xen VM.

---

## Overview

The AI can guide users through:

1. **Prerequisites Check** - Verify system requirements
2. **Windows Server Core VM Setup** - Download ISO, create VM, install Windows
3. **DBXForensics Tools Installation** - Install all 9 tools on VM
4. **RPC Server Setup** - Configure Flask server for Dom0 communication
5. **Shared Folder Configuration** - Mount SMB/CIFS folders
6. **Integration Testing** - Verify end-to-end functionality
7. **Performance Optimization** - Apply Xen PV drivers and tuning

---

## Phase 1: Prerequisites Check

### AI Instructions

When a user asks to set up forensics tools, first check prerequisites:

```bash
# Check if Xen is installed
xl list

# Check available disk space (need 30GB)
df -h /

# Check available memory (need 4GB free for VM)
free -h

# Check network configuration
ip addr show
```

### Expected Output

- Xen should show Dom0 running
- At least 30GB free disk space
- At least 4GB free RAM
- Network interface configured (typically xenbr0)

### AI Response Template

```
I'll help you set up the DBXForensics environment with Xen VM.

✓ Prerequisites check:
  - Xen hypervisor: [FOUND/NOT FOUND]
  - Disk space: [XXG available / 30GB required]
  - RAM: [XXG available / 4GB required]
  - Network: [CONFIGURED/NOT CONFIGURED]

[If all pass]:
Great! Your system meets all requirements. Let's proceed with installation.

[If any fail]:
We need to address the following issues first:
  - [List issues and solutions]
```

---

## Phase 2: Windows Server Core ISO Download

### AI Instructions

Guide user to download Windows Server Core ISO:

### AI Response Template

```
Next, we need to download Windows Server Core ISO:

Option 1: Official Microsoft Download
  1. Visit: https://www.microsoft.com/en-us/evalcenter/evaluate-windows-server-2022
  2. Select "ISO - Server Core"
  3. Download to: /tmp/windows-server-core.iso
  4. Expected size: ~5GB

Option 2: Use existing evaluation ISO
  If you already have Windows Server 2022 ISO, we can use that.

Once downloaded, run:
  ls -lh /tmp/windows-server-core.iso

Let me know when the ISO is ready!
```

---

## Phase 3: Xen VM Creation

### AI Instructions

Once ISO is ready, create the VM:

```bash
# Create VM storage (30GB)
sudo lvcreate -L 30G -n forensics-vm vg0
# Or use file-based storage:
sudo dd if=/dev/zero of=/var/lib/xen/images/forensics-vm.img bs=1G count=30

# Create VM configuration
sudo tee /etc/xen/forensics-vm.cfg > /dev/null << 'EOF'
# Forensics VM Configuration
name = "forensics-vm"
type = "hvm"
memory = 4096
vcpus = 2
vif = ['bridge=xenbr0']

# Disk configuration
disk = [
    'phy:/dev/vg0/forensics-vm,xvda,w',
    'file:/tmp/windows-server-core.iso,xvdc:cdrom,r'
]

# Boot from CD first, then disk
boot = "dc"

# VNC configuration for installation
vnc = 1
vnclisten = "0.0.0.0"
vncpasswd = "forensics"

# Enable PV drivers (install after Windows setup)
viridian = 1
EOF

# Create VM
sudo xl create /etc/xen/forensics-vm.cfg

# Check VM status
sudo xl list
```

### AI Response Template

```
Creating Xen VM for forensics tools...

VM Configuration:
  Name: forensics-vm
  RAM: 4GB
  vCPUs: 2
  Disk: 30GB
  ISO: /tmp/windows-server-core.iso

Commands to run:
  [Provide commands above]

Once the VM is created, access it via VNC:
  IP: [Dom0 IP address]
  Port: 5900
  Password: forensics

VNC Clients:
  - Linux: vncviewer [IP]:5900
  - Windows: TightVNC, RealVNC
  - Mac: Screen Sharing app

Let me know when you're connected to VNC!
```

---

## Phase 4: Windows Server Core Installation

### AI Instructions

Guide user through Windows installation:

### AI Response Template

```
Windows Server Core Installation Steps:

Via VNC console:

1. **Language Selection**
   - Select language, time, keyboard
   - Click "Next"

2. **Install Now**
   - Click "Install now"
   - Select "Windows Server 2022 Standard (Desktop Experience)" if GUI needed
   - OR "Windows Server 2022 Standard" for Core (no GUI, recommended)

3. **License Agreement**
   - Accept license terms
   - Click "Next"

4. **Installation Type**
   - Select "Custom: Install Windows only (advanced)"

5. **Disk Configuration**
   - Select the 30GB disk
   - Click "Next"
   - Installation begins (10-15 minutes)

6. **Administrator Password**
   - Set a strong password
   - Remember this - you'll need it!

7. **Login**
   - Press Ctrl+Alt+Delete (or menu option)
   - Login as Administrator

Once logged in, you'll see a command prompt (Server Core).

Let me know when Windows is installed!
```

---

## Phase 5: Windows Network Configuration

### AI Instructions

Configure static IP for VM:

### Windows Commands (run in VM console)

```powershell
# Show network adapters
Get-NetAdapter

# Set static IP (adjust to your network)
New-NetIPAddress -InterfaceAlias "Ethernet" -IPAddress 192.168.100.10 -PrefixLength 24 -DefaultGateway 192.168.100.1

# Set DNS
Set-DnsClientServerAddress -InterfaceAlias "Ethernet" -ServerAddresses 8.8.8.8,8.8.4.4

# Verify configuration
Get-NetIPAddress
Test-Connection -ComputerName google.com -Count 2
```

### AI Response Template

```
Configuring VM network (static IP: 192.168.100.10)...

Run these PowerShell commands in the VM:
  [Provide commands above]

This gives the VM:
  - Static IP: 192.168.100.10
  - Gateway: 192.168.100.1
  - DNS: Google DNS (8.8.8.8)

Test connectivity:
  From Dom0: ping 192.168.100.10
  From VM: ping 8.8.8.8

Let me know when networking works!
```

---

## Phase 6: DBXForensics Tools Installation

### AI Instructions

Install forensics tools on Windows VM:

### Windows Commands

```powershell
# Create tools directory
mkdir C:\Forensics
mkdir C:\Forensics\Tools
mkdir C:\Forensics\input
mkdir C:\Forensics\output

# Copy tools from shared folder (we'll set this up next)
# Or download directly from DBXForensics website
```

### AI Response Template

```
Installing DBXForensics tools on Windows VM...

Method 1: Copy from Dom0 (via shared folder - next step)
Method 2: Download directly on VM

For now, create directory structure:
  mkdir C:\Forensics
  mkdir C:\Forensics\Tools
  mkdir C:\Forensics\input
  mkdir C:\Forensics\output

Next, we'll set up the shared folder to transfer the tool installers.

Let me know when directories are created!
```

---

## Phase 7: Shared Folder Setup (SMB/CIFS)

### AI Instructions

Set up shared folders for file transfer between Dom0 and VM:

### Windows Commands (VM)

```powershell
# Enable SMB feature
Install-WindowsFeature -Name FS-FileServer

# Create shared folder
New-Item -Path "C:\Forensics\shared" -ItemType Directory
New-SmbShare -Name "ForensicsShare" -Path "C:\Forensics\shared" -FullAccess "Everyone"

# Set permissions
icacls "C:\Forensics\shared" /grant Everyone:F /T

# Show share info
Get-SmbShare -Name "ForensicsShare"
```

### Linux Commands (Dom0)

```bash
# Install CIFS utilities
sudo apt-get install -y cifs-utils

# Create mount point
sudo mkdir -p /mnt/forensics_vm/input
sudo mkdir -p /mnt/forensics_vm/output

# Mount shared folder
sudo mount -t cifs //192.168.100.10/ForensicsShare /mnt/forensics_vm/input -o username=Administrator,vers=3.0

# Test access
sudo ls -la /mnt/forensics_vm/input

# Make mount persistent (add to /etc/fstab)
echo "//192.168.100.10/ForensicsShare /mnt/forensics_vm/input cifs username=Administrator,password=YOUR_PASSWORD,vers=3.0 0 0" | sudo tee -a /etc/fstab
```

### AI Response Template

```
Setting up shared folders (Dom0 ↔ VM)...

On Windows VM:
  [Provide Windows commands]

On Dom0 (Linux):
  [Provide Linux commands]

This creates:
  Dom0: /mnt/forensics_vm/input → VM: C:\Forensics\shared

Now you can copy tool installers:
  sudo cp 04-integrations/forensics/tools/Setup_*.exe /mnt/forensics_vm/input/

Let me know when shared folders are working!
```

---

## Phase 8: Install Forensics Tools on VM

### AI Instructions

Install all 9 tools:

### Windows Commands (VM)

```powershell
# Navigate to shared folder
cd C:\Forensics\shared

# Install each tool (silent mode)
.\Setup_dbxScreenshot_1000.exe /S
.\Setup_dbxELA_1000.exe /S
.\Setup_dbxNoiseMap_1000.exe /S
.\Setup_dbxMetadata_1000.exe /S
.\Setup_dbxHashFile_1000.exe /S
.\Setup_dbxSeqCheck_1000.exe /S
.\Setup_dbxCsvViewer_1000.exe /S
.\Setup_dbxGhost_1000.exe /S
.\Setup_dbxMouseRecorder_1000.exe /S

# Wait for installations to complete
Start-Sleep -Seconds 30

# Verify installations
Get-ChildItem "C:\Program Files\DME Forensics" -Recurse -Filter "*.exe"

# Copy tools to C:\Forensics\Tools for easy access
Copy-Item "C:\Program Files\DME Forensics\*\*.exe" -Destination "C:\Forensics\Tools\" -Recurse
```

### AI Response Template

```
Installing all 9 DBXForensics tools...

Tools to install:
  1. dbxScreenshot - Forensic capture
  2. dbxELA - Error Level Analysis
  3. dbxNoiseMap - Device fingerprinting
  4. dbxMetadata - EXIF extraction
  5. dbxHashFile - Multi-algorithm hashing
  6. dbxSeqCheck - Sequence integrity
  7. dbxCsvViewer - CSV analysis
  8. dbxGhost - Visual comparison
  9. dbxMouseRecorder - Workflow automation

Commands to run on VM:
  [Provide commands above]

This will install all tools to:
  C:\Program Files\DME Forensics\

And copy to:
  C:\Forensics\Tools\

Let me know when installation completes!
```

---

## Phase 9: RPC Server Setup

### AI Instructions

Set up Flask RPC server on Windows VM:

### Windows Commands (VM)

```powershell
# Install Python 3.11 (download from python.org)
# Or use winget:
winget install Python.Python.3.11

# Install Flask
pip install flask requests

# Create RPC server script
# Copy forensics_rpc_server.py to VM via shared folder
copy C:\Forensics\shared\forensics_rpc_server.py C:\Forensics\

# Test RPC server
cd C:\Forensics
python forensics_rpc_server.py

# Server should start on port 5000
# Test from Dom0: curl http://192.168.100.10:5000/health
```

### Linux Commands (Dom0)

First, copy the RPC server script:

```bash
# Copy RPC server to shared folder
sudo cp /home/user/LAT5150DRVMIL/04-integrations/forensics/forensics_rpc_server.py /mnt/forensics_vm/input/

# Test RPC server
curl http://192.168.100.10:5000/health
# Expected: {"status": "healthy", "tools_loaded": 9, ...}
```

### AI Response Template

```
Setting up RPC server for forensics tools...

Step 1: Install Python on VM
  winget install Python.Python.3.11
  pip install flask requests

Step 2: Copy RPC server script
  From Dom0:
    sudo cp 04-integrations/forensics/forensics_rpc_server.py /mnt/forensics_vm/input/

  On VM:
    copy C:\Forensics\shared\forensics_rpc_server.py C:\Forensics\

Step 3: Start RPC server
  cd C:\Forensics
  python forensics_rpc_server.py

Step 4: Test from Dom0
  curl http://192.168.100.10:5000/health

Expected response:
  {"status": "healthy", "tools_loaded": 9, ...}

Let me know when RPC server is running!
```

---

## Phase 10: Make RPC Server Auto-Start

### AI Instructions

Configure RPC server to start automatically:

### Windows Commands (VM)

```powershell
# Create Windows service using NSSM (Non-Sucking Service Manager)
# Download from: https://nssm.cc/download

# Or use Task Scheduler:
$action = New-ScheduledTaskAction -Execute "python.exe" -Argument "C:\Forensics\forensics_rpc_server.py" -WorkingDirectory "C:\Forensics"
$trigger = New-ScheduledTaskTrigger -AtStartup
$principal = New-ScheduledTaskPrincipal -UserId "SYSTEM" -RunLevel Highest
Register-ScheduledTask -TaskName "ForensicsRPC" -Action $action -Trigger $trigger -Principal $principal

# Start the task
Start-ScheduledTask -TaskName "ForensicsRPC"

# Verify task is running
Get-ScheduledTask -TaskName "ForensicsRPC"
```

### AI Response Template

```
Making RPC server start automatically on boot...

Using Windows Task Scheduler to create auto-start task:
  [Provide commands above]

This creates a scheduled task that:
  - Runs at system startup
  - Executes: python forensics_rpc_server.py
  - Runs as SYSTEM (highest privileges)

Verify:
  Get-ScheduledTask -TaskName "ForensicsRPC"

Now the RPC server will start automatically when VM boots!

Let me know when auto-start is configured!
```

---

## Phase 11: Integration Testing

### AI Instructions

Test end-to-end integration:

### Linux Commands (Dom0)

```bash
# Test VM executor
cd /home/user/LAT5150DRVMIL/04-integrations/forensics

# Run VM executor test
python3 xen_vm_executor.py

# Expected output:
# === Xen VM Executor Test ===
#
# 1. Checking VM health...
#    ✓ Forensics VM is healthy
#
# 2. Listing available tools...
#    ✓ dbxScreenshot: C:\Forensics\Tools\dbxScreenshot.exe
#    ✓ dbxELA: C:\Forensics\Tools\dbxELA.exe
#    ...

# Test full forensics analysis
python3 << 'EOF'
from forensics_analyzer import ForensicsAnalyzer
from pathlib import Path

# Initialize with VM mode
analyzer = ForensicsAnalyzer(use_vm=True)

# Test with sample screenshot
test_image = Path("/path/to/test/screenshot.jpg")
report = analyzer.analyze_screenshot(test_image)

print(f"✓ Forensic analysis complete!")
print(f"  Verdict: {report.forensic_verdict}")
print(f"  Authenticity: {report.authenticity_score}/100")
print(f"  Manipulation detected: {report.manipulation_detected}")
EOF
```

### AI Response Template

```
Testing end-to-end forensics integration...

Test 1: VM Health Check
  Command: python3 xen_vm_executor.py
  Expected: ✓ Forensics VM is healthy

Test 2: Tool Availability
  Expected: All 9 tools found and accessible

Test 3: Forensic Analysis
  Command: [Provide Python test above]
  Expected: Complete forensic report with verdict

If all tests pass:
  ✓ Installation complete!
  ✓ VM is running
  ✓ RPC server is healthy
  ✓ All 9 tools are accessible
  ✓ Forensic analysis working

You can now use forensics tools via natural language!

Let me know test results!
```

---

## Phase 12: Natural Language Interface Test

### AI Instructions

Test NLI access to forensics tools:

### Test Queries

```
# Test forensic analysis via NLI
"Analyze this screenshot for authenticity: /path/to/screenshot.jpg"

# Test device fingerprinting
"What device captured this screenshot?"

# Test ELA analysis
"Is this image tampered?"

# Test full forensic workflow
"Run complete forensic analysis on all screenshots in /evidence/"
```

### AI Response Template

```
Testing natural language interface to forensics tools...

Try these queries:
  1. "Analyze this screenshot for authenticity: [path]"
     → Should run ELA + NoiseMap + Metadata + Hash

  2. "What device captured this screenshot?"
     → Should run NoiseMap device fingerprinting

  3. "Is this image tampered?"
     → Should run ELA manipulation detection

  4. "Run full forensic analysis on [directory]"
     → Should batch process all screenshots

If queries work:
  ✓ NLI integration successful!
  ✓ AI can invoke forensics tools
  ✓ Ready for production use

Let me know if NLI is working!
```

---

## Troubleshooting Guide for AI

### Common Issues and Solutions

#### Issue 1: VM Won't Start

**Symptoms:** `xl create` fails

**AI Response:**
```
VM creation failed. Let's troubleshoot:

1. Check Xen installation:
   xl list
   # Should show Dom0

2. Check disk space:
   df -h /var/lib/xen
   # Need 30GB free

3. Check VM config:
   cat /etc/xen/forensics-vm.cfg
   # Verify paths exist

4. Check logs:
   sudo xl dmesg | tail -20
   sudo journalctl -xe

Let me know the error message!
```

#### Issue 2: Network Not Working

**Symptoms:** Can't ping VM from Dom0

**AI Response:**
```
Network connectivity issue. Let's fix it:

1. Check VM IP:
   # On VM:
   Get-NetIPAddress
   # Should show 192.168.100.10

2. Check Dom0 bridge:
   # On Dom0:
   brctl show xenbr0
   # Should show VM interface

3. Check firewall:
   # On VM:
   Get-NetFirewallProfile
   # May need to disable for testing

4. Test connectivity:
   # From Dom0:
   ping 192.168.100.10

   # From VM:
   Test-Connection -ComputerName [Dom0 IP]

Let me know where connectivity fails!
```

#### Issue 3: Shared Folder Won't Mount

**Symptoms:** Can't access /mnt/forensics_vm/input

**AI Response:**
```
Shared folder mount failed. Let's troubleshoot:

1. Check SMB share on VM:
   Get-SmbShare -Name "ForensicsShare"
   # Should show share details

2. Check SMB permissions:
   Get-SmbShareAccess -Name "ForensicsShare"
   # Should show Everyone: Full

3. Install CIFS utilities (Dom0):
   sudo apt-get install cifs-utils

4. Try manual mount:
   sudo mount -t cifs //192.168.100.10/ForensicsShare /mnt/forensics_vm/input -o username=Administrator,password=YOUR_PASSWORD,vers=3.0

5. Check mount:
   mount | grep forensics_vm

Let me know the error message!
```

#### Issue 4: RPC Server Not Responding

**Symptoms:** curl to port 5000 fails

**AI Response:**
```
RPC server issue. Let's debug:

1. Check if server is running (VM):
   netstat -an | findstr 5000
   # Should show LISTENING on port 5000

2. Check Flask logs (VM):
   # Look at console output where you ran:
   python forensics_rpc_server.py

3. Check firewall (VM):
   Get-NetFirewallRule | Where-Object {$_.LocalPort -eq 5000}
   # May need to allow port 5000

4. Test from VM itself:
   Invoke-WebRequest -Uri http://localhost:5000/health

5. Test from Dom0:
   curl http://192.168.100.10:5000/health

Let me know where it fails!
```

#### Issue 5: Tools Not Found

**Symptoms:** "Tool not found" errors

**AI Response:**
```
Tools not accessible. Let's verify installation:

1. Check installation directory (VM):
   Get-ChildItem "C:\Program Files\DME Forensics" -Recurse -Filter "*.exe"

2. Check if tools were copied:
   Get-ChildItem "C:\Forensics\Tools" -Filter "*.exe"

3. Re-copy tools if needed:
   Copy-Item "C:\Program Files\DME Forensics\*\*.exe" -Destination "C:\Forensics\Tools\" -Recurse -Force

4. Verify RPC server config:
   # Check TOOLS dict in forensics_rpc_server.py
   # Paths should match actual tool locations

5. Update tool paths in RPC server if needed

Let me know which tools are missing!
```

---

## Summary - Complete Installation Checklist

AI can use this checklist to track progress:

```
DBXForensics Installation Checklist:

Phase 1: Prerequisites
  [ ] Xen hypervisor installed and running
  [ ] 30GB+ disk space available
  [ ] 4GB+ RAM available
  [ ] Network configured

Phase 2: Windows VM
  [ ] Windows Server Core ISO downloaded
  [ ] Xen VM created (forensics-vm)
  [ ] Windows installed
  [ ] Static IP configured (192.168.100.10)
  [ ] Network connectivity verified

Phase 3: Shared Folders
  [ ] SMB share created on VM
  [ ] CIFS mount working from Dom0
  [ ] File transfer tested

Phase 4: Forensics Tools
  [ ] All 9 tool installers copied to VM
  [ ] All tools installed successfully
  [ ] Tools accessible at C:\Forensics\Tools\

Phase 5: RPC Server
  [ ] Python 3.11+ installed on VM
  [ ] Flask installed
  [ ] RPC server script deployed
  [ ] RPC server running and accessible
  [ ] Auto-start configured

Phase 6: Integration
  [ ] VM executor tested (Dom0)
  [ ] Health check passing
  [ ] Tool listing working
  [ ] Forensic analysis functional

Phase 7: NLI
  [ ] Natural language queries working
  [ ] AI can invoke all 9 tools
  [ ] Batch processing functional

Status: [X/7 phases complete]
```

---

## AI Usage Examples

### Example 1: Complete Setup

**User:** "Help me set up the forensics tools with Xen VM"

**AI Response:**
```
I'll guide you through setting up DBXForensics with Xen VM! This will take about 30-60 minutes.

Let's start with a prerequisites check:

Run these commands:
  xl list
  df -h /
  free -h

Share the output and I'll verify everything is ready!
```

### Example 2: Troubleshooting

**User:** "The VM can't reach the network"

**AI Response:**
```
Let's troubleshoot VM networking:

1. First, check the VM's IP configuration:
   On VM: Get-NetIPAddress

2. Check if Dom0 can see the VM on the bridge:
   On Dom0: brctl show xenbr0

3. Try pinging the VM:
   On Dom0: ping 192.168.100.10

Share the results and I'll help fix the issue!
```

### Example 3: Testing

**User:** "How do I test if everything is working?"

**AI Response:**
```
Let's run a complete end-to-end test:

1. Test VM health:
   cd /home/user/LAT5150DRVMIL/04-integrations/forensics
   python3 xen_vm_executor.py

2. Test forensic analysis:
   python3 -c "from forensics_analyzer import ForensicsAnalyzer; a = ForensicsAnalyzer(use_vm=True); print(a.toolkit.vm_executor.check_health())"

3. Test natural language:
   Ask me: "Analyze this screenshot for authenticity: /path/to/test.jpg"

Share the results and I'll confirm everything is working!
```

---

## Advanced: Performance Optimization

Once basic installation works, AI can guide users through optimization:

```
Performance optimization options:

1. Install Xen PV Drivers (VM)
   - Improves disk and network I/O
   - Download from: https://xenbits.xen.org/

2. Enable CPU Pinning (Dom0)
   xl vcpu-pin forensics-vm 0 2
   xl vcpu-pin forensics-vm 1 3

3. Increase VM RAM
   xl mem-set forensics-vm 8192

4. Enable Parallel Processing (Dom0)
   from forensics_parallel import ParallelForensicsAnalyzer
   analyzer = ParallelForensicsAnalyzer(workers=8, use_vm=True)

Would you like me to guide you through any of these optimizations?
```

---

## Quick Reference Commands

AI can provide these quick references:

### Start/Stop VM

```bash
# Start VM
sudo xl create /etc/xen/forensics-vm.cfg

# Stop VM (graceful)
sudo xl shutdown forensics-vm

# Force stop
sudo xl destroy forensics-vm

# Restart VM
sudo xl shutdown forensics-vm && sleep 5 && sudo xl create /etc/xen/forensics-vm.cfg
```

### Check VM Status

```bash
# List all VMs
sudo xl list

# Show VM info
sudo xl list -l forensics-vm

# Show VM console
sudo xl console forensics-vm
```

### Test RPC Server

```bash
# Health check
curl http://192.168.100.10:5000/health

# List tools
curl http://192.168.100.10:5000/tools

# Test analysis (requires input file in shared folder)
curl -X POST http://192.168.100.10:5000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "tool": "dbxELA",
    "input_file": "C:\\Forensics\\input\\test.jpg",
    "args": []
  }'
```

### Mount/Unmount Shared Folder

```bash
# Mount
sudo mount -t cifs //192.168.100.10/ForensicsShare /mnt/forensics_vm/input -o username=Administrator,password=YOUR_PASSWORD,vers=3.0

# Unmount
sudo umount /mnt/forensics_vm/input

# Check mount
mount | grep forensics_vm
```

---

**LAT5150 DRVMIL - DBXForensics AI-Guided Installation Complete** 🔬

The AI can now guide users through the entire setup process via natural language!
