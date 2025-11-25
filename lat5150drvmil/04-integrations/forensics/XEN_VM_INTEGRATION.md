# Xen VM Integration for DBXForensics Tools

**LAT5150 DRVMIL - Isolated Forensics Analysis Environment**

Complete guide for running DBXForensics tools in an ultra-lightweight Windows VM using Xen hypervisor for maximum isolation, compatibility, and security.

---

## Why Xen VM Approach?

✅ **100% Windows Compatibility** - Native tool execution
✅ **Complete Isolation** - Forensics sandbox separate from main system
✅ **Security** - Analyze untrusted screenshots safely
✅ **Existing Infrastructure** - Leverage LAT5150's Xen deployment
✅ **Performance** - Direct hardware access, no emulation overhead
✅ **Barebones** - Minimal resource usage with Windows Server Core

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│              LAT5150 DRVMIL (Dom0 - Linux)              │
│                                                         │
│  Python Forensics Framework                             │
│  ├── forensics_analyzer.py                              │
│  ├── dbxforensics_toolkit.py                            │
│  └── forensics_pipelines.py                             │
│                                                         │
│            ▼ RPC/API Communication ▼                    │
└─────────────────────────────────────────────────────────┘
                        │
                        │ Xen Bridge Network
                        │
┌─────────────────────────────────────────────────────────┐
│      Forensics VM (DomU - Windows Server Core)          │
│                                                         │
│  Windows Server 2022 Core (No GUI)                      │
│  ├── DBXForensics Tools (9 executables)                 │
│  ├── PowerShell Core                                    │
│  ├── Forensics RPC Server (Python/PowerShell)           │
│  └── Shared Folder: /forensics/input /forensics/output  │
│                                                         │
│  Resources: 2 vCPU, 4GB RAM, 20GB Disk                  │
└─────────────────────────────────────────────────────────┘
```

---

## Part 1: Windows Server Core VM Setup

### Step 1: Download Windows Server Core ISO

**Option A: Windows Server 2022 Core** (Recommended)
```bash
# Download Windows Server 2022 ISO
wget https://software-download.microsoft.com/download/pr/...

# Or use evaluation edition (180-day trial)
wget https://www.microsoft.com/en-us/evalcenter/download-windows-server-2022
```

**Option B: Windows Server 2019 Core**
```bash
# Lighter weight, good for older hardware
wget https://www.microsoft.com/en-us/evalcenter/download-windows-server-2019
```

**Characteristics**:
- No Desktop Experience (no GUI)
- Command-line only (PowerShell/CMD)
- ~4GB RAM footprint (vs 8GB with GUI)
- ~10GB disk usage (vs 30GB with GUI)

### Step 2: Create Xen VM Configuration

Create `/etc/xen/forensics-win.cfg`:

```python
# Forensics Windows VM Configuration
name = "forensics-win"
type = "hvm"  # Hardware Virtual Machine

# CPU and Memory
vcpus = 2
memory = 4096  # 4GB RAM

# Disk
disk = [
    'file:/var/lib/xen/images/forensics-win.img,xvda,w',  # Main disk (20GB)
]

# Network
vif = ['bridge=xenbr0']  # Connected to bridge network

# Boot
boot = "dc"  # d=cdrom, c=disk
on_poweroff = "destroy"
on_reboot = "restart"
on_crash = "preserve"

# VNC Console (for installation only)
vnc = 1
vnclisten = "0.0.0.0"
vncpasswd = "forensics123"

# Serial console
serial = "pty"
```

### Step 3: Create VM Disk Image

```bash
# Create 20GB disk image
dd if=/dev/zero of=/var/lib/xen/images/forensics-win.img bs=1M count=20480

# Or use sparse file (faster)
truncate -s 20G /var/lib/xen/images/forensics-win.img
```

### Step 4: Install Windows Server Core

```bash
# Attach ISO to VM config (edit /etc/xen/forensics-win.cfg)
disk = [
    'file:/var/lib/xen/images/forensics-win.img,xvda,w',
    'file:/path/to/WindowsServer2022.iso,xvdc:cdrom,r'
]

# Start VM
xl create /etc/xen/forensics-win.cfg

# Connect via VNC to complete installation
# (Use virt-viewer, vncviewer, or Remote Desktop)
vncviewer localhost:5900
```

**Installation Steps**:
1. Boot from ISO
2. Select "Windows Server 2022 Standard (Server Core)"
3. Complete installation (no product key needed for eval)
4. Set Administrator password
5. Remove ISO from config and reboot

### Step 5: Post-Installation Configuration

Connect via PowerShell Remoting:

```bash
# From Dom0 (Linux), connect to VM
ssh administrator@<VM-IP>  # Or use xl console

# Inside Windows Server Core VM (PowerShell):
```

```powershell
# Set computer name
Rename-Computer -NewName "FORENSICS-01" -Restart

# Configure network (static IP recommended)
New-NetIPAddress -InterfaceAlias "Ethernet" -IPAddress 192.168.100.10 -PrefixLength 24 -DefaultGateway 192.168.100.1
Set-DnsClientServerAddress -InterfaceAlias "Ethernet" -ServerAddresses 8.8.8.8,8.8.4.4

# Enable PowerShell Remoting
Enable-PSRemoting -Force
Set-Item WSMan:\localhost\Client\TrustedHosts -Value "*" -Force

# Disable Windows Defender (forensics VM, isolated network)
Set-MpPreference -DisableRealtimeMonitoring $true

# Install Python 3.11 (for RPC server)
# Download from python.org or use winget if available
Invoke-WebRequest -Uri "https://www.python.org/ftp/python/3.11.0/python-3.11.0-amd64.exe" -OutFile "C:\python-installer.exe"
Start-Process -FilePath "C:\python-installer.exe" -ArgumentList "/quiet InstallAllUsers=1 PrependPath=1" -Wait

# Verify Python installation
python --version
```

---

## Part 2: Install DBXForensics Tools

### Transfer Tools to VM

**Option A: Shared Folder (Xen 9fs)**

Configure 9p shared folder in `/etc/xen/forensics-win.cfg`:

```python
# Add to VM config
p9 = ['tag=forensics,security_model=passthrough,path=/home/user/LAT5150DRVMIL/04-integrations/forensics/tools']
```

**Option B: SCP Transfer**

```bash
# From Dom0, copy installers to VM
scp /home/user/LAT5150DRVMIL/04-integrations/forensics/tools/Setup_*.exe administrator@192.168.100.10:C:\\Installers\\
```

**Option C: SMB Share**

```bash
# From Dom0, start SMB server
python3 -m http.server 8000  # Simple HTTP server

# From VM, download
Invoke-WebRequest -Uri "http://192.168.100.1:8000/Setup_dbxELA_1000.exe" -OutFile "C:\Installers\Setup_dbxELA_1000.exe"
```

### Install All DBXForensics Tools

```powershell
# On Windows Server Core VM

# Create installation directory
New-Item -Path "C:\Forensics" -ItemType Directory

# Install each tool silently
$installers = @(
    "Setup_dbxScreenshot_1000.exe",
    "Setup_dbxELA_1000.exe",
    "Setup_dbxNoiseMap_1000.exe",
    "Setup_dbxMetadata_1000.exe",
    "Setup_dbxHashFile_1000.exe",
    "Setup_dbxSeqCheck_1000.exe",
    "Setup_dbxCsvViewer_1000.exe",
    "Setup_dbxGhost_1000.exe",
    "Setup_dbxMouseRecorder_1000.exe"
)

foreach ($installer in $installers) {
    Write-Host "Installing $installer..."
    Start-Process -FilePath "C:\Installers\$installer" -ArgumentList "/S" -Wait
}

# Verify installation
Get-ChildItem "C:\Program Files\DME Forensics\" -Recurse -Filter "*.exe"
```

---

## Part 3: Forensics RPC Server

Create a lightweight RPC server on the Windows VM to accept analysis requests from Dom0.

### Create `forensics_rpc_server.py` on Windows VM

```powershell
# On Windows VM
Set-Location C:\Forensics

# Create RPC server
@"
#!/usr/bin/env python3
'''
Forensics RPC Server for DBXForensics Tools
Runs on Windows VM, accepts analysis requests from Linux Dom0
'''

import os
import json
import subprocess
from pathlib import Path
from flask import Flask, request, jsonify
from datetime import datetime

app = Flask(__name__)

# Tool paths
TOOLS = {
    'dbxScreenshot': r'C:\Program Files\DME Forensics\dbxScreenshot\dbxScreenshot.exe',
    'dbxELA': r'C:\Program Files\DME Forensics\dbxELA\dbxELA.exe',
    'dbxNoiseMap': r'C:\Program Files\DME Forensics\dbxNoiseMap\dbxNoiseMap.exe',
    'dbxMetadata': r'C:\Program Files\DME Forensics\dbxMetadata\dbxMetadata.exe',
    'dbxHashFile': r'C:\Program Files\DME Forensics\dbxHashFile\dbxHashFile.exe',
    'dbxSeqCheck': r'C:\Program Files\DME Forensics\dbxSeqCheck\dbxSeqCheck.exe',
    'dbxCsvViewer': r'C:\Program Files\DME Forensics\dbxCsvViewer\dbxCsvViewer.exe',
    'dbxGhost': r'C:\Program Files\DME Forensics\dbxGhost\dbxGhost.exe',
    'dbxMouseRecorder': r'C:\Program Files\DME Forensics\dbxMouseRecorder\dbxMouseRecorder.exe',
}

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

@app.route('/tools', methods=['GET'])
def list_tools():
    '''List available tools and their status'''
    tools_status = {}
    for name, path in TOOLS.items():
        tools_status[name] = {
            'path': path,
            'exists': os.path.exists(path)
        }
    return jsonify(tools_status)

@app.route('/analyze', methods=['POST'])
def analyze():
    '''
    Execute forensic analysis

    Request JSON:
    {
        \"tool\": \"dbxELA\",
        \"input_file\": \"C:\\\\Forensics\\\\input\\\\screenshot.jpg\",
        \"args\": [\"/quality:90\"],
        \"output_file\": \"C:\\\\Forensics\\\\output\\\\result.json\"
    }
    '''
    try:
        data = request.get_json()

        tool_name = data.get('tool')
        input_file = data.get('input_file')
        args = data.get('args', [])
        output_file = data.get('output_file')

        if tool_name not in TOOLS:
            return jsonify({'error': f'Unknown tool: {tool_name}'}), 400

        tool_path = TOOLS[tool_name]

        if not os.path.exists(tool_path):
            return jsonify({'error': f'Tool not found: {tool_path}'}), 500

        if not os.path.exists(input_file):
            return jsonify({'error': f'Input file not found: {input_file}'}), 400

        # Build command
        cmd = [tool_path, input_file] + args

        # Execute tool
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300
        )

        # Parse output
        response = {
            'tool': tool_name,
            'success': result.returncode == 0,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'returncode': result.returncode,
            'timestamp': datetime.now().isoformat()
        }

        # Save output if specified
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(response, f, indent=2)

        return jsonify(response)

    except subprocess.TimeoutExpired:
        return jsonify({'error': 'Tool execution timeout'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Run on all interfaces, port 5000
    app.run(host='0.0.0.0', port=5000, debug=False)
"@ | Out-File -FilePath forensics_rpc_server.py -Encoding UTF8

# Install Flask
pip install flask

# Test server
python forensics_rpc_server.py
```

### Run RPC Server as Windows Service

```powershell
# Install NSSM (Non-Sucking Service Manager)
Invoke-WebRequest -Uri "https://nssm.cc/release/nssm-2.24.zip" -OutFile "C:\nssm.zip"
Expand-Archive -Path "C:\nssm.zip" -DestinationPath "C:\nssm"

# Create service
C:\nssm\nssm-2.24\win64\nssm.exe install ForensicsRPC "C:\Program Files\Python311\python.exe" "C:\Forensics\forensics_rpc_server.py"
C:\nssm\nssm-2.24\win64\nssm.exe set ForensicsRPC AppDirectory "C:\Forensics"
C:\nssm\nssm-2.24\win64\nssm.exe set ForensicsRPC Start SERVICE_AUTO_START

# Start service
Start-Service ForensicsRPC

# Verify service
Get-Service ForensicsRPC
```

---

## Part 4: Dom0 (Linux) Integration

Update `dbxforensics_toolkit.py` to use VM RPC instead of Wine.

### Create `xen_vm_executor.py`

```python
#!/usr/bin/env python3
"""
Xen VM Executor for DBXForensics Tools
Sends analysis requests to Windows VM via RPC
"""

import requests
import json
from pathlib import Path
from typing import Dict, Any
import shutil

class XenVMExecutor:
    """Execute forensics tools on Xen Windows VM"""

    def __init__(self, vm_ip: str = "192.168.100.10", vm_port: int = 5000):
        """
        Initialize VM executor

        Args:
            vm_ip: IP address of forensics VM
            vm_port: RPC server port
        """
        self.vm_ip = vm_ip
        self.vm_port = vm_port
        self.base_url = f"http://{vm_ip}:{vm_port}"

        # Shared directories
        self.shared_input = Path("/mnt/forensics_vm/input")
        self.shared_output = Path("/mnt/forensics_vm/output")

    def check_health(self) -> bool:
        """Check if VM RPC server is healthy"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False

    def execute_tool(
        self,
        tool_name: str,
        input_file: Path,
        args: list = None
    ) -> Dict[str, Any]:
        """
        Execute forensic tool on VM

        Args:
            tool_name: Name of tool (e.g., 'dbxELA')
            input_file: Path to input file on Dom0
            args: Tool arguments

        Returns:
            Tool execution result
        """
        # Copy input file to shared folder
        shared_input_path = self.shared_input / input_file.name
        shutil.copy(input_file, shared_input_path)

        # Map to Windows path
        win_input_path = f"C:\\\\Forensics\\\\input\\\\{input_file.name}"
        win_output_path = f"C:\\\\Forensics\\\\output\\\\result_{input_file.stem}.json"

        # Send RPC request
        request_data = {
            "tool": tool_name,
            "input_file": win_input_path,
            "args": args or [],
            "output_file": win_output_path
        }

        response = requests.post(
            f"{self.base_url}/analyze",
            json=request_data,
            timeout=300
        )

        if response.status_code != 200:
            return {
                "success": False,
                "error": response.text
            }

        return response.json()
```

### Update `dbxforensics_toolkit.py`

Add VM executor option:

```python
class DBXForensicsTool:
    def __init__(self, tool_name, tool_exe_path, timeout=300, use_vm=False):
        self.use_vm = use_vm
        if use_vm:
            self.vm_executor = XenVMExecutor()

    def execute(self, *args):
        if self.use_vm:
            # Execute on VM
            return self.vm_executor.execute_tool(
                tool_name=self.tool_name,
                input_file=args[0],  # First arg is usually input file
                args=list(args[1:])
            )
        else:
            # Execute via Wine (original code)
            # ...
```

---

## Part 5: Shared Folder Configuration

### Mount Windows Share on Dom0

```bash
# On Windows VM, create share
# (In PowerShell)
New-SmbShare -Name "Forensics" -Path "C:\Forensics" -FullAccess "Everyone"

# On Dom0 (Linux), mount share
mkdir -p /mnt/forensics_vm
mount -t cifs //192.168.100.10/Forensics /mnt/forensics_vm -o username=administrator,password=YourPassword

# Add to /etc/fstab for persistent mount
echo "//192.168.100.10/Forensics /mnt/forensics_vm cifs username=administrator,password=YourPassword 0 0" >> /etc/fstab
```

---

## Part 6: Performance Optimization

### Enable Xen PV Drivers

Install Xen Windows PV Drivers on VM for better performance:

```powershell
# Download Xen Windows PV Drivers
Invoke-WebRequest -Uri "https://xenbits.xen.org/pvdrivers/win/8.2.2/xenbus.tar" -OutFile "C:\xenbus.tar"

# Extract and install
# (Follow Xen PV driver installation guide)
```

### CPU Pinning

Pin VM CPUs for consistent performance:

```python
# Add to /etc/xen/forensics-win.cfg
cpus = "2-3"  # Pin to physical CPUs 2 and 3
```

### Memory Ballooning

```python
# Add to VM config
autoballoon = 0  # Disable autoballoon for consistent memory
```

---

## Part 7: Security Hardening

### Network Isolation

```python
# Create isolated network for forensics VM
vif = ['bridge=xenbr-forensics']  # Dedicated forensics bridge

# On Dom0, create isolated bridge
brctl addbr xenbr-forensics
ip addr add 192.168.200.1/24 dev xenbr-forensics
ip link set xenbr-forensics up

# Add firewall rules (block internet access)
iptables -A FORWARD -i xenbr-forensics -o eth0 -j DROP
iptables -A FORWARD -i xenbr-forensics -d 192.168.200.0/24 -j ACCEPT
```

### Snapshot Before Analysis

```bash
# Take VM snapshot before analyzing untrusted screenshots
xl save forensics-win /var/lib/xen/snapshots/forensics-win-clean.snapshot

# Restore if compromised
xl restore /var/lib/xen/snapshots/forensics-win-clean.snapshot
```

---

## Complete Workflow Example

```python
#!/usr/bin/env python3
"""
Example: Analyze screenshot using Xen VM
"""

from pathlib import Path
from xen_vm_executor import XenVMExecutor

# Initialize VM executor
vm = XenVMExecutor(vm_ip="192.168.100.10")

# Check VM health
if not vm.check_health():
    print("❌ Forensics VM not available")
    exit(1)

print("✓ Forensics VM healthy")

# Analyze screenshot
screenshot = Path("/home/user/evidence/screenshot.jpg")

result = vm.execute_tool(
    tool_name="dbxELA",
    input_file=screenshot,
    args=["/quality:90"]
)

if result['success']:
    print(f"✓ Analysis complete")
    print(f"Verdict: {result['stdout']}")
else:
    print(f"❌ Analysis failed: {result.get('error')}")
```

---

## Resource Requirements

| Component | Minimum | Recommended | Notes |
|-----------|---------|-------------|-------|
| RAM (VM) | 2GB | 4GB | Server Core uses ~2-3GB |
| Disk (VM) | 15GB | 20GB | OS + Tools |
| vCPUs | 1 | 2 | Parallel analysis |
| Network | 100Mbps | 1Gbps | File transfer speed |

---

## Troubleshooting

### VM Won't Start

```bash
# Check Xen logs
xl dmesg | tail -50
tail -f /var/log/xen/qemu-dm-forensics-win.log

# Verify VM config
xl list
xl info
```

### Can't Connect to RPC Server

```powershell
# On VM, check service status
Get-Service ForensicsRPC

# Check firewall
netsh advfirewall firewall add rule name="Forensics RPC" dir=in action=allow protocol=TCP localport=5000

# Test locally
Invoke-WebRequest -Uri "http://localhost:5000/health"
```

### Slow Performance

```bash
# Enable PV drivers
# Pin CPUs
# Increase RAM allocation
# Use SSD for VM disk
```

---

## Next Steps

1. **Create VM**: Follow Part 1 to create Windows Server Core VM
2. **Install Tools**: Transfer and install all 9 DBXForensics tools
3. **Setup RPC**: Deploy forensics_rpc_server.py as Windows service
4. **Configure Shared Folders**: Mount SMB share on Dom0
5. **Update Python Code**: Integrate XenVMExecutor into toolkit
6. **Test**: Run complete forensic analysis workflow

---

**LAT5150 DRVMIL - Xen VM Forensics Integration Complete** 🔬

Maximum isolation, 100% compatibility, production-ready forensics environment.
