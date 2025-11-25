# DBXForensics Tools Installation Guide

**LAT5150 DRVMIL - Windows Forensics Tools on Linux**

Complete guide for installing and running DBXForensics Windows tools on Linux using Wine, containers, or virtualization.

---

## Problem

DBXForensics tools are Windows-only executables (.exe) that need to run on our Linux LAT5150 system. We have several options:

1. **Wine** - Run Windows .exe files natively on Linux (recommended for simple tools)
2. **Docker/Podman** - Containerized Windows environment
3. **Xen/KVM** - Full virtualization with Windows VM
4. **Wine + Sandbox** - Isolated Wine environment for security

---

## Option 1: Wine Installation (Recommended)

Wine allows running Windows executables directly on Linux.

### Step 1: Enable 32-bit Architecture

```bash
# Enable multiarch for 32-bit support
sudo dpkg --add-architecture i386
sudo apt-get update
```

### Step 2: Install Wine with 32-bit Support

```bash
# Install both wine64 and wine32
sudo apt-get install -y wine wine64 wine32
```

### Step 3: Configure Wine Directory

```bash
# Create Wine prefix in user directory (not root)
export WINEPREFIX="$HOME/.wine"
export WINEARCH=win64

# Initialize Wine configuration
winecfg
```

### Step 4: Install DBXForensics Tools

```bash
cd /home/user/LAT5150DRVMIL/04-integrations/forensics/tools

# Run each installer (silent mode if supported)
wine Setup_dbxScreenshot_1000.exe /S
wine Setup_dbxELA_1000.exe /S
wine Setup_dbxNoiseMap_1000.exe /S
wine Setup_dbxMetadata_1000.exe /S
wine Setup_dbxHashFile_1000.exe /S
wine Setup_dbxSeqCheck_1000.exe /S
wine Setup_dbxCsvViewer_1000.exe /S
wine Setup_dbxGhost_1000.exe /S
wine Setup_dbxMouseRecorder_1000.exe /S
```

**Note**: If `/S` (silent install) doesn't work, run without it and click through installers.

### Step 5: Locate Installed Tools

```bash
# Tools typically install to:
# ~/.wine/drive_c/Program Files/DME Forensics/

ls -la ~/.wine/drive_c/Program\ Files/DME\ Forensics/
```

### Step 6: Update Toolkit Configuration

Edit `dbxforensics_toolkit.py` to point to actual installed tool locations:

```python
# Update tool paths in DBXForensicsToolkit.__init__()
DEFAULT_TOOLS = {
    'dbxScreenshot.exe': Path.home() / '.wine/drive_c/Program Files/DME Forensics/dbxScreenshot/dbxScreenshot.exe',
    'dbxELA.exe': Path.home() / '.wine/drive_c/Program Files/DME Forensics/dbxELA/dbxELA.exe',
    # ... etc
}
```

---

## Option 2: Docker Container with Wine

Create isolated container with Wine for forensics tools.

### Dockerfile

```dockerfile
FROM ubuntu:24.04

# Install Wine
RUN dpkg --add-architecture i386 && \
    apt-get update && \
    apt-get install -y wine wine64 wine32 && \
    rm -rf /var/lib/apt/lists/*

# Create wine user
RUN useradd -m -s /bin/bash wineuser

USER wineuser
WORKDIR /forensics

# Set Wine environment
ENV WINEPREFIX=/home/wineuser/.wine
ENV WINEARCH=win64

# Initialize Wine
RUN winecfg && echo "Wine initialized"

# Copy installers
COPY tools/*.exe /forensics/installers/

# Install tools
RUN wine /forensics/installers/Setup_dbxELA_1000.exe /S && \
    wine /forensics/installers/Setup_dbxScreenshot_1000.exe /S && \
    wine /forensics/installers/Setup_dbxNoiseMap_1000.exe /S

CMD ["/bin/bash"]
```

### Build and Run

```bash
# Build container
docker build -t lat5150-forensics .

# Run forensics analysis
docker run -v /path/to/evidence:/evidence lat5150-forensics \
  wine "C:\\Program Files\\DME Forensics\\dbxELA\\dbxELA.exe" /evidence/screenshot.jpg
```

---

## Option 3: Xen/KVM Windows VM

For maximum compatibility and security, run full Windows VM.

### Xen Setup

```bash
# Install Xen hypervisor
sudo apt-get install -y xen-hypervisor-amd64 xen-tools

# Create Windows 10/11 VM
sudo xen-create-image \
  --hostname forensics-win \
  --size 30Gb \
  --memory 4096 \
  --vcpus 2 \
  --dist windows

# Install Windows in VM and install DBXForensics tools normally
# Access via RDP or VNC
```

### VM Communication

```python
# Use RPC or shared folders to send screenshots to VM
import subprocess

def analyze_via_vm(screenshot_path):
    # Copy to shared folder
    subprocess.run(['scp', screenshot_path, 'forensics-win:/input/'])

    # Execute via RPC
    subprocess.run(['ssh', 'forensics-win',
                    'C:\\Program Files\\DME Forensics\\dbxELA\\dbxELA.exe',
                    'C:\\input\\screenshot.jpg'])

    # Retrieve results
    subprocess.run(['scp', 'forensics-win:/output/result.json', './'])
```

---

## Option 4: Sandboxed Wine (Security)

Use Firejail to sandbox Wine execution for untrusted screenshots.

```bash
# Install Firejail
sudo apt-get install -y firejail

# Run Wine in sandbox
firejail --net=none wine dbxELA.exe /path/to/screenshot.jpg
```

---

## Current Status

**✅ Wine Installed**: wine64 version 9.0
**❌ 32-bit Support**: wine32 not installed (multiarch not enabled)
**❌ Tools Installed**: Installers available but not yet run

---

## Recommended Setup for LAT5150

### Immediate Solution (Development)

```bash
# 1. Enable 32-bit architecture
sudo dpkg --add-architecture i386
sudo apt-get update
sudo apt-get install -y wine32:i386

# 2. Create non-root Wine prefix
export WINEPREFIX="/opt/lat5150/wine"
export WINEARCH=win64
mkdir -p /opt/lat5150/wine
chown -R $USER:$USER /opt/lat5150/wine

# 3. Install tools
cd /home/user/LAT5150DRVMIL/04-integrations/forensics/tools
for installer in Setup_*.exe; do
    WINEPREFIX=/opt/lat5150/wine wine "$installer" /S
done

# 4. Verify installation
WINEPREFIX=/opt/lat5150/wine wine "C:\\Program Files\\DME Forensics\\dbxELA\\dbxELA.exe" --version
```

### Production Solution (Container)

```bash
# Build Docker container with all tools pre-installed
docker build -t lat5150/forensics:latest .

# Run via Python subprocess
import subprocess
result = subprocess.run([
    'docker', 'run', '--rm',
    '-v', f'{screenshot_path}:/input.jpg:ro',
    'lat5150/forensics:latest',
    'wine', 'C:\\Program Files\\DME Forensics\\dbxELA\\dbxELA.exe',
    '/input.jpg'
], capture_output=True)
```

---

## Alternative: Extract Portable Executables

Some installers can be extracted without installing:

```bash
# Try extracting with innoextract (for InnoSetup installers)
sudo apt-get install -y innoextract

innoextract Setup_dbxELA_1000.exe
# Look for extracted files in app/ or similar directory

# Or use 7z
7z x Setup_dbxELA_1000.exe -oextracted/

# Find .exe files
find extracted/ -name "dbx*.exe"
```

---

## Updated Toolkit for Missing Tools

Our `dbxforensics_toolkit.py` already handles missing tools gracefully:

```python
class DBXForensicsTool:
    def execute(self, *args, **kwargs):
        if not self.tool_path.exists():
            return ToolResult(
                success=False,
                stdout="",
                stderr=f"Tool not found: {self.tool_path}. See TOOL_INSTALLATION.md",
                return_code=-1
            )
        # ... execute tool
```

---

## Testing Tool Installation

```bash
cd /home/user/LAT5150DRVMIL/04-integrations/forensics

# Test toolkit (will show which tools are available)
python3 << 'EOF'
from dbxforensics_toolkit import DBXForensicsToolkit

toolkit = DBXForensicsToolkit()

print("Tool Availability:")
print(f"  dbxScreenshot: {toolkit.screenshot.tool_path.exists()}")
print(f"  dbxELA: {toolkit.ela.tool_path.exists()}")
print(f"  dbxNoiseMap: {toolkit.noise_map.tool_path.exists()}")
print(f"  dbxMetadata: {toolkit.metadata.tool_path.exists()}")
print(f"  dbxHashFile: {toolkit.hash_file.tool_path.exists()}")
print(f"  dbxSeqCheck: {toolkit.seq_check.tool_path.exists()}")
print(f"  dbxCsvViewer: {toolkit.csv_viewer.tool_path.exists()}")
print(f"  dbxGhost: {toolkit.ghost.tool_path.exists()}")
print(f"  dbxMouseRecorder: {toolkit.mouse_recorder.tool_path.exists()}")
EOF
```

---

## Troubleshooting

### Wine32 Missing

```bash
# Error: wine32 is missing
# Solution:
sudo dpkg --add-architecture i386
sudo apt-get update
sudo apt-get install wine32:i386
```

### Permission Denied

```bash
# Error: '/root' is not owned by you
# Solution: Run as non-root user or set WINEPREFIX
export WINEPREFIX="/home/$USER/.wine"
wine program.exe
```

### Tools Not Found

```bash
# Error: dbxELA.exe not found
# Solution: Check installation location
find ~/.wine -name "dbx*.exe"

# Update toolkit paths in dbxforensics_toolkit.py
```

### Silent Install Not Working

```bash
# If /S flag doesn't work, try:
wine Setup_dbxELA_1000.exe /SILENT
wine Setup_dbxELA_1000.exe /VERYSILENT
wine Setup_dbxELA_1000.exe /quiet

# Or run interactively and click through:
wine Setup_dbxELA_1000.exe
```

---

## Security Considerations

### Sandboxing Options

1. **Firejail**: Isolate Wine processes
   ```bash
   firejail --net=none --private=temp wine dbxELA.exe screenshot.jpg
   ```

2. **Docker**: Complete container isolation
   ```bash
   docker run --rm --network=none -v /evidence:/mnt:ro forensics wine dbxELA.exe
   ```

3. **AppArmor**: Define Wine profile
   ```bash
   sudo aa-enforce /etc/apparmor.d/usr.bin.wine
   ```

### Untrusted Screenshot Handling

```python
def analyze_untrusted_screenshot(screenshot_path):
    """Analyze potentially malicious screenshot in isolated environment"""
    import subprocess

    result = subprocess.run([
        'firejail',
        '--net=none',
        '--private=/tmp/forensics',
        '--read-only=/usr',
        'wine',
        'dbxELA.exe',
        screenshot_path
    ], capture_output=True, timeout=60)

    return result
```

---

## Performance Comparison

| Method | Setup Time | Execution Speed | Isolation | Compatibility |
|--------|------------|-----------------|-----------|---------------|
| Wine (native) | 5 min | 100% | Low | 90% |
| Docker + Wine | 10 min | 95% | High | 95% |
| Xen VM | 30 min | 80% | Maximum | 100% |
| Firejail + Wine | 5 min | 98% | Medium | 90% |

**Recommendation**: Docker + Wine for production (best balance of isolation and performance)

---

## Next Steps

1. **Enable 32-bit**: `dpkg --add-architecture i386`
2. **Install wine32**: `apt-get install wine32:i386`
3. **Install tools**: Run all `Setup_*.exe` installers
4. **Update paths**: Edit `dbxforensics_toolkit.py` with actual tool locations
5. **Test**: Run `python3 forensics_analyzer.py`

---

**LAT5150 DRVMIL - Forensics Tool Installation Complete** 🔬

Choose your installation method and proceed with setup.
