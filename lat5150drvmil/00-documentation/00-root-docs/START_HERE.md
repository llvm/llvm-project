# 🚀 LAT5150DRVMIL - DSMIL AI Platform

## Quick Launch

**Launch the complete DSMIL Control Center with one command:**

```bash
sudo ./launch-dsmil-control-center.sh
```

This opens a **4-panel monitoring and control interface** for all 84 DSMIL devices.

---

## What is This?

The **Dell System Military Integration Layer (DSMIL)** AI Platform provides:

- 🔧 **84 DSMIL Devices** (0x8000-0x806B) - Military-grade hardware subsystems
- ⚡ **656 Operations** - Complete device operation catalog
- 📊 **Real-time Monitoring** - Multi-panel control center
- 🛡️ **Safety Enforcement** - Quarantine protection for dangerous devices

---

## System Components

### Hardware Platform
- **Dell Latitude 5450 MIL-SPEC** (JRTC1 Training Variant)
- **Intel Arc Graphics** (Xe-LPG) - 40 TOPS @ MILITARY mode
- **Intel NPU 3720** - 26.4 TOPS @ MILITARY mode
- **Intel NCS2** - 2x devices (20 TOPS), 3rd in mail
- **Physical RAM**: 64GB (2GB reserved by DSMIL firmware)
- **Total System TOPS**: 86.4 (current) → 96.4 (when 3rd NCS2 arrives)

### DSMIL Device Groups

| Group | Range | Devices | Function |
|-------|-------|---------|----------|
| **0** | 0x8000-0x800B | 12 | Core Security & Emergency |
| **1** | 0x800C-0x8017 | 12 | Extended Security |
| **2** | 0x8018-0x8023 | 12 | Network/Communications |
| **3** | 0x8024-0x802F | 12 | Data Processing |
| **4** | 0x8030-0x803B | 12 | Storage Management |
| **5** | 0x803C-0x8047 | 12 | Peripheral Control |
| **6** | 0x8048-0x8053 | 12 | Training/Simulation |

### AI Integration
- **WhiteRabbitNeo-13B** - Multi-device inference engine
- **Dynamic hardware allocation** - NPU/Arc/NCS2 optimization
- **Intelligent routing** - Task-to-accelerator matching

---

## Control Center Layout

When you run the launcher, you get 4 panels:

```
┌──────────────────────┬──────────────────────┐
│  CONTROL CONSOLE     │   SYSTEM LOGS        │
│  • Hardware status   │   • Live log view    │
│  • Driver status     │   • Color-coded      │
│  • Live metrics      │   • Event tracking   │
├──────────────────────┼──────────────────────┤
│  DEVICE ACTIVATION   │  OPERATION MONITOR   │
│  • 84 devices        │  • 656 operations    │
│  • Group navigation  │  • Execution engine  │
│  • Activation ctrl   │  • History tracking  │
└──────────────────────┴──────────────────────┘
```

---

## First-Time Setup

### 1. Install Dependencies

```bash
sudo apt update
sudo apt install tmux python3 -y
```

### 2. Optional: Load DSMIL Driver (Hardware Only)

```bash
sudo modprobe dsmil-84dev    # legacy alias: dsmil-72dev
lsmod | grep dsmil  # Verify
```

### 3. Launch Control Center

```bash
sudo ./launch-dsmil-control-center.sh
```

---

## Navigation

### tmux Controls
- `Ctrl+B` then `Arrow Keys` - Switch between panels
- `Ctrl+B` then `D` - Detach (keeps running)
- `Ctrl+B` then `Z` - Zoom current panel
- `Ctrl+B` then `?` - Help

### Device Activation Panel
- `←/→` - Switch groups
- `↑/↓` - Navigate devices
- `ENTER` - Activate device
- `I` - Device info
- `Q` - Quit

### Operation Monitor Panel
- `↑/↓` - Navigate
- `ENTER` - View details
- `E` - Execute operation
- `L` - Execution log
- `Q` - Quit

---

## Key Files

### Launchers
- **launch-dsmil-control-center.sh** - Main integrated launcher
- **QUICK_START.md** - Detailed usage guide (READ THIS!)

### Core Tools
- **02-ai-engine/dsmil_guided_activation.py** - Device activation TUI
- **02-ai-engine/dsmil_operation_monitor.py** - Operation browser/executor
- **02-ai-engine/hardware_profile.py** - Hardware specifications

### Documentation
- **00-documentation/00-root-docs/DSMIL_CURRENT_REFERENCE.md** - Complete device reference
- **DSMIL_DEVICE_CAPABILITIES.json** - 656 operation catalog (222 KB)
- **DSMIL_ENUMERATION_GUIDE.md** - Enumeration procedures

---

## Safety Notes

### ✅ Safe to Use
- All read operations
- Device status queries
- Hardware monitoring
- Logging and analysis

### ⚠️ Use with Caution
- Device activation (check dependencies first)
- Write operations (understand function first)
- Configuration changes

### 🔴 NEVER TOUCH (Quarantined)
- **0x8009** - DATA DESTRUCTION
- **0x800A** - CASCADE WIPE
- **0x800B** - HARDWARE SANITIZE
- **0x8019** - NETWORK KILL
- **0x8029** - COMMS BLACKOUT

*Quarantine enforced in code - these devices are blocked.*

---

## Common Tasks

### Activate Core Security Group

1. Launch control center
2. Focus Device Activation panel (bottom-left)
3. Navigate to Group 0
4. Select DSMIL0D0 (Master Controller)
5. Press ENTER
6. Watch activation in logs

### Browse TPM Operations

1. Focus Operation Monitor panel (bottom-right)
2. Find TPMControlDevice (0x8000)
3. Press ENTER to see 41 operations
4. Navigate to `get_status()`
5. Press E to execute

### Monitor System Health

1. Check Control Console panel (top-left)
2. View CPU, Memory, Load Average
3. Check DSMIL driver status
4. Monitor device nodes

---

## Logs and Output

### Live Logs
- **Top-Right Panel** - Real-time color-coded logs

### Log Files
```bash
/tmp/dsmil_guided_activation.log      # Device activation
/tmp/dsmil_operation_monitor.log      # Operation execution
/tmp/dsmil_execution_history.json     # Execution records
```

### View Logs
```bash
tail -f /tmp/dsmil_*.log
cat /tmp/dsmil_execution_history.json | jq .
```

---

## Reattach to Session

If you detach or lose connection:

```bash
tmux attach-session -t dsmil-control-center
```

Or shorter:

```bash
tmux a -t dsmil-control-center
```

---

## Stop Everything

```bash
# From outside session
tmux kill-session -t dsmil-control-center

# Or from inside
# Press Ctrl+B then type:
:kill-session
```

---

## Troubleshooting

### Session already exists?
```bash
tmux kill-session -t dsmil-control-center
sudo ./launch-dsmil-control-center.sh
```

### tmux not found?
```bash
sudo apt install tmux
```

### Permission denied?
```bash
sudo ./launch-dsmil-control-center.sh
```

### Panels too small?
Maximize terminal window, or zoom a pane with `Ctrl+B` then `Z`

---

## Statistics

- **Total Devices**: 84 (80 implemented, 5 quarantined, 23 unknown)
- **Total Operations**: 656
- **Total Registers**: 273
- **Most Complex Device**: TPMControlDevice (41 operations)
- **Implementation Coverage**: 74.1%

---

## Architecture

### Compute Resources
```
Intel Arc GPU:     40 TOPS (MILITARY mode)
Intel NPU 3720: 26.4 TOPS (MILITARY mode)
Intel NCS2 (2x):   20 TOPS (10 each)
                ─────────
Total:          86.4 TOPS (current)
                96.4 TOPS (with 3rd NCS2)
               137 TOPS (CLASSIFIED mode with 3 NCS2)
```

### Memory Architecture
```
Physical RAM:      64 GB
DSMIL Reserved:     2 GB (firmware-level)
OS-Visible RAM:    62 GB
Usable RAM:      55.8 GB (90% of 62 GB)
```

---

## For More Information

📖 **Comprehensive Guide**: `QUICK_START.md`
📖 **Device Reference**: `00-documentation/00-root-docs/DSMIL_CURRENT_REFERENCE.md`
📖 **Operation Monitor**: `02-ai-engine/README_OPERATION_MONITOR.md`
📖 **Enumeration Guide**: `DSMIL_ENUMERATION_GUIDE.md`

---

## Let's Go! 🚀

```bash
cd /home/user/LAT5150DRVMIL
sudo ./launch-dsmil-control-center.sh
```

**The integrated control center will launch with all monitoring and activation interfaces ready!**

---

**Platform**: Dell Latitude 5450 MIL-SPEC
**Version**: 2.0.0
**Classification**: DSMIL Subsystem Management
**Last Updated**: 2025-11-09
