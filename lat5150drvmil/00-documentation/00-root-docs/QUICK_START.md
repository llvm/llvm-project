# DSMIL Control Center - Quick Start

## 🚀 Launch Everything Now!

```bash
cd /home/user/LAT5150DRVMIL
sudo ./launch-dsmil-control-center.sh
```

That's it! This launches the complete DSMIL activation and monitoring system.

---

## What You Get

The launcher creates a **4-panel tmux session** with:

```
┌─────────────────────────┬─────────────────────────┐
│                         │                         │
│   CONTROL CONSOLE       │    SYSTEM LOGS          │
│                         │                         │
│   • System status       │   • Live log viewer     │
│   • Hardware info       │   • Error tracking      │
│   • Driver status       │   • Activation events   │
│   • Live metrics        │   • Color-coded output  │
│                         │                         │
├─────────────────────────┼─────────────────────────┤
│                         │                         │
│   DEVICE ACTIVATION     │   OPERATION MONITOR     │
│                         │                         │
│   • 84 DSMIL devices    │   • 656 operations      │
│   • Group navigation    │   • Device browsing     │
│   • Activation control  │   • Operation execution │
│   • Dependency tracking │   • Execution history   │
│                         │                         │
└─────────────────────────┴─────────────────────────┘
```

---

## First Time Setup

### 1. Install tmux (if needed)

```bash
sudo apt update
sudo apt install tmux -y
```

### 2. Verify files present

```bash
ls -la 02-ai-engine/dsmil_*.py
# Should show:
#   dsmil_guided_activation.py
#   dsmil_operation_monitor.py
```

### 3. Optional: Load DSMIL driver

```bash
# Only if running on actual hardware
sudo modprobe dsmil-84dev   # legacy alias: dsmil-72dev

# Verify
lsmod | grep dsmil
ls /dev/dsmil*
```

---

## Navigation Guide

### tmux Keyboard Shortcuts

All tmux commands start with **Ctrl+B**, then release and press the action key:

| Action | Keys | Description |
|--------|------|-------------|
| **Switch Panes** | `Ctrl+B` then `Arrow Keys` | Move between the 4 panels |
| **Detach** | `Ctrl+B` then `D` | Detach (session keeps running) |
| **Scroll Mode** | `Ctrl+B` then `[` | Scroll back in logs (Q to exit) |
| **Zoom Pane** | `Ctrl+B` then `Z` | Full-screen current pane (toggle) |
| **Help** | `Ctrl+B` then `?` | Show all keybindings |

### Device Activation Panel (Bottom-Left)

| Key | Action |
|-----|--------|
| `←/→` | Switch between device groups (0-6) |
| `↑/↓` | Navigate devices in current group |
| `ENTER` | Activate selected device |
| `I` | Show device info dialog |
| `S` | Export activation status |
| `Q` | Quit |

### Operation Monitor Panel (Bottom-Right)

| Key | Action |
|-----|--------|
| `↑/↓` | Navigate devices or operations |
| `ENTER` | View operations / operation details |
| `E` | Execute selected operation |
| `L` | View execution log |
| `ESC` | Go back |
| `Q` | Quit |

---

## Common Tasks

### Activate Device Group 0 (Core Security)

1. Launch control center
2. Click/focus on **Device Activation** panel (bottom-left)
3. Use `←/→` to select "Group 0"
4. Use `↑/↓` to select device (e.g., DSMIL0D0 - Master Controller)
5. Press `ENTER` to activate
6. Watch logs in **System Logs** panel (top-right)

### Execute TPM Operations

1. Focus on **Operation Monitor** panel (bottom-right)
2. Navigate to `TPMControlDevice (0x8000)`
3. Press `ENTER` to view its 41 operations
4. Navigate to operation (e.g., `get_status()`)
5. Press `E` to execute
6. Press `L` to view execution log

### Monitor System Health

1. Focus on **Control Console** panel (top-left)
2. Watch live metrics: CPU, Memory, Load Average
3. Check DSMIL driver status
4. Monitor device node presence

### View All Logs

1. Focus on **System Logs** panel (top-right)
2. Color-coded output:
   - 🔴 **Red**: Errors
   - 🟡 **Yellow**: Warnings
   - 🟢 **Green**: Success
   - 🔵 **Cyan**: Info
3. Press `Ctrl+B` then `[` to scroll back
4. Press `Q` to exit scroll mode

---

## Reattach to Running Session

If you detach or disconnect, reattach with:

```bash
tmux attach-session -t dsmil-control-center
```

Or the shorter version:

```bash
tmux a -t dsmil-control-center
```

---

## Stop Everything

### Option 1: From Inside Session

Press `Q` in each panel to quit the applications, then:

```bash
# Ctrl+B then type:
:kill-session
```

### Option 2: From Outside Session

```bash
tmux kill-session -t dsmil-control-center
```

---

## Troubleshooting

### "Session already exists"

The launcher will ask if you want to kill and restart:

```
Session 'dsmil-control-center' already exists!
Kill existing session and restart? (y/N)
```

Press `y` to restart, or `n` to attach to existing session.

### "tmux not found"

Install tmux:

```bash
sudo apt update && sudo apt install tmux
```

### "Required file missing"

Ensure you're in the project root:

```bash
cd /home/user/LAT5150DRVMIL
./launch-dsmil-control-center.sh
```

### Device Activation Shows Errors

Check if running with sudo:

```bash
sudo ./launch-dsmil-control-center.sh
```

Check DSMIL driver loaded:

```bash
lsmod | grep dsmil
```

### Panels Are Too Small

Maximize terminal window before launching, or zoom a pane:

```
Ctrl+B then Z    # Toggle pane zoom
```

---

## Advanced Usage

### Custom tmux Layout

After launching, you can resize panes:

```
Ctrl+B then Ctrl+Arrow    # Resize current pane
Ctrl+B then Alt+1         # Even horizontal layout
Ctrl+B then Alt+2         # Even vertical layout
```

### Save Execution History

Both tools save execution data:

```bash
# Device activation history
cat /tmp/dsmil-logs/activation_status_*.txt

# Operation execution history
cat /tmp/dsmil_execution_history.json
```

### Run Tools Separately (No tmux)

If you prefer separate terminals:

```bash
# Terminal 1: Device Activation
sudo python3 02-ai-engine/dsmil_guided_activation.py

# Terminal 2: Operation Monitor
sudo python3 02-ai-engine/dsmil_operation_monitor.py

# Terminal 3: Watch logs
tail -f /tmp/dsmil_*.log
```

---

## What's Happening Behind the Scenes

### Control Console
- Shows system hardware status
- Monitors DSMIL driver
- Displays live CPU/Memory/Load metrics
- Checks device node availability

### System Logs
- Tails both log files simultaneously
- Color-codes by severity
- Shows activation events
- Displays operation results

### Device Activation
- Enumerates all 84 DSMIL devices (0x8000-0x806B)
- Shows dependencies and safety status
- Integrates with existing DSMILDeviceActivator
- Prevents quarantined device activation

### Operation Monitor
- Loads 656 operations from capabilities JSON
- Shows signatures, parameters, return types
- Executes safe read-only operations
- Tracks execution history with timestamps

---

## Files Created During Session

### Logs
- `/tmp/dsmil_guided_activation.log` - Device activation log
- `/tmp/dsmil_operation_monitor.log` - Operation execution log
- `/tmp/dsmil-logs/control_console.sh` - Console helper script
- `/tmp/dsmil-logs/log_monitor.sh` - Log monitoring script

### Execution Data
- `/tmp/dsmil_execution_history.json` - Operation execution records
- `/tmp/dsmil-logs/session_info.txt` - Session information

### Status Exports
- Generated when pressing `S` in Device Activation panel
- Timestamped files in `/tmp/` or current directory

---

## System Requirements

- **OS**: Linux (tested on Ubuntu/Debian)
- **Shell**: Bash 4.0+
- **Terminal**: Any with 80x24 minimum (120x40 recommended)
- **Python**: 3.6+
- **tmux**: 2.0+ (install with `apt install tmux`)
- **Permissions**: Root/sudo for hardware access (optional for simulation)

---

## Performance Notes

- **CPU Usage**: Minimal (~2-5% across all panels)
- **Memory**: ~50-100MB total
- **Logs**: Rotate manually if needed (check `/tmp/`)
- **Network**: None required (fully local)

---

## Safety Features

✓ **Quarantine Enforcement**: 5 dangerous devices blocked
✓ **Read-Only Default**: Only safe operations execute automatically
✓ **Complete Logging**: Every action logged with timestamp
✓ **Graceful Degradation**: Works without hardware/driver
✓ **Error Recovery**: Robust exception handling in all panels

---

## Getting Help

### In-Session Help

```
Ctrl+B then ?     # tmux keybindings
Q in any panel    # Quit that panel
```

### Documentation

```bash
# Device activation docs
cat 02-ai-engine/README_OPERATION_MONITOR.md

# Operation monitor docs
cat 02-ai-engine/README_OPERATION_MONITOR.md

# Device reference
cat 00-documentation/00-root-docs/DSMIL_CURRENT_REFERENCE.md
```

### Logs

```bash
# View activation log
tail -f /tmp/dsmil_guided_activation.log

# View operation log
tail -f /tmp/dsmil_operation_monitor.log

# View execution history
cat /tmp/dsmil_execution_history.json | jq .
```

---

## Ready to Go!

```bash
cd /home/user/LAT5150DRVMIL
sudo ./launch-dsmil-control-center.sh
```

**Press ENTER in each bottom panel when ready to launch the TUIs!**

The system will:
1. ✓ Check prerequisites
2. ✓ Verify required files
3. ✓ Create tmux session with 4 panes
4. ✓ Launch monitoring tools
5. ✓ Display live system status
6. ✓ Ready for device activation!

---

**Version**: 1.0.0
**Platform**: Dell Latitude 5450 MIL-SPEC
**Classification**: DSMIL Subsystem Management
**Last Updated**: 2025-11-09
