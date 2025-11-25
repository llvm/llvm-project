# 🔐 LOCAL OPUS INTERFACE - COMPLETE GUIDE

## Quick Start

Access the interface at: **http://localhost:8080**

The server is already running on port 8080 with full functionality.

## Interface Features

### Main Components

1. **Status Bar** (Top)
   - Real-time kernel build status
   - DSMIL device count
   - Mode 5 security level
   - Overall system status

2. **Sidebar** (Left)
   - Quick action buttons for all documentation
   - Project statistics
   - Critical warning display

3. **Chat Area** (Center)
   - Chat-style message interface
   - Auto-scroll to latest messages
   - Copy button on each message

4. **Input Area** (Bottom)
   - Text input with Ctrl+Enter to send
   - Quick action chips for common questions
   - Send button

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| **Ctrl+Enter** | Send message |
| **Ctrl+E** | Export chat to markdown |
| **Ctrl+L** | Clear chat history |
| **Ctrl+K** | Copy all install commands |
| **Ctrl+1** | Load install commands |
| **Ctrl+2** | Load full handoff document |
| **Ctrl+3** | Load Opus context |
| **Ctrl+4** | Load APT defenses |
| **Ctrl+5** | Load Mode 5 warnings |
| **Ctrl+6** | Load build status |
| **Ctrl+7** | Load DSMIL details |
| **Ctrl+8** | Load hardware specs |
| **Up Arrow** | Previous command in history |
| **Down Arrow** | Next command in history |

## Quick Actions (Sidebar Buttons)

### 📝 Install Commands
Complete step-by-step installation guide including:
- Kernel module installation
- GRUB configuration
- Post-boot verification
- AVX-512 module loading
- livecd-gen compilation

### 📄 Full Handoff Document
Complete military-spec handoff documentation with:
- DSMIL framework details
- Mode 5 platform integrity
- APT-level defense capabilities
- Hardware specifications
- Installation procedures

### 🔍 Opus Context
Project overview including:
- Build status
- DSMIL integration details
- Hardware target
- Completed tasks
- Pending tasks

### 🛡️ APT Defenses
Advanced Persistent Threat defense mechanisms:
- Protection against APT-41, Lazarus, APT29, etc.
- IOMMU/DMA protection
- Memory encryption (TME)
- Firmware attestation
- Boot chain security

### ⚠️ Mode 5 Warnings
**CRITICAL SAFETY INFORMATION:**
- STANDARD: Safe, reversible (current)
- ENHANCED: Partially reversible
- PARANOID: Permanent lockdown
- **PARANOID_PLUS: NEVER USE - Will brick system**

### ✅ Build Status
Complete build report:
- Compilation fixes applied
- Kernel details
- DSMIL driver info
- Pending tasks
- Token usage stats

### 🔧 DSMIL Details
Deep technical information:
- 84 device categories
- SMI interface details
- Mode 5 integration
- Hardware requirements
- Safety notes

### 💻 Hardware Specs
Target system specifications:
- Dell Latitude 5450 details
- Intel Core Ultra 7 165H CPU
- Intel NPU 3720
- TPM 2.0 details
- Security features

## Chat Input Examples

Type these questions in the input box:

- "What are the next installation steps?"
- "Tell me about Mode 5 STANDARD"
- "Show me the install commands"
- "What APT defenses are included?"
- "Tell me about DSMIL"
- "What hardware is supported?"
- "Show me the build status"

The interface will automatically respond with relevant information.

## Quick Action Chips

Located above the input box for one-click questions:
- **Next steps?** - Installation guide
- **Mode 5?** - Security level explanation
- **Install commands** - Command list
- **APT defenses?** - Security features
- **DSMIL info** - Driver details

## Features

### Auto-Save
- Chat history saved every 30 seconds
- Automatic restore on page reload
- 24-hour backup retention

### Export Chat
- Press Ctrl+E or use export function
- Downloads complete chat as markdown
- Includes timestamps and all messages

### Copy Functionality
- Each message has a copy button
- Ctrl+K copies all install commands
- Preserves formatting

### Command History
- Up/Down arrows navigate previous commands
- Stores last 50 commands
- Persistent across page reloads

## Project Statistics (Sidebar)

- **Driver Size**: 584KB compiled
- **Lines of Code**: 2800+ military-spec
- **DSMIL Devices**: 84 endpoints
- **Build Time**: ~15 minutes

## Critical Warnings

The interface displays a persistent warning:
```
⚠️ NEVER ENABLE PARANOID_PLUS MODE!
Will permanently brick system
```

This is a safety reminder about Mode 5 security levels.

## Server Information

### Backend Server (opus_server.py)
- **Port**: 8080
- **Endpoints**:
  - `/` - Main interface
  - `/commands` - Installation commands
  - `/handoff` - Full documentation
  - `/status` - System status JSON

### Starting/Stopping

**Check if running:**
```bash
lsof -i :8080
```

**Stop server:**
```bash
pkill -f opus_server.py
```

**Start server:**
```bash
cd /home/john && python3 opus_server.py &
```

## Files Overview

### Interface Files
- `/home/john/opus_interface.html` - Main HTML interface
- `/home/john/opus_server.py` - Backend server
- `/home/john/enhance_interface.js` - Additional features

### Documentation Files
- `/home/john/COMPLETE_MILITARY_SPEC_HANDOFF.md` - Full handoff
- `/home/john/FINAL_HANDOFF_DOCUMENT.md` - Project status
- `/home/john/OPUS_LOCAL_CONTEXT.md` - Opus context
- `/home/john/APT_ADVANCED_SECURITY_FEATURES.md` - APT defenses
- `/home/john/MODE5_SECURITY_LEVELS_WARNING.md` - Safety warnings
- `/home/john/DSMIL_INTEGRATION_SUCCESS.md` - Integration details

### Kernel Files
- `/home/john/linux-6.16.9/arch/x86/boot/bzImage` - Built kernel (13MB)
- `/home/john/linux-6.16.9/drivers/platform/x86/dell-milspec/` - DSMIL driver

## Troubleshooting

### Interface won't load
```bash
# Check if server is running
lsof -i :8080

# If not running, start it
cd /home/john && python3 opus_server.py &

# Access at http://localhost:8080
```

### Buttons don't respond
- Hard refresh: Ctrl+Shift+R
- Clear browser cache
- Check browser console (F12) for errors

### Chat history lost
- Check localStorage in browser dev tools
- Export chat regularly with Ctrl+E
- Auto-save runs every 30 seconds

### Can't copy text
- Use the copy button on each message
- Or manually select and copy
- Ctrl+K for all commands

## Security Notes

1. This interface runs **locally only**
2. No data sent to external servers
3. All information is from pre-built documentation
4. Safe to use for handoff purposes

## Next Steps After Using Interface

1. **Install the kernel:**
   ```bash
   cd /home/john/linux-6.16.9
   sudo make modules_install
   sudo make install
   sudo update-grub
   ```

2. **Configure GRUB** (add to /etc/default/grub):
   ```
   GRUB_CMDLINE_LINUX="intel_iommu=on mode5.level=standard"
   ```

3. **Reboot and verify:**
   ```bash
   sudo reboot
   dmesg | grep "MIL-SPEC"
   ```

4. **Load AVX-512:**
   ```bash
   sudo insmod /home/john/livecd-gen/kernel-modules/dsmil_avx512_enabler.ko
   ```

5. **Compile livecd-gen modules:**
   ```bash
   cd /home/john/livecd-gen
   for m in ai_hardware_optimizer meteor_lake_scheduler dell_platform_optimizer tpm_kernel_security avx512_optimizer; do
       gcc -O3 -march=native ${m}.c -o ${m}
   done
   ```

## Support

All documentation is available through the interface buttons and in the `/home/john/` directory.

For the complete technical handoff:
```bash
cat /home/john/COMPLETE_MILITARY_SPEC_HANDOFF.md
```

---

**Remember**: Mode 5 is currently set to STANDARD (safe and reversible). Never enable PARANOID_PLUS!

Interface Version: 1.0
Date: 2025-10-15
Built by: Claude Code (Sonnet 4.5)
Handoff to: Local Opus