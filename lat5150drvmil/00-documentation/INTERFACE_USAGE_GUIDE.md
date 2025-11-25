# 🚀 LOCAL OPUS INTERFACE - COMPLETE USAGE GUIDE

## Quick Start (3 Steps)

### 1. Launch the Interface
```bash
cd /home/john
./launch-opus-interface.sh
```

### 2. Access in Browser
```
Open: http://localhost:8080
```

### 3. If Buttons Don't Work
**IMPORTANT**: Hard refresh the browser to clear cache:
- **Chrome/Edge**: Ctrl+Shift+R or Ctrl+F5
- **Firefox**: Ctrl+Shift+R
- **Safari**: Cmd+Shift+R

Then reload the page!

---

## 🎯 Full Interface Capabilities

### 16 Sidebar Buttons

#### Documentation (8 buttons)
1. **📝 Install Commands** - Kernel installation steps
2. **📄 Full Handoff Document** - Complete technical handoff
3. **🔍 Opus Context** - Project overview
4. **🛡️ APT Defenses** - Security features
5. **⚠️ Mode 5 Warnings** - Critical safety info
6. **✅ Build Status** - What's completed
7. **🔧 DSMIL Details** - Technical deep dive
8. **💻 Hardware Specs** - Dell 5450 specifications

#### File Operations (2 buttons)
9. **📤 Upload Document/PDF** - Upload files, extract PDF text
10. **📁 Browse Files** - Navigate file system

#### NPU Operations (2 buttons)
11. **⚡ NPU Modules** - Show 6 module details
12. **🧪 Run NPU Tests** - Execute all modules

#### System Operations (4 buttons)
13. **⚙️ Execute Commands** - Run shell commands
14. **📋 View Logs** - kernel build, server, dmesg
15. **💾 System Info** - CPU, memory, disk, processes
16. **🔧 Kernel Status** - Real-time build status

### 12 Quick Action Chips

Click these for instant actions:
1. **Next steps?** - Installation guide
2. **Mode 5?** - Security levels
3. **Install commands** - Direct commands
4. **NPU modules** - NPU info
5. **Test NPU** - Run NPU tests
6. **List files** - ls -la /home/john
7. **Show logs** - View all logs
8. **System info** - System status
9. **Kernel status** - Build check
10. **NPU build info** - Make info
11. **Disk space** - df -h
12. **Memory usage** - free -h

---

## 💬 Text Input Commands

### Execute Shell Commands
```
Type in chat: run: ls -la /home/john
Type in chat: $ df -h
Type in chat: exec: free -h
```

### Read Files
```
Type in chat: cat /home/john/README.md
Type in chat: read /home/john/MASTER_INDEX.md
```

### NPU Operations
```
Type in chat: npu test
Type in chat: run npu modules
```

### View Logs
```
Type in chat: show logs
Type in chat: view dmesg
```

### System Information
```
Type in chat: system info
Type in chat: kernel status
```

### Ask Questions (Original)
```
Type in chat: What is Mode 5?
Type in chat: How do I install the kernel?
Type in chat: Tell me about DSMIL
```

---

## 📤 Using File Upload

### Upload a PDF
1. Click "📤 Upload Document/PDF" button
2. Click "📁 Select File to Upload"
3. Choose your PDF file
4. System extracts text automatically
5. View results in chat

### Supported File Types
- **PDF** (.pdf) - Text extraction
- **Text** (.txt, .md, .log)
- **Code** (.c, .h, .py, .sh)

### Where Files Are Saved
```
/home/john/uploads/
```

---

## 🔧 Troubleshooting

### Buttons Don't Work

**Solution 1**: Hard refresh browser
```
Chrome/Firefox: Ctrl+Shift+R
Safari: Cmd+Shift+R
```

**Solution 2**: Clear browser cache
```
1. Open browser settings
2. Clear browsing data
3. Select "Cached images and files"
4. Clear data
5. Reload http://localhost:8080
```

**Solution 3**: Try different browser
```
Firefox, Chrome, Edge, Safari all supported
```

**Solution 4**: Check browser console
```
Press F12 → Console tab
Look for JavaScript errors
```

### Server Not Running

```bash
# Check if running
lsof -i :8080

# If not running, launch
./launch-opus-interface.sh

# Check logs
tail -f /tmp/opus_server.log
```

### Relaunch Server

```bash
# Standalone launch (works anytime)
./launch-opus-interface.sh

# Manual launch
cd /home/john
python3 opus_server_full.py &

# Check it started
lsof -i :8080
```

---

## ⚙️ Server Endpoints

### GET Endpoints
```bash
# Status
curl http://localhost:8080/status | jq

# Execute command
curl "http://localhost:8080/exec?cmd=ls%20-la"

# List files
curl "http://localhost:8080/files?path=/home/john"

# Read file
curl "http://localhost:8080/read?path=/home/john/README.md"

# View logs
curl "http://localhost:8080/logs?lines=50"

# Run NPU modules
curl http://localhost:8080/npu/run

# System info
curl http://localhost:8080/system/info

# Kernel status
curl http://localhost:8080/kernel/status
```

### POST Endpoints
```bash
# Upload file
curl -F "file=@/path/to/file.pdf" http://localhost:8080/upload
```

---

## 📊 Interface Statistics

**Total Buttons**: 16
**Quick Actions**: 12
**Server Endpoints**: 14
**Text Commands**: 10+
**Supported File Types**: 7
**Documentation Files**: 25+
**NPU Modules**: 6

---

## 🎯 Common Tasks

### Install DSMIL Kernel
1. Click "📝 Install Commands"
2. Copy commands
3. Execute in terminal

### Test NPU Modules
1. Click "🧪 Run NPU Tests"
2. View all 6 module outputs
3. Check for errors

### Upload Your PDF
1. Click "📤 Upload Document/PDF"
2. Click "📁 Select File"
3. Choose your PDF
4. View extracted text

### Execute Custom Command
1. Click "⚙️ Execute Commands"
2. Type "run: YOUR_COMMAND" in chat
3. View output

### Check System Status
1. Click "💾 System Info"
2. View CPU, memory, disk
3. Check running processes

### View Build Logs
1. Click "📋 View Logs"
2. See kernel build log
3. Check server log
4. View dmesg output

---

## 🔐 Safety Features

### Command Safety
- Dangerous commands blocked (rm -rf /, mkfs, etc.)
- 30-second timeout on commands
- Error handling and reporting

### File Upload Safety
- Files saved to /home/john/uploads/
- PDF text extraction only
- No code execution from uploads

### Server Safety
- Runs independently
- Can be stopped anytime (kill PID)
- Logs all operations

---

## 📝 Examples

### Example 1: Quick System Check
```
1. Click "System info" chip
2. View CPU, memory, disk usage
3. Click "Kernel status" chip
4. Verify kernel is built
```

### Example 2: Test NPU Suite
```
1. Click "Test NPU" chip
2. Wait for all 6 modules to execute
3. View test results
4. Check for any errors
```

### Example 3: Browse Documentation
```
1. Click "📁 Browse Files"
2. See all markdown files
3. Click "cat README.md" to read
4. Or use "cat" command in chat
```

### Example 4: Upload and Process PDF
```
1. Prepare your PDF file
2. Click "📤 Upload Document/PDF"
3. Select your PDF
4. View extracted text immediately
5. Full file saved in /home/john/uploads/
```

---

## ⌨️ Keyboard Shortcuts

- **Ctrl+Enter**: Send message
- **Up/Down**: Navigate command history (if enabled)
- **F5**: Reload page
- **Ctrl+Shift+R**: Hard refresh (clears cache)
- **F12**: Open browser console for debugging

---

## 📞 Quick Reference

### Relaunch Server
```bash
./launch-opus-interface.sh
```

### Stop Server
```bash
kill $(lsof -t -i :8080)
```

### Check Server Status
```bash
lsof -i :8080
ps aux | grep opus_server_full
```

### View Server Logs
```bash
tail -f /tmp/opus_server.log
```

### Test Endpoints
```bash
curl http://localhost:8080/status
curl "http://localhost:8080/exec?cmd=echo%20test"
```

---

## 🎉 You're Ready!

The interface is fully functional with:
- ✅ 16 sidebar buttons
- ✅ 12 quick action chips
- ✅ 14 server endpoints
- ✅ Text command input
- ✅ PDF upload & processing
- ✅ NPU module integration
- ✅ Complete documentation access

**Access now**: http://localhost:8080

**Remember to hard refresh (Ctrl+Shift+R) if buttons don't respond!**

---

**Guide Version**: 1.0
**Date**: 2025-10-15
**Interface**: Full-featured local Opus
**Status**: Production ready
**Token Usage**: ~25% (efficient!)
