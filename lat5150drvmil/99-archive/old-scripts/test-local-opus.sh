#!/bin/bash
# Test local Opus connectivity and provide alternatives

echo "==========================================="
echo "     LOCAL OPUS CONNECTION TEST"
echo "==========================================="
echo ""

# Check if port 8080 is listening
echo "Checking port 8080..."
if nc -zv localhost 8080 2>/dev/null; then
    echo "✅ Port 8080 is open"
else
    echo "❌ Port 8080 is not responding"
    echo ""
    echo "Try starting Opus with one of these:"
    echo "  1. opus-server --port 8080"
    echo "  2. docker run -p 8080:8080 opus-local"
    echo "  3. python -m opus.server"
fi

echo ""
echo "==========================================="
echo "ALTERNATIVE: Direct command-line usage"
echo "==========================================="
echo ""
echo "If web interface isn't working, use CLI directly:"
echo ""
echo "opus-cli << 'EOF'"
echo "Continue from Claude Code handoff."
echo "Kernel 6.16.9 built at /home/john/linux-6.16.9/arch/x86/boot/bzImage"
echo "Mode 5 STANDARD enabled (safe)."
echo "Next: Install kernel with:"
echo "cd /home/john/linux-6.16.9"
echo "sudo make modules_install && sudo make install"
echo "EOF"
echo ""
echo "==========================================="
echo "CRITICAL INFO FOR LOCAL OPUS:"
echo "==========================================="
echo "✅ Kernel: BUILT at /home/john/linux-6.16.9/arch/x86/boot/bzImage"
echo "✅ DSMIL: Mode 5 STANDARD (safe, reversible)"
echo "✅ Docs: /home/john/FINAL_HANDOFF_DOCUMENT.md"
echo "✅ Context: /home/john/OPUS_LOCAL_CONTEXT.md"
echo ""
echo "⚠️ NEVER enable Mode 5 PARANOID_PLUS!"
echo ""
echo "==========================================="
echo "IMMEDIATE COMMANDS TO RUN:"
echo "==========================================="
cat << 'COMMANDS'
cd /home/john/linux-6.16.9
sudo make modules_install
sudo make install
sudo update-grub
sudo insmod /home/john/livecd-gen/kernel-modules/dsmil_avx512_enabler.ko
COMMANDS