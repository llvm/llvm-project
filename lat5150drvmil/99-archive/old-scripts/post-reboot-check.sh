#!/bin/bash
# Post-Reboot AVX-512 Verification Script
# Run this after rebooting into DSMIL kernel

echo "═══════════════════════════════════════════════════════════"
echo "  POST-REBOOT VERIFICATION - AVX-512 UNLOCK"
echo "═══════════════════════════════════════════════════════════"
echo ""

echo "[1/7] Kernel Version:"
uname -r
echo ""

echo "[2/7] Microcode Version:"
grep microcode /proc/cpuinfo | head -1
echo ""

echo "[3/7] Boot Parameters:"
cat /proc/cmdline
echo ""

echo "[4/7] DSMIL AVX-512 Module Status:"
lsmod | grep dsmil || echo "Module not loaded - loading now..."
if ! lsmod | grep -q dsmil_avx512_enabler; then
    echo "Loading DSMIL AVX-512 enabler module..."
    sudo modprobe dsmil_avx512_enabler 2>&1
    sleep 1
fi
echo ""

echo "[5/7] DSMIL AVX-512 Unlock Status:"
if [ -f /proc/dsmil_avx512 ]; then
    cat /proc/dsmil_avx512
else
    echo "ERROR: /proc/dsmil_avx512 not found"
    echo "Module may not be loaded properly"
fi
echo ""

echo "[6/7] AVX-512 CPU Flags Check:"
AVX512_COUNT=$(cat /proc/cpuinfo | grep flags | grep -o "avx512[^ ]*" | sort -u | wc -l)
if [ "$AVX512_COUNT" -gt 0 ]; then
    echo "✅ AVX-512 FLAGS FOUND: $AVX512_COUNT unique flags"
    cat /proc/cpuinfo | grep flags | head -1 | grep -o "avx512[^ ]*" | sort -u
else
    echo "❌ NO AVX-512 FLAGS DETECTED"
    echo "Available AVX flags:"
    cat /proc/cpuinfo | grep flags | head -1 | grep -o "avx[^ ]*"
fi
echo ""

echo "[7/7] NPU and AI Hardware Status:"
ls -l /dev/accel0 2>/dev/null && echo "✅ NPU device active" || echo "❌ NPU device missing"
lsmod | grep intel_vpu && echo "✅ Intel VPU driver loaded" || echo "❌ Intel VPU driver missing"
echo ""

echo "═══════════════════════════════════════════════════════════"
echo "  SYSTEM STATUS SUMMARY"
echo "═══════════════════════════════════════════════════════════"
echo ""

# Temperature
echo "Temperature:"
sensors 2>/dev/null | grep -E "Package|Core 0" || echo "sensors not available"
echo ""

# Services
echo "AI Services:"
systemctl is-active ollama && echo "✅ Ollama: ACTIVE" || echo "❌ Ollama: INACTIVE"
pgrep -f opus_server && echo "✅ Military Terminal: RUNNING" || echo "❌ Military Terminal: STOPPED"
echo ""

if [ "$AVX512_COUNT" -gt 0 ]; then
    echo "🎉 SUCCESS: AVX-512 UNLOCKED!"
    echo ""
    echo "Next steps:"
    echo "1. Test AVX-512 performance with benchmarks"
    echo "2. Restart military terminal: python3 /home/john/opus_server_full.py &"
    echo "3. Access interface: http://localhost:9876"
else
    echo "⚠️  AVX-512 NOT UNLOCKED"
    echo ""
    echo "Troubleshooting:"
    echo "1. Check module loaded: lsmod | grep dsmil"
    echo "2. Check kernel messages: dmesg | grep -i avx"
    echo "3. Check DSMIL status: cat /proc/dsmil_avx512"
    echo "4. Verify microcode version (should be 0x1c-0x24)"
fi

echo "═══════════════════════════════════════════════════════════"
