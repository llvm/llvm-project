#!/bin/bash
# DSMIL AI System Verification

echo "═══════════════════════════════════════════════════════════"
echo "DSMIL AI SYSTEM - HEALTH CHECK"
echo "═══════════════════════════════════════════════════════════"
echo ""

# Check Ollama
echo "1. Checking Ollama models..."
ollama list 2>/dev/null | grep -E "llama3.2|codellama" && echo "   ✅ AI models available" || echo "   ❌ Models missing"
echo ""

# Check server
echo "2. Checking web server..."
curl -s http://localhost:9876/ai/status >/dev/null 2>&1 && echo "   ✅ Server running on port 9876" || echo "   ❌ Server not responding"
echo ""

# Check AVX-512
echo "3. Checking AVX-512 unlock..."
if [ -f /proc/dsmil_avx512 ]; then
    grep -q "Unlock Successful: YES" /proc/dsmil_avx512 2>/dev/null && echo "   ✅ AVX-512 unlocked (12 P-cores)" || echo "   ⚠ AVX-512 driver loaded but not unlocked"
else
    echo "   ⚠ AVX-512 driver not loaded (run: sudo insmod ...)"
fi
echo ""

# Check NPU mode
echo "4. Checking NPU military mode..."
if [ -f ~/.claude/npu-military.env ]; then
    grep -q "NPU_MILITARY_MODE=1" ~/.claude/npu-military.env && echo "   ✅ NPU military mode enabled (26.4 TOPS)" || echo "   ⚠ NPU in standard mode (11 TOPS)"
else
    echo "   ⚠ NPU config not found"
fi
echo ""

# Check huge pages
echo "5. Checking huge pages..."
HUGEPAGES=$(grep HugePages_Total /proc/meminfo | awk '{print $2}')
if [ "$HUGEPAGES" -ge 16000 ]; then
    echo "   ✅ Huge pages allocated: ${HUGEPAGES} pages (32GB)"
else
    echo "   ⚠ Huge pages: ${HUGEPAGES} (expected 16384)"
fi
echo ""

# Check files
echo "6. Checking system files..."
FILES=(
    "~/dsmil_ai_engine.py"
    "~/dsmil_military_mode.py"
    "~/opus_server_full.py"
    "~/military_terminal.html"
    "~/flux_idle_provider.py"
    "~/rag_system.py"
    "~/smart_paper_collector.py"
)
MISSING=0
for file in "${FILES[@]}"; do
    expanded=$(eval echo $file)
    if [ -f "$expanded" ]; then
        echo "   ✅ $(basename $expanded)"
    else
        echo "   ❌ $(basename $expanded) missing"
        MISSING=$((MISSING + 1))
    fi
done
echo ""

# Summary
echo "═══════════════════════════════════════════════════════════"
if [ "$MISSING" -eq 0 ]; then
    echo "SYSTEM STATUS: ✅ FULLY OPERATIONAL"
    echo ""
    echo "Access your system: http://localhost:9876"
else
    echo "SYSTEM STATUS: ⚠ PARTIAL ($MISSING files missing)"
fi
echo "═══════════════════════════════════════════════════════════"
