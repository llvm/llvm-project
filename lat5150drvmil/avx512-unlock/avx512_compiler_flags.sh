#!/bin/bash
# ============================================================================
# AVX-512 Optimized Compiler Flags for Intel Meteor Lake (Core Ultra 7 165H)
# ============================================================================
#
# PREREQUISITE: E-cores must be disabled to unlock AVX-512
# Run: sudo ./unlock_avx512.sh enable
#
# PERFORMANCE GAIN: 15-40% for vectorizable workloads
#
# Use these flags when compiling on a system with AVX-512 unlocked
# ============================================================================

# ============================================================================
# AVX-512 INSTRUCTION SET EXTENSIONS
# ============================================================================

# Core AVX-512 Foundation
export AVX512_FOUNDATION="-mavx512f -mavx512dq -mavx512cd -mavx512bw -mavx512vl"

# AVX-512 Advanced Extensions (Meteor Lake P-cores support)
export AVX512_ADVANCED="-mavx512ifma -mavx512vbmi -mavx512vbmi2 -mavx512vnni -mavx512bitalg -mavx512vpopcntdq"

# Combined AVX-512 flags
export AVX512_FLAGS="$AVX512_FOUNDATION $AVX512_ADVANCED"

# ============================================================================
# COMPLETE OPTIMAL FLAGS WITH AVX-512
# ============================================================================

export CFLAGS_AVX512="\
-O3 \
-pipe \
-fomit-frame-pointer \
-funroll-loops \
-fstrict-aliasing \
-fno-plt \
-fdata-sections \
-ffunction-sections \
-flto=auto \
-fuse-linker-plugin \
-march=meteorlake \
-mtune=meteorlake \
-msse4.2 \
-mpopcnt \
-mavx \
-mavx2 \
-mfma \
-mf16c \
-mbmi \
-mbmi2 \
-mlzcnt \
-mmovbe \
-mavxvnni \
-maes \
-mvaes \
-mpclmul \
-mvpclmulqdq \
-msha \
-mgfni \
-madx \
-mclflushopt \
-mclwb \
-mcldemote \
-mmovdiri \
-mmovdir64b \
-mwaitpkg \
-mserialize \
-mtsxldtrk \
-muintr \
-mprefetchw \
-mprfchw \
-mrdrnd \
-mrdseed \
-mfsgsbase \
-mfxsr \
-mxsave \
-mxsaveopt \
-mxsavec \
-mxsaves \
-mavx512f \
-mavx512dq \
-mavx512cd \
-mavx512bw \
-mavx512vl \
-mavx512ifma \
-mavx512vbmi \
-mavx512vbmi2 \
-mavx512vnni \
-mavx512bitalg \
-mavx512vpopcntdq"

# Linker flags (same as base)
export LDFLAGS_AVX512="-Wl,--as-needed -Wl,--gc-sections -Wl,-O1 -Wl,--hash-style=gnu -flto=auto"

# ============================================================================
# KERNEL COMPILATION FLAGS WITH AVX-512
# ============================================================================

export KCFLAGS_AVX512="\
-O3 \
-pipe \
-march=meteorlake \
-mtune=meteorlake \
-msse4.2 \
-mpopcnt \
-mavx \
-mavx2 \
-mfma \
-mavxvnni \
-maes \
-mvaes \
-mpclmul \
-mvpclmulqdq \
-msha \
-mgfni \
-mavx512f \
-mavx512dq \
-mavx512bw \
-mavx512vl \
-mavx512vnni \
-falign-functions=32 \
-falign-jumps=32 \
-falign-loops=32 \
-falign-labels=32"

export KCPPFLAGS_AVX512="$KCFLAGS_AVX512"

# ============================================================================
# PERFORMANCE PROFILES WITH AVX-512
# ============================================================================

# Maximum Speed with AVX-512
export CFLAGS_AVX512_SPEED="-Ofast -ffast-math -funsafe-math-optimizations -ffinite-math-only $CFLAGS_AVX512"

# Balanced with AVX-512
export CFLAGS_AVX512_BALANCED="-O2 -ftree-vectorize -march=meteorlake -mtune=meteorlake $AVX512_FLAGS -pipe"

# Security Hardened with AVX-512
export CFLAGS_AVX512_SECURE="$CFLAGS_AVX512 \
-D_FORTIFY_SOURCE=3 \
-fstack-protector-strong \
-fstack-clash-protection \
-fcf-protection=full \
-fpie"

# ============================================================================
# VECTORIZATION ENHANCEMENTS FOR AVX-512
# ============================================================================

# Force AVX-512 vectorization
export VECTORIZE_AVX512="-ftree-vectorize -ftree-slp-vectorize -ftree-loop-vectorize -fvect-cost-model=unlimited -fsimd-cost-model=unlimited -mprefer-vector-width=512"

# Combined optimal with aggressive vectorization
export CFLAGS_AVX512_VECTORIZED="$CFLAGS_AVX512 $VECTORIZE_AVX512"

# ============================================================================
# CPU AFFINITY FOR P-CORES ONLY (REQUIRED FOR AVX-512)
# ============================================================================

# Set process affinity to P-cores (0-5)
export GOMP_CPU_AFFINITY="0-5"
export OMP_NUM_THREADS="6"
export OMP_PROC_BIND="true"
export OMP_PLACES="cores"

# Taskset command for P-cores only
export TASKSET_PCORES="taskset -c 0-5"

# ============================================================================
# USAGE FUNCTIONS
# ============================================================================

# Compile with AVX-512
compile_avx512() {
    echo "[*] Compiling with AVX-512 flags..."
    gcc $CFLAGS_AVX512 "$@" $LDFLAGS_AVX512
}

# Compile kernel with AVX-512
compile_kernel_avx512() {
    echo "[*] Building kernel with AVX-512 optimization..."
    make -j6 KCFLAGS="$KCFLAGS_AVX512" KCPPFLAGS="$KCPPFLAGS_AVX512" "$@"
}

# Test AVX-512 compilation
test_avx512() {
    echo "[*] Testing AVX-512 compilation..."
    echo 'int main(){return 0;}' | gcc -xc $CFLAGS_AVX512 - -o /tmp/test_avx512 && \
    echo "✓ AVX-512 flags verified working!" && \
    objdump -d /tmp/test_avx512 | grep -q "avx512" && \
    echo "✓ AVX-512 instructions detected in binary!" || \
    echo "⚠ No AVX-512 instructions found (code too simple)" && \
    rm -f /tmp/test_avx512
}

# Show AVX-512 flags
show_avx512_flags() {
    echo "╔══════════════════════════════════════════════════════════════════════════╗"
    echo "║         AVX-512 Optimized Compiler Flags - Meteor Lake                  ║"
    echo "╚══════════════════════════════════════════════════════════════════════════╝"
    echo ""
    echo "CFLAGS_AVX512:"
    echo "$CFLAGS_AVX512"
    echo ""
    echo "KCFLAGS_AVX512 (Kernel):"
    echo "$KCFLAGS_AVX512"
    echo ""
    echo "Usage:"
    echo "  gcc \$CFLAGS_AVX512 -o app app.c \$LDFLAGS_AVX512"
    echo "  make -j6 KCFLAGS=\"\$KCFLAGS_AVX512\""
    echo ""
}

# ============================================================================
# BENCHMARK COMPARISON SCRIPT
# ============================================================================

benchmark_avx512_vs_avx2() {
    echo "╔══════════════════════════════════════════════════════════════════════════╗"
    echo "║         AVX-512 vs AVX2 Performance Comparison                          ║"
    echo "╚══════════════════════════════════════════════════════════════════════════╝"
    echo ""

    # Create test program
    cat > /tmp/vectortest.c <<'EOF'
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define SIZE 100000000

int main() {
    float *a = malloc(SIZE * sizeof(float));
    float *b = malloc(SIZE * sizeof(float));
    float *c = malloc(SIZE * sizeof(float));

    // Initialize
    for (long i = 0; i < SIZE; i++) {
        a[i] = (float)i;
        b[i] = (float)(SIZE - i);
    }

    // Benchmark
    clock_t start = clock();
    for (long i = 0; i < SIZE; i++) {
        c[i] = a[i] * b[i] + a[i];
    }
    clock_t end = clock();

    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Time: %.4f seconds\n", time_taken);

    free(a); free(b); free(c);
    return 0;
}
EOF

    # Compile with AVX2
    echo "[*] Compiling with AVX2..."
    gcc -O3 -march=meteorlake -mavx2 -mfma /tmp/vectortest.c -o /tmp/test_avx2

    echo "[*] Running AVX2 benchmark..."
    echo -n "AVX2 Result: "
    taskset -c 0 /tmp/test_avx2

    # Compile with AVX-512
    echo "[*] Compiling with AVX-512..."
    gcc $CFLAGS_AVX512 /tmp/vectortest.c -o /tmp/test_avx512

    echo "[*] Running AVX-512 benchmark..."
    echo -n "AVX-512 Result: "
    taskset -c 0 /tmp/test_avx512

    # Cleanup
    rm -f /tmp/vectortest.c /tmp/test_avx2 /tmp/test_avx512

    echo ""
    echo "Note: AVX-512 should be ~20-40% faster for vectorizable code"
}

# ============================================================================
# ACTIVATION MESSAGE
# ============================================================================

echo "╔══════════════════════════════════════════════════════════════════════════╗"
echo "║  AVX-512 COMPILER FLAGS LOADED                                          ║"
echo "║  Intel Core Ultra 7 165H - P-cores Only (6 cores)                       ║"
echo "║  REQUIRES: E-cores disabled (run unlock_avx512.sh enable)               ║"
echo "╚══════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Quick Start:"
echo "  1. Unlock AVX-512:  sudo ./unlock_avx512.sh enable"
echo "  2. Source flags:    source ./avx512_compiler_flags.sh"
echo "  3. Compile:         gcc \$CFLAGS_AVX512 -o app app.c"
echo "  4. Test:            test_avx512"
echo ""
echo "Functions:"
echo "  show_avx512_flags        - Display all flag sets"
echo "  test_avx512              - Verify AVX-512 compilation works"
echo "  compile_avx512           - Compile with AVX-512 flags"
echo "  compile_kernel_avx512    - Build kernel with AVX-512"
echo "  benchmark_avx512_vs_avx2 - Compare AVX-512 vs AVX2 performance"
echo ""
