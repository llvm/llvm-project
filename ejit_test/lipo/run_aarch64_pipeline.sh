#!/bin/bash
# EJIT aarch64 ejit.o pipeline — native build with clang, no bare-metal trimming.
# Run from the llvm-project root, after building:
#   ninja -C build_release_aarch64 clang LLVMEJIT lld
#
# Uses the build directory's own clang/clang++ and ld.lld (native aarch64).
# No --exclude needed; the resulting ejit.o is ~37 MB (larger than the
# bare-metal version, but includes full LLVM functionality).

set -euo pipefail

BUILD_DIR="${1:-build_release_aarch64}"
OUTPUT="${2:-ejit_test/lipo/ejit.o}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORK_DIR="$SCRIPT_DIR/.lipo_work_aarch64"

rm -rf "$WORK_DIR"

# ── Step 1: extract ───────────────────────────────────────────────────────────
# Compiles a reference binary, parses the linker map, and uses nm -u dependency
# tracing to extract only the .o files actually needed by EJIT.
python3 "$SCRIPT_DIR/lipo.py" extract \
  --arch=aarch64 --build-dir="$BUILD_DIR"

# ── Step 2: gc-merge ─────────────────────────────────────────────────────────
# ld -r --gc-sections: eliminates unreferenced sections from the merged .o,
# using EJIT API entry points as gc roots.
python3 "$SCRIPT_DIR/lipo.py" gc-merge \
  --input="$SCRIPT_DIR/libejit_lipo_aarch64.a" \
  --build-dir="$BUILD_DIR"

# ── Step 3: merge ─────────────────────────────────────────────────────────────
# ld -r -T merge.ld: merges per-function sections into single .text/.rodata/.data
# sections, discards .group metadata, producing the final ejit.o.
python3 "$SCRIPT_DIR/lipo.py" merge \
  --input="$SCRIPT_DIR/libejit_lipo_aarch64_gc.a" \
  --build-dir="$BUILD_DIR" \
  --output="$OUTPUT"

echo ""
echo "Done: $(ls -lh "$OUTPUT" | awk '{print $5, $NF}')"
size -A "$OUTPUT" 2>/dev/null | head -12
