#!/bin/bash
# EJIT aarch64 ejit.o pipeline — native build with clang, no bare-metal trimming.
# Run from the llvm-project root, after building:
#   ninja -C build_release_aarch64 clang LLVMEJIT lld
#
# Uses the build directory's own clang/clang++ and ld.lld (native aarch64).
#
# Options:
#   --keep CPUNAME   Strip all per-CPU scheduling models except CPUNAME.
#                    CPUNAME must be an LLVM CPU name present in
#                    AArch64SubTypeKV (e.g. "cortex-a57", "neoverse-n2").
#   --keep-none      Strip scheduling models for all CPUs.
#                    Without --keep / --keep-none, no stripping is done (default).
#
# Default behaviour (no --keep):
#   If AArch64GenSubtargetInfo.inc was previously patched (contains the EJIT
#   sentinel), it is regenerated from TableGen sources (llvm-tblgen) and the
#   libraries are rebuilt before running the lipo pipeline.
#
# Positional args (both optional):
#   $1  BUILD_DIR   (default: build_release_aarch64)
#   $2  OUTPUT      (default: ejit_test/lipo/ejit.o)

set -euo pipefail

# ── Parse options ─────────────────────────────────────────────────────────────
DO_STRIP=0
KEEP_MODEL=""
POSITIONAL=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --keep)
      DO_STRIP=1
      KEEP_MODEL="$2"
      shift 2
      ;;
    --keep-none)
      DO_STRIP=1
      KEEP_MODEL=""
      shift
      ;;
    *)
      POSITIONAL+=("$1")
      shift
      ;;
  esac
done

BUILD_DIR="${POSITIONAL[0]:-build_release_aarch64}"
OUTPUT="${POSITIONAL[1]:-ejit_test/lipo/ejit.o}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORK_DIR="$SCRIPT_DIR/.lipo_work_aarch64"

INC="$BUILD_DIR/lib/Target/AArch64/AArch64GenSubtargetInfo.inc"

rm -rf "$WORK_DIR"

# Regenerate AArch64GenSubtargetInfo.inc from TableGen sources.
# Deleting the file forces ninja to re-run llvm-tblgen even if AArch64.td
# has not changed (a patched .inc would otherwise look up-to-date by mtime).
regen_inc() {
  echo "── Regenerating AArch64GenSubtargetInfo.inc (llvm-tblgen -gen-subtarget) ──"
  rm -f "$INC"
  ninja -C "$BUILD_DIR" AArch64CommonTableGen
}

# ── Step 0: manage per-CPU scheduling model tables ───────────────────────────
# AArch64GenSubtargetInfo.inc defines ~24 per-CPU MCSchedModel structs (~600 KB
# of .rodata total).  They survive gc-sections because AArch64SubTypeKV holds a
# pointer to every model struct.
#
# With --keep CPUNAME: patch the .inc, nulling all models except CPUNAME's.
#   gc-sections in Step 2 then eliminates the unused arrays (~575 KB saved).
# With --keep-none: null every model (~600 KB saved).
# Without --keep / --keep-none (default): use the full unstripped tablegen.
#   If the .inc was previously patched it is regenerated from source first.
if [[ $DO_STRIP -eq 1 ]]; then
  # If the .inc is already patched from a previous run, regenerate to get a
  # clean slate before applying the new patch.
  if grep -q "EJIT: stripped" "$INC" 2>/dev/null; then
    regen_inc
  fi
  if [[ -n "$KEEP_MODEL" ]]; then
    echo "── Step 0: stripping sched models (keeping ${KEEP_MODEL}) ──"
  else
    echo "── Step 0: stripping all sched models (--keep-none) ──"
  fi
  python3 "$SCRIPT_DIR/ejit_strip_sched_models.py" "$BUILD_DIR" "$KEEP_MODEL"
  echo "── Rebuilding LLVMAArch64CodeGen + LLVMAArch64Desc ──"
  ninja -C "$BUILD_DIR" LLVMAArch64CodeGen LLVMAArch64Desc
else
  # No-strip: if the .inc is currently patched, regenerate from source.
  if grep -q "EJIT: stripped" "$INC" 2>/dev/null; then
    echo "── Step 0: .inc is patched — regenerating from TableGen sources ──"
    regen_inc
    echo "── Rebuilding LLVMAArch64CodeGen + LLVMAArch64Desc ──"
    ninja -C "$BUILD_DIR" LLVMAArch64CodeGen LLVMAArch64Desc
  else
    echo "── Step 0: no-strip, .inc is already original ──"
  fi
fi
echo ""

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
