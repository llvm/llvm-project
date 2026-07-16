#!/usr/bin/env bash
# Regenerate comgr-hotswap-entry-trampoline-fast-stub.inc from its .s source.
#
# The B0->B0 entry-trampoline fast path emits its stub from a pre-encoded byte
# template instead of running the MC layer at rewrite time (that is the whole
# point of the fast path). To keep those bytes from silently drifting from what
# the assembler produces -- and to satisfy the "no hand-maintained encoded byte
# sequences" convention -- the template is GENERATED from
# comgr-hotswap-entry-trampoline-fast-stub.s by this script and checked in as a
# .inc. HotswapMCTest's StubTemplateMatchesMCOutput test additionally proves the
# checked-in bytes still equal fresh MC output.
#
# Usage:
#   ./gen-hotswap-fast-stub-inc.sh [path-to-llvm-mc] [path-to-llvm-objcopy]
# Defaults look for llvm-mc / llvm-objcopy on PATH. Run from anywhere; paths are
# resolved relative to this script.
#
# The stub body is spelled with the fixed s[100:101] scratch pair; the runtime
# patches the six SGPR register-field bytes per kernel (see
# comgr-hotswap-internal.h FastEntry*Offset). The two s_add immediates are
# assembled with a non-zero literal to force the 32-bit-literal encoding (imm=0
# would assemble as the shorter inline-constant form), then zeroed here so the
# template carries the delta slots as zero.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
SRC="$HERE/comgr-hotswap-entry-trampoline-fast-stub.s"
OUT="$HERE/../src/comgr-hotswap-entry-trampoline-fast-stub.inc"
MC="${1:-llvm-mc}"
OBJCOPY="${2:-llvm-objcopy}"
TRIPLE="amdgcn-amd-amdhsa"
MCPU="gfx1250"

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

emit_section() { # asm-file section-symbol-name  -> prints "0xNN, 0xNN, ..."
  local asm="$1"
  "$MC" -triple "$TRIPLE" -mcpu="$MCPU" -filetype=obj "$asm" -o "$TMP/o.o"
  "$OBJCOPY" -O binary --only-section=.text "$TMP/o.o" "$TMP/o.bin"
  od -An -tu1 "$TMP/o.bin" | tr -s ' ' '\n' | grep -E '^[0-9]+$'
}

# --- Stub body (40 bytes), imm words forced to the literal form then zeroed. ---
# offsets of the two 32-bit literal imm dwords within the body (see the .s).
STUB_BYTES=($(emit_section "$SRC"))
# Zero the two delta immediates: s_add_co_u32 imm @ 24..27, s_add_co_ci_u32 @ 32..35.
for i in 24 25 26 27 32 33 34 35; do STUB_BYTES[$i]=0; done

printf 's_code_end\n'  > "$TMP/ce.s"; CE_BYTES=($(emit_section "$TMP/ce.s"))
printf 's_nop 0\n'     > "$TMP/nop.s"; NOP_BYTES=($(emit_section "$TMP/nop.s"))

fmt() { # name bytes...
  local name="$1"; shift
  local total=$#
  printf 'static constexpr uint8_t %s[] = {' "$name"
  local n=0
  for b in "$@"; do
    (( n % 12 == 0 )) && printf '\n   '
    printf ' 0x%02x,' "$b"; n=$((n+1))
  done
  printf '\n};\n'
}

{
  # This file is generated; keep clang-format from reflowing the byte rows and
  # the reproducibility comment (the CI code_formatter treats .inc as C++).
  echo "// clang-format off"
  echo "// GENERATED - DO NOT EDIT."
  echo "//"
  echo "// Regenerate with amd/comgr/utils/gen-hotswap-fast-stub-inc.sh from"
  echo "// comgr-hotswap-entry-trampoline-fast-stub.s. Ground-truth encodings from:"
  echo "//   llvm-mc -triple $TRIPLE -mcpu=$MCPU -filetype=obj <src> -o <obj>"
  echo "//   llvm-objcopy -O binary --only-section=.text <obj> <bin>"
  echo "// The two s_add immediates are assembled with a literal to force the"
  echo "// 32-bit-literal encoding, then zeroed (the runtime writes the PC-relative"
  echo "// delta there). The scratch pair is spelled s[100:101]; the runtime"
  echo "// rewrites the six SGPR register-field bytes per kernel."
  echo "//"
  echo "// HotswapMCTest.StubTemplateMatchesMCOutput proves these bytes still equal"
  echo "// fresh MC output, so any assembler drift fails the build's tests."
  echo
  fmt "StubTemplate" "${STUB_BYTES[@]}"
  echo
  fmt "SCodeEnd" "${CE_BYTES[@]}"
  echo
  fmt "SNop" "${NOP_BYTES[@]}"
  echo "// clang-format on"
} > "$OUT"

echo "wrote $OUT"
