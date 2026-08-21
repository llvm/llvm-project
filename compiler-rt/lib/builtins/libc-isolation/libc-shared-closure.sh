#!/usr/bin/env bash
# Print the libc headers the compiler-rt libc-backed builtins include, one
# libc-root-relative path per line, sorted. Used to regenerate the manifest and
# to drift-check it in CI. Run from the llvm-project root; override with CXX=.
set -euo pipefail

CXX="${CXX:-clang++}"
LIBC_ROOT="${LIBC_ROOT:-libc}"

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
printf '#include "shared/builtins.h"\n' > "$tmp/probe.cpp"

"$CXX" -std=c++17 \
  -DLIBC_NAMESPACE=__llvm_libc_common_utils \
  -DLIBC_MATH=12 \
  -I "$LIBC_ROOT" -I "$LIBC_ROOT/include" \
  -MM -MG "$tmp/probe.cpp" 2>/dev/null \
  | tr ' \\' '\n\n' \
  | grep -E "^$LIBC_ROOT/" \
  | LIBC_ROOT="$LIBC_ROOT" python3 -c '
import os, sys
libc = os.path.abspath(os.environ["LIBC_ROOT"])
seen = set()
for line in sys.stdin:
    p = line.strip()
    if not p:
        continue
    rel = os.path.relpath(os.path.normpath(os.path.abspath(p)), libc)
    if not rel.startswith(".."):
        seen.add(rel)
for rel in sorted(seen):
    print(rel)
'
