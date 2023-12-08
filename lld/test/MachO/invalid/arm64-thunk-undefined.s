# REQUIRES: aarch64

# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %s -o %t.o
## This shouldn't assert.
# RUN: not %lld -arch arm64 -lSystem -o %t/thunk %t.o 2>&1 | FileCheck %s

# CHECK: error: undefined symbol: _g

.subsections_via_symbols

.p2align 2

.globl _main, _g

.globl _main
_main:
  bl _g
  ret

_filler1:
.space 0x4000000

_filler2:
.space 0x4000000
