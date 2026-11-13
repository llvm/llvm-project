# REQUIRES: aarch64
## arm64e names the stubs section __auth_stubs rather than __stubs. Like
## __stubs, it must come directly after __text: needsThunks() assumes that when
## it estimates the reach of branches to the stubs.

# RUN: llvm-mc -filetype=obj -triple=arm64e-apple-darwin %s -o %t.o
# RUN: %lld -arch arm64e -lSystem -U _external_func -o %t.out %t.o
# RUN: llvm-objdump --section-headers %t.out | FileCheck %s

## __auth_stubs must come directly after __text, before any read-only data.
# CHECK:      __text
# CHECK-NEXT: __auth_stubs

## Same placement for plain arm64, where the section is called __stubs.
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %s -o %t64.o
# RUN: %lld -arch arm64 -lSystem -U _external_func -o %t64.out %t64.o
# RUN: llvm-objdump --section-headers %t64.out | FileCheck --check-prefix=ARM64 %s

# ARM64:      __text
# ARM64-NEXT: __stubs

.globl _main
.p2align 2
_main:
  stp x29, x30, [sp, #-16]!
  bl _external_func
  ldp x29, x30, [sp], #16
  ret

.section __TEXT,__const
.p2align 4
_rodata:
  .space 64

.subsections_via_symbols
