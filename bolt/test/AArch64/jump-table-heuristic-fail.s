## Verify that BOLT does not crash while encountering instruction sequence that
## does not perfectly match jump table pattern.

# REQUIRES: system-linux

# RUN: llvm-mc -filetype=obj -triple aarch64-unknown-unknown %s -o %t.o
# RUN: %clang %cflags --target=aarch64-unknown-linux %t.o -o %t.exe -Wl,-q
# RUN: llvm-bolt %t.exe -o %t.bolt --print-cfg 2>&1 | FileCheck %s

  .section .text
  .align 4
  .globl _start
  .type  _start, %function
_start:
  sub     w0, w0, #0x4a
## The address loaded into x22 is undefined. However, the instructions that
## follow ldr, use the x22 address as a regular jump table.
  ldr     x22, [x29, #0x98]
  ldrb    w0, [x22, w0, uxtw]
  adr     x1, #12
  add     x0, x1, w0, sxtb #2
  br      x0
# CHECK: br x0 # UNKNOWN
.L0:
  ret
.size _start, .-_start

## Force relocation mode.
  .reloc 0, R_AARCH64_NONE
