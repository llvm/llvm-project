## Verify that llvm-bolt correctly prints relocation types.

# REQUIRES: system-linux

# RUN: %clang %cflags -nostartfiles %s -o %t.exe -Wl,-q,--no-relax
# RUN: llvm-bolt %t.exe --print-cfg --print-relocations -o %t.bolt \
# RUN:   | FileCheck %s

  .section .text
  .align 4
  .globl _start
  .type _start, %function
_start:

  adrp x0, _start
# CHECK: adrp
# CHECK-SAME: R_AARCH64_ADR_PREL_PG_HI21

  add x0, x0, :lo12:_start
# CHECK-NEXT: add
# CHECK-SAME: R_AARCH64_ADD_ABS_LO12_NC

  ret
  .size _start, .-_start
