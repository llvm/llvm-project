// REQUIRES: aarch64
// RUN: llvm-mc -filetype=obj -triple=aarch64-none-freebsd %s -o %t.o
// RUN: not ld.lld -shared %t.o -o %t.so 2>&1 | FileCheck %s
// CHECK: can't create dynamic relocation R_AARCH64_LDST32_ABS_LO12_NC against symbol dat

  ldr s4, [x0, :lo12:dat]
.data
.globl dat
dat:
  .word 0
