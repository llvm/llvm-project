// REQUIRES: aarch64
// RUN: llvm-mc -filetype=obj -triple=aarch64-none-freebsd %s -o %t.o
// RUN: not ld.lld -shared %t.o -o %t.so 2>&1 | FileCheck %s
// CHECK: Relocation R_AARCH64_LDST64_ABS_LO12_NC cannot be used when making a shared object; recompile with -fPIC.

  ldr x0, [x0, :lo12:dat]
.data
.globl dat
dat:
  .word 0
