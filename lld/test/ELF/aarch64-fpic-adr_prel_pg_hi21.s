// REQUIRES: aarch64
// RUN: llvm-mc -filetype=obj -triple=aarch64-none-freebsd %s -o %t.o
// RUN: not ld.lld -shared %t.o -o %t.so 2>&1 | FileCheck %s
// CHECK: Relocation R_AARCH64_ADR_PREL_PG_HI21 cannot be used when making a shared object; recompile with -fPIC.

  adrp x0, dat
.data
.globl dat
dat:
  .word 0
