// REQUIRES: aarch64
// RUN: llvm-mc -filetype=obj -triple=aarch64-none-freebsd %s -o %t.o
// RUN: not ld.lld -shared %t.o -o %t.so 2>&1 | FileCheck %s
// CHECK: {{.*}}.o:(.text+0x0): can't create dynamic relocation R_AARCH64_ADD_ABS_LO12_NC against symbol 'dat' defined in {{.*}}.o

  add x0, x0, :lo12:dat
.data
.globl dat
dat:
  .word 0
