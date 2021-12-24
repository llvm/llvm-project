// REQUIRES: aarch64
// RUN: llvm-mc -filetype=obj -triple=aarch64-none-freebsd %s -o %t.o
// RUN: not ld.lld -shared %t.o -o /dev/null 2>&1 | FileCheck %s
// CHECK: error: relocation R_AARCH64_LDST8_ABS_LO12_NC cannot be used against symbol 'dat'; recompile with -fPIC
// CHECK: >>> defined in {{.*}}.o
// CHECK: >>> referenced by {{.*}}.o:(.text+0x0)

  ldrsb x0, [x1, :lo12:dat]
.data
.globl dat
dat:
  .word 0
