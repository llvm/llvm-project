// RUN: %clang %cflags -o %t %s
// RUN: llvm-bolt --print-cfg --print-only=_start -o %t.null %t \
// RUN:    | FileCheck %s

  .text

  .global f
  .p2align 1
f:
  ret
  .size f, .-f

// CHECK: Binary Function "_start" after building cfg {
  .globl _start
  .p2align 1
_start:
// CHECK: jal f
  jal ra, f
  ret
  .size _start, .-_start
