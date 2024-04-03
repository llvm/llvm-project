// RUN: %clang %cflags -o %t %s
// RUN: llvm-bolt --print-cfg --print-only=_start -o %t.null %t \
// RUN:    | FileCheck %s

  .text
  .globl _start
  .p2align 1
// CHECK: Binary Function "_start" after building cfg {
_start:
// CHECK: j .Ltmp0
  c.j 1f
  nop
// CHECK: .Ltmp0
1:
  ret
  .size _start, .-_start
