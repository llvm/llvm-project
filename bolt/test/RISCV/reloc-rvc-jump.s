// RUN: %clang %cflags64 -o %t %s
// RUN: llvm-bolt --print-cfg --print-only=_start -o %t.null %t \
// RUN:    | FileCheck %s
// RUN: llvm-mc -triple riscv32 -mattr=+c -filetype=obj -o %t.rv32.o %s
// RUN: ld.lld -q -o %t.rv32 %t.rv32.o
// RUN: llvm-bolt --print-cfg --print-only=_start -o %t.rv32.null %t.rv32 \
// RUN:    | FileCheck %s

  .text
  .globl _start
  .p2align 1
// CHECK: Binary Function "_start" after building cfg {
_start:
// CHECK: {{(c.)?}}j .Ltmp0
  c.j 1f
  nop
// CHECK: .Ltmp0
1:
  ret
  .size _start, .-_start
