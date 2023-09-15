// RUN: %clang %cflags -o %t %s
// RUN: llvm-bolt --print-fix-riscv-calls --print-only=_start -o /dev/null %t \
// RUN:    | FileCheck %s

  .text

  .global f
  .p2align 1
f:
  ret
  .size f, .-f

// CHECK: Binary Function "_start" after fix-riscv-calls {
  .globl _start
  .p2align 1
_start:
// CHECK: call f
  call f
  ret
  .size _start, .-_start
