// RUN: llvm-mc -triple riscv64 -filetype=obj -o %t.o %s
// RUN: ld.lld -q -o %t %t.o
// RUN: llvm-bolt --print-cfg --print-only=_start -o /dev/null %t \
// RUN:    | FileCheck %s

  .data
  .globl d
  .p2align 3
d:
  .dword 0

// CHECK-LABEL: Binary Function "_start" after building cfg {
// CHECK:      lui t0, %hi(d)
// CHECK-NEXT: ld t0, %lo(d)(t0)
// CHECK-NEXT: lui t0, %hi(d)
// CHECK-NEXT: sd t0, %lo(d)(t0)
  .text
  .globl _start
  .p2align 1
_start:
  lui t0, %hi(d)
  ld t0, %lo(d)(t0)
  lui t0, %hi(d)
  sd t0, %lo(d)(t0)
  ret
  .size _start, .-_start
