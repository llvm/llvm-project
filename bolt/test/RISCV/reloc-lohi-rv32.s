/// RV32 variant of reloc-lohi.s — uses lw/sw instead of ld/sd.

// RUN: llvm-mc -triple riscv32 -filetype=obj -o %t.o %s
// RUN: ld.lld -q -o %t %t.o
// RUN: llvm-bolt --print-cfg --print-only=_start -o %t.null %t \
// RUN:    | FileCheck %s

  .data
  .globl d
  .p2align 2
d:
  .word 0

// CHECK-LABEL: Binary Function "_start" after building cfg {
// CHECK:      lui t0, %hi(d)
// CHECK-NEXT: lw t0, %lo(d)(t0)
// CHECK-NEXT: lui t0, %hi(d)
// CHECK-NEXT: sw t0, %lo(d)(t0)
  .text
  .globl _start
  .p2align 1
_start:
  lui t0, %hi(d)
  lw t0, %lo(d)(t0)
  lui t0, %hi(d)
  sw t0, %lo(d)(t0)
  ret
  .size _start, .-_start
