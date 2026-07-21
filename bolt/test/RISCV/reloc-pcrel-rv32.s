/// RV32 variant of reloc-pcrel.s — uses lw/sw instead of ld/sd.

// RUN: llvm-mc -triple riscv32 -mattr=+c -filetype=obj -o %t.o %s
// RUN: ld.lld -q -o %t %t.o
// RUN: llvm-bolt --print-cfg --print-only=_start -o %t.null %t \
// RUN:    | FileCheck %s

  .data
  .globl d
  .p2align 2
d:
  .word 0

  .text
  .globl _start
  .p2align 1
// CHECK: Binary Function "_start" after building cfg {
_start:
  nop // Here to not make the _start and .Ltmp0 symbols coincide
// CHECK: auipc t0, %pcrel_hi(d)
// CHECK-NEXT: lw t0, %pcrel_lo({{.*}})(t0)
  lw t0, d
// CHECK: auipc t1, %pcrel_hi(d)
// CHECK-NEXT: sw t0, %pcrel_lo({{.*}})(t1)
  sw t0, d, t1
  ret
  .size _start, .-_start
