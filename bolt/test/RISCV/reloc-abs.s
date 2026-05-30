// RUN: %clang %cflags64 -Wl,--defsym='__global_pointer$'=0x2800 -o %t %s
// RUN: llvm-bolt --print-cfg --print-only=_start -o %t.null %t \
// RUN:    | FileCheck %s
// RUN: llvm-mc -triple riscv32 -mattr=+c -filetype=obj -o %t.rv32.o %s
// RUN: ld.lld -q --defsym='__global_pointer$'=0x2800 -o %t.rv32 %t.rv32.o
// RUN: llvm-bolt --print-cfg --print-only=_start -o %t.rv32.null %t.rv32 \
// RUN:    | FileCheck %s

  .data
  .globl d
  .p2align 3
d:
  .dword 0

  .text
  .globl _start
  .p2align 1
// CHECK: Binary Function "_start" after building cfg {
_start:
  nop
  .option push
  .option norelax
1:
// CHECK: auipc gp, %pcrel_hi(__global_pointer$)
// CHECK-NEXT: addi gp, gp, %pcrel_lo(
  auipc gp, %pcrel_hi(__global_pointer$)
  addi  gp, gp, %pcrel_lo(1b)
  .option pop
  .size _start, .-_start
