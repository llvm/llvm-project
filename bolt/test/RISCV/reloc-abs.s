// RUN: %clang %cflags -Wl,--defsym='__global_pointer$'=0x2800 -o %t %s
// RUN: llvm-bolt --print-cfg --print-only=_start -o /dev/null %t \
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
// CHECK: auipc gp, %pcrel_hi(__global_pointer$) # Label: .Ltmp0
// CHECK-NEXT: addi gp, gp, %pcrel_lo(.Ltmp0)
  auipc gp, %pcrel_hi(__global_pointer$)
  addi  gp, gp, %pcrel_lo(1b)
  .option pop
  .size _start, .-_start
