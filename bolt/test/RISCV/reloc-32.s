/// RV32 counterpart of reloc-64.s — tests R_RISCV_32 relocation parsing.
/// Verifies BOLT can read an RV32 ELF with R_RISCV_32 data relocations.

// RUN: llvm-mc -triple riscv32 -filetype=obj -o %t.o %s
// RUN: ld.lld -q -o %t %t.o
// RUN: llvm-bolt -o %t.bolt %t 2>&1 | FileCheck %s

// CHECK: BOLT-INFO: Target architecture: riscv32
// CHECK: BOLT-INFO: enabling relocation mode

  .data
  .globl d
  .p2align 2
d:
  .word _start

  .text
  .globl _start
  .p2align 1
_start:
  ret
  .reloc 0, R_RISCV_NONE
  .size _start, .-_start
