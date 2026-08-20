# This test verifies that rescanning references in an ignored RISC-V function
# does not try to redirect short branch/jump relocations to moved code.

# RUN: llvm-mc -triple riscv64 -mattr=+c -filetype=obj -o %t.o %s
# RUN: ld.lld -q -o %t.exe %t.o
# RUN: llvm-bolt %t.exe -o %t.bolt -reorder-blocks=ext-tsp \
# RUN:   -reorder-functions=cdsort -simplify-rodata-loads -plt=hot \
# RUN:   -split-eh -use-gnu-stack 2>&1 | FileCheck %s

# CHECK: BOLT-WARNING: corrupted control flow detected in function source:
# CHECK: BOLT-WARNING: ignoring entry point at address 0x{{[0-9a-f]+}} in constant island of function target
# CHECK-NOT: unsupported relocation
# CHECK-NOT: could not find corresponding %pcrel_hi
# CHECK-NOT: target out of range

  .text
  .globl target
  .type target, @function
target:
  j after_data

data_label:
  .word 0

after_data:
  ret
  .size target, .-target

  .globl callee
  .type callee, @function
callee:
  nop
  nop
  nop
  nop
  nop
  nop
  nop
  nop
  ret
  .size callee, .-callee

  .globl source
  .type source, @function
source:
  j data_label
  beqz a0, callee
  ret
  .size source, .-source

  .globl _start
  .type _start, @function
_start:
  call source
  ret
  .size _start, .-_start

  .reloc 0, R_RISCV_NONE
