# This test verifies that BOLT does not crash while rescanning references from
# an ignored function when its branch target lies in another function's constant
# island.

# RUN: llvm-mc -triple riscv64 -mattr=+c -filetype=obj -o %t.o %s
# RUN: ld.lld -q -o %t.exe %t.o
# RUN: llvm-bolt %t.exe -o %t.bolt -reorder-blocks=ext-tsp \
# RUN:   -reorder-functions=cdsort -simplify-rodata-loads -plt=hot \
# RUN:   -split-eh -use-gnu-stack 2>&1 | FileCheck %s

# CHECK: BOLT-WARNING: corrupted control flow detected in function source:
# CHECK-SAME: an external branch/call targets an invalid instruction
# CHECK-SAME: in function target at address 0x{{[0-9a-f]+}}; ignoring both functions
# CHECK: BOLT-WARNING: ignoring entry point at address 0x{{[0-9a-f]+}} in constant island of function target
# CHECK-NOT: cannot add entry point that points to constant data

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

  .globl source
  .type source, @function
source:
  j data_label
  ret
  .size source, .-source

  .globl _start
  .type _start, @function
_start:
  call source
  ret
  .size _start, .-_start

  .reloc 0, R_RISCV_NONE
