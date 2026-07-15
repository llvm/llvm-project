## Check that a conditional branch to another function is handled as a
## conditional tail call. The target-specific jump-to-tail-call conversion must
## return false for conditional branches so the generic code records the CTC
## annotation instead of marking the branch as an unconditional tail call.

# RUN: llvm-mc -triple riscv64 -mattr=+c -filetype=obj -o %t.o %s
# RUN: ld.lld --emit-relocs -o %t.exe %t.o
# RUN: llvm-bolt %t.exe -o %t.bolt --print-after-lowering --print-only=_start \
# RUN:   2>&1 | FileCheck %s

# CHECK: Binary Function "_start"
# CHECK: bnez a0, .Ltmp[[#]]
# CHECK: tail callee
# CHECK: End of Function "_start"

  .text
  .globl callee
  .type callee, @function
callee:
  ret
  .size callee, .-callee

  .globl _start
  .type _start, @function
_start:
  beq a0, zero, callee
  ret
  .size _start, .-_start

  .reloc 0, R_RISCV_NONE
