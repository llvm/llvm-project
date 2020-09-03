# REQUIRES: riscv

# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+relax %s -o %t.o
# RUN: not ld.lld %t.o -o /dev/null 2>&1 | FileCheck %s

# CHECK: relocation R_RISCV_ALIGN requires unimplemented linker relaxation

.global _start
_start:
    nop
    .balign 8
    nop
