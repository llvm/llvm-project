# REQUIRES: riscv
# RUN: llvm-mc -filetype=obj -triple=riscv64 %s -o %t.o
# RUN: not ld.lld %t.o --defsym external=0 2>&1 | FileCheck %s

# CHECK: error: R_RISCV_PCREL_LO12 relocation points to an absolute symbol: external

# We provide a dummy %pcrel_hi referred to by external to appease the
# assembler, but make external weak so --defsym can still override it at link
# time.
.weak external
external:
auipc sp,%pcrel_hi(external)
addi sp,sp,%pcrel_lo(external)
