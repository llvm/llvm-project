# REQUIRES: riscv

# RUN: llvm-mc -filetype=obj -triple=riscv64 %s -o %t.o
# RUN: not ld.lld %t.o 2>&1 | FileCheck %s

# CHECK: error: {{.*}}:(.text.sec_one+0x0): R_RISCV_PCREL_LO12 relocation points to a symbol '.Lpcrel_hi0' in a different section '.text.sec_two'
# CHECK: error: {{.*}}:(.text.sec_one+0x4): R_RISCV_PCREL_LO12 relocation points to a symbol '.Lpcrel_hi1' in a different section '.text.sec_two'
# CHECK-NOT: R_RISCV_PCREL_LO12 relocation points to a symbol '.Lpcrel_hi2'

## This test is checking that we warn the user when the relocations in their
## object don't follow the RISC-V psABI. In particular, the psABI requires
## that PCREL_LO12 relocations are in the same section as the pcrel_hi
## instruction they point to.

  .section .text.sec_one,"ax"
  addi a0, a0, %pcrel_lo(.Lpcrel_hi0)
  sw a0, %pcrel_lo(.Lpcrel_hi1)(a1)

  .section .text.sec_two,"ax"
.Lpcrel_hi0:
  auipc a0, %pcrel_hi(a)
.Lpcrel_hi1:
  auipc a1, %pcrel_hi(a)

.Lpcrel_hi2:
  auipc a2, %pcrel_hi(a)
  addi a2, a2, %pcrel_lo(.Lpcrel_hi2)

  .data
  .global a
a:
  .word 50
