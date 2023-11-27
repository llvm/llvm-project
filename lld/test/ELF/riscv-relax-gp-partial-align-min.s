# REQUIRES: riscv
# RUN: rm -rf %t && split-file %s %t && cd %t

# RUN: llvm-mc -filetype=obj -triple=riscv32-unknown-elf -mattr=+relax a.s -o rv32.o
# RUN: llvm-mc -filetype=obj -triple=riscv64-unknown-elf -mattr=+relax a.s -o rv64.o

# RUN: ld.lld --relax-gp --undefined=__global_pointer$ rv32.o lds -o rv32
# RUN: ld.lld --relax-gp --undefined=__global_pointer$ rv64.o lds -o rv64
# RUN: llvm-objdump -td -M no-aliases --no-show-raw-insn rv32 | FileCheck %s
# RUN: llvm-objdump -td -M no-aliases --no-show-raw-insn rv64 | FileCheck %s
 
# CHECK: 000017e0 l       .data  {{0+}}80 Var1
# CHECK: 00000ffc g       .sdata {{0+}}00 __global_pointer$

# CHECK: <_start>:
# CHECK-NEXT: lui     a1, 1
# CHECK-NEXT: lw      a0, 2020(gp)
# CHECK-NEXT: lw      a1, 2044(a1)

#--- a.s
.global _start
_start:
        lui     a1, %hi(Var1)
        lw      a0, %lo(Var1)(a1)      # First part is reachable from gp
        lw      a1, %lo(Var1+28)(a1)   # The second part is not reachable

.section .sdata,"aw"
.section .data,"aw"
  .p2align 5
Var1:
  .quad 0
  .zero 120
  .size   Var1, 128

#--- lds
SECTIONS {
  .text : { }
  .sdata 0x07fc : { }
  .data  0x17E0 : { }
}
