# REQUIRES: riscv
# RUN: rm -rf %t && split-file %s %t && cd %t

# RUN: llvm-mc -filetype=obj -triple=riscv32-unknown-elf -mattr=+relax a.s -o rv32.o
# RUN: llvm-mc -filetype=obj -triple=riscv64-unknown-elf -mattr=+relax a.s -o rv64.o

# RUN: ld.lld --relax-gp --undefined=__global_pointer$ rv32.o lds -o rv32
# RUN: ld.lld --relax-gp --undefined=__global_pointer$ rv64.o lds -o rv64
# RUN: llvm-objdump -td -M no-aliases --no-show-raw-insn rv32 | FileCheck %s
# RUN: llvm-objdump -td -M no-aliases --no-show-raw-insn rv64 | FileCheck %s
 
# CHECK: 00001000 l       .data  {{0+}}08 Var0
# CHECK: 00001f80 l       .data1 {{0+}}80 Var1
# CHECK: 00001800 g       .sdata {{0+}}00 __global_pointer$

# CHECK: <_start>:
# CHECK-NEXT: lui     a1, 1
# CHECK-NEXT: lw      a0, -2048(gp)
# CHECK-NEXT: lw      a1, -2044(gp)
# CHECK-NEXT: lui     a1, 2
# CHECK-NEXT: lw      a0, 1920(gp)
# CHECK-NEXT: lw      a1, 2044(gp)

#--- a.s
.global _start
_start:
        lui     a1, %hi(Var0+4)        # Cannot prove that %lo relocs will be reachable
        lw      a0, %lo(Var0)(a1)      # Reachable from GP
        lw      a1, %lo(Var0+4)(a1)    # Reachable from GP
        lui     a1, %hi(Var1+124)      # Cannot prove that %lo relocs will be reachable
        lw      a0, %lo(Var1)(a1)      # Reachable from GP
        lw      a1, %lo(Var1+124)(a1)  # Reachable from GP

.section .sdata,"aw"
.section .data,"aw"
  .p2align 3
Var0:
  .quad 0
  .size   Var0, 8

.section .data1,"aw"
  .p2align 7
Var1:
  .quad 0
  .zero 120
  .size   Var1, 128

#--- lds
SECTIONS {
  .text : { }
  .sdata 0x1000 : { }
  .data  0x1000 : { }
  .data1 0x1f80 : { }
}
