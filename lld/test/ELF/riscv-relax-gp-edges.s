# REQUIRES: riscv
# RUN: rm -rf %t && split-file %s %t && cd %t

# RUN: llvm-mc -filetype=obj -triple=riscv32-unknown-elf -mattr=+relax a.s -o rv32a.o
# RUN: llvm-mc -filetype=obj -triple=riscv64-unknown-elf -mattr=+relax a.s -o rv64a.o
# RUN: llvm-mc -filetype=obj -triple=riscv32-unknown-elf -mattr=+relax b.s -o rv32b.o
# RUN: llvm-mc -filetype=obj -triple=riscv64-unknown-elf -mattr=+relax b.s -o rv64b.o
# RUN: llvm-mc -filetype=obj -triple=riscv32-unknown-elf -mattr=+relax c.s -o rv32c.o
# RUN: llvm-mc -filetype=obj -triple=riscv64-unknown-elf -mattr=+relax c.s -o rv64c.o

# RUN: ld.lld --relax-gp --undefined=__global_pointer$ rv32a.o a.lds -o rv32a
# RUN: ld.lld --relax-gp --undefined=__global_pointer$ rv64a.o a.lds -o rv64a
# RUN: ld.lld --relax-gp --undefined=__global_pointer$ rv32b.o b.lds -o rv32b
# RUN: ld.lld --relax-gp --undefined=__global_pointer$ rv64b.o b.lds -o rv64b
# RUN: ld.lld --relax-gp --undefined=__global_pointer$ rv32c.o c.lds -o rv32c
# RUN: ld.lld --relax-gp --undefined=__global_pointer$ rv64c.o c.lds -o rv64c
# RUN: llvm-objdump -td -M no-aliases --no-show-raw-insn --no-print-imm-hex rv32a | FileCheck %s --check-prefix=CHECK-ALIGN
# RUN: llvm-objdump -td -M no-aliases --no-show-raw-insn --no-print-imm-hex rv64a | FileCheck %s --check-prefix=CHECK-ALIGN
# RUN: llvm-objdump -td -M no-aliases --no-show-raw-insn --no-print-imm-hex rv32b | FileCheck %s --check-prefix=CHECK-HI-ADD
# RUN: llvm-objdump -td -M no-aliases --no-show-raw-insn --no-print-imm-hex rv64b | FileCheck %s --check-prefix=CHECK-HI-ADD
# RUN: llvm-objdump -td -M no-aliases --no-show-raw-insn --no-print-imm-hex rv32c | FileCheck %s --check-prefix=CHECK-SIZE
# RUN: llvm-objdump -td -M no-aliases --no-show-raw-insn --no-print-imm-hex rv64c | FileCheck %s --check-prefix=CHECK-SIZE
 
# CHECK-ALIGN: 000017e0 l       .data  {{0+}}80 Var1
# CHECK-ALIGN: 00000ffc g       .sdata {{0+}}00 __global_pointer$

# CHECK-ALIGN: <_start>:
# CHECK-ALIGN-NEXT: lui     a1, 1
# CHECK-ALIGN-NEXT: lw      a0, 2016(a1)
# CHECK-ALIGN-NEXT: lw      a1, 2044(a1)

#--- a.s
# The relaxation on the lui is rejected because the alignment (which is smaller
# than the size) doesn't allow it.
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

#--- a.lds
SECTIONS {
  .text : { }
  .sdata 0x07fc : { }
  .data  0x17E0 : { }
}
 
# CHECK-HI-ADD: 00001000 l       .data  {{0+}}08 Var0
# CHECK-HI-ADD: 00001f80 l       .data1 {{0+}}80 Var1
# CHECK-HI-ADD: 00001800 g       .sdata {{0+}}00 __global_pointer$

# CHECK-HI-ADD: <_start>:
# CHECK-HI-ADD-NEXT: lw      a0, -2048(gp)
# CHECK-HI-ADD-NEXT: lw      a1, -2044(gp)
# CHECK-HI-ADD-NEXT: lw      a0, 1920(gp)
# CHECK-HI-ADD-NEXT: lw      a1, 2044(gp)

#--- b.s
# The relaxation on the two lui are rejected because the amount of data a LO12
# reloc is allowed to address below and above the respective HI20 goes past
# the amount reachable from GP.
.global _start
_start:
        lui     a1, %hi(Var0+4)
        lw      a0, %lo(Var0)(a1)
        lw      a1, %lo(Var0+4)(a1)
        lui     a1, %hi(Var1+124)
        lw      a0, %lo(Var1)(a1)
        lw      a1, %lo(Var1+124)(a1)

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

#--- b.lds
SECTIONS {
  .text : { }
  .sdata 0x1000 : { }
  .data  0x1000 : { }
  .data1 0x1f80 : { }
}

# CHECK-SIZE: 00000080 l       .data  {{0+}}08 Var0
# CHECK-SIZE: 00001000 l       .data1 {{0+}}80 Var1
# CHECK-SIZE: 00000815 g       .sdata {{0+}}00 __global_pointer$

# CHECK-SIZE: <_start>:
# CHECK-SIZE-NOT:  lui
# CHECK-SIZE-NEXT: lw      a0, -1941(gp)
# CHECK-SIZE-NEXT: lw      a1, -1937(gp)
# CHECK-SIZE-NEXT: lui     a1, 1
# CHECK-SIZE-NEXT: lw      a0, 0(a1)
# CHECK-SIZE-NEXT: lw      a1, 124(a1)

#--- c.s
# The relaxation on the second lui is rejected because the size (and alignment)
# allow for a LO12 that cannot reach its target from GP.
.global _start
_start:
        lui     a1, %hi(Var0)
        lw      a0, %lo(Var0)(a1)
        lw      a1, %lo(Var0+4)(a1)
        lui     a1, %hi(Var1)
        lw      a0, %lo(Var1)(a1)      # First part is reachable from gp
        lw      a1, %lo(Var1+124)(a1)  # The second part is not reachable

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

#--- c.lds
SECTIONS {
  .text : { }
  .sdata  0x0015 : { }
  .data   0x0080 : { }
  .data1  0x1000 : { }
}
