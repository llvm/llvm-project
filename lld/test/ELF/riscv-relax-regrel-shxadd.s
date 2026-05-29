# REQUIRES: riscv
# RUN: rm -rf %t && split-file %s %t && cd %t

# RUN: llvm-mc -filetype=obj -triple=riscv32-unknown-elf -mattr=+relax,+zba a.s -o rv32.o
# RUN: llvm-mc -filetype=obj -triple=riscv64-unknown-elf -mattr=+relax,+zba a.s -o rv64.o

# RUN: ld.lld --relax-gp --undefined=__global_pointer$ rv32.o lds -o rv32
# RUN: ld.lld --relax-gp --undefined=__global_pointer$ rv64.o lds -o rv64
# RUN: llvm-objdump --mattr=+zba -td -M no-aliases --no-show-raw-insn rv32 | FileCheck %s
# RUN: llvm-objdump --mattr=+zba -td -M no-aliases --no-show-raw-insn rv64 | FileCheck %s

# CHECK: 00000000 l       .text {{0*}}0 $x

# CHECK-NOT:  lui
# CHECK:      addi    a1, a1, -0x800
# CHECK-NEXT: sh1add  a0, a0, gp
# CHECK-NEXT: lw      a0, -0x800(a0)
# CHECK-NEXT: sw      a0, -0x800(a0)
# CHECK-NOT:  lui
# CHECK-NEXT: addi    a1, a1, 0x7fa
# CHECK-NEXT: sh1add  a0, a0, gp
# CHECK-NEXT: lw      a0, 0x7fa(a0)
# CHECK-NEXT: sw      a0, 0x7fa(a0)
# CHECK-NEXT: lui     a1, 0x201
# CHECK-NEXT: addi    a1, a1, 0xe
# CHECK-NEXT: sh1add  a0, a0, a1
# CHECK-NEXT: lw      a0, 0xe(a0)
# CHECK-NEXT: sw      a0, 0xe(a0)
# CHECK-EMPTY:
# CHECK-NEXT: <a>:
# CHECK-NEXT: addi a0, a0, 0x1

#--- a.s
.global _start
_start:
  lui  a1, %hi(array)
  addi a1, a1, %regrel_lo(array)
  sh1add  a0, a0, a1, %regrel_add(array)
  lw   a0, %regrel_lo(array)(a0)
  sw   a0, %regrel_lo(array)(a0)
  lui  a1, %hi(array1+10)
  addi a1, a1, %regrel_lo(array1+10)
  sh1add  a0, a0, a1, %regrel_add(array1+10)
  lw   a0, %regrel_lo(array1+10)(a0)
  sw   a0, %regrel_lo(array1+10)(a0)
  lui  a1, %hi(norelax+10)
  addi a1, a1, %regrel_lo(norelax+10)
  sh1add  a0, a0, a1, %regrel_add(norelax+10)
  lw   a0, %regrel_lo(norelax+10)(a0)
  sw   a0, %regrel_lo(norelax+10)(a0)
a:
  addi a0, a0, 1

.section .sdata,"aw"
array:
  .zero   4080
  .size   array, 4080
array1:
  .zero   20
  .size   array, 20
norelax:
  .zero   6
  .size   array, 6

#--- lds
SECTIONS {
  .text : {*(.text) }
  .sdata 0x200000 : { }
}

