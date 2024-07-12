# REQUIRES: riscv
# RUN: rm -rf %t && split-file %s %t && cd %t

# RUN: llvm-mc -filetype=obj -triple=riscv32-unknown-elf -mattr=+relax a.s -o rv32.o
# RUN: llvm-mc -filetype=obj -triple=riscv64-unknown-elf -mattr=+relax a.s -o rv64.o
# RUN: llvm-mc -filetype=obj -triple=riscv64-unknown-elf -mattr=+relax a.s -o rv64-pie.o

# RUN: ld.lld --relax-gp --undefined=__global_pointer$ rv32.o lds -pie -o rv32
# RUN: ld.lld --relax-gp --undefined=__global_pointer$ rv64.o lds -shared -o rv64
# RUN: llvm-objdump -td -M no-aliases --no-show-raw-insn rv32 | FileCheck %s
# RUN: llvm-objdump -td -M no-aliases --no-show-raw-insn rv64 | FileCheck %s

# CHECK:      lui     a0, 0x200
# CHECK-NEXT: addi    a0, a0, 0x1
# CHECK-NEXT: lw      a0, 0x1(a0)
# CHECK-NEXT: sw      a0, 0x1(a0)

#--- a.s
.globl abs
abs = 0x200001

.global _start
_start:
  lui a0, %hi(abs)
  addi a0, a0, %lo(abs)
  lw a0, %lo(abs)(a0)
  sw a0, %lo(abs)(a0)

#--- lds
SECTIONS {
  .text : {*(.text) }
  .sdata 0x200000 : {}
}
