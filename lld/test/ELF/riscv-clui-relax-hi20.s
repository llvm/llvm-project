# REQUIRES: riscv
# RUN: rm -rf %t && split-file %s %t && cd %t

# RUN: llvm-mc -filetype=obj -triple=riscv32-unknown-elf -mattr=+relax,+c a.s -o rv32.o
# RUN: llvm-mc -filetype=obj -triple=riscv64-unknown-elf -mattr=+relax,+c a.s -o rv64.o

# RUN: ld.lld --relax-gp --undefined=__global_pointer$ rv32.o lds -o rv32
# RUN: ld.lld rv32.o lds -o rv32_nogp
# RUN: ld.lld --relax-gp --undefined=__global_pointer$ rv64.o lds -o rv64
# RUN: llvm-objdump -td -M no-aliases rv32 | FileCheck %s
# RUN: llvm-objdump -td -M no-aliases rv32_nogp | FileCheck %s --check-prefix=NOGP
# RUN: llvm-objdump -td -M no-aliases rv64 | FileCheck %s

# CHECK: 00002004 l       .data  {{0+}} foo
# CHECK: 00020004 l       .far   {{0+}} bar
# CHECK: 00001800 g       .sdata {{0+}} __global_pointer$

# CHECK: <_start>:
# CHECK: 6509          c.lui   a0, 0x2
# CHECK: 00450513      addi    a0, a0, 0x4
# CHECK: 00020537      lui     a0, 0x20
# CHECK: f8450513      addi    a0, a0, -0x7c
# CHECK: 00020537      lui     a0, 0x20
# CHECK: 00450513      addi    a0, a0, 0x4

# NOGP: 00002004 l       .data  {{0+}} foo
# NOGP: 00020004 l       .far   {{0+}} bar
# NOGP-NOT: 00001800 g       .sdata {{0+}} __global_pointer$

# NOGP: <_start>:
# NOGP: 6509          c.lui   a0, 0x2
# NOGP: 00450513      addi    a0, a0, 0x4
# NOGP: 00020537      lui     a0, 0x20
# NOGP: f8450513      addi    a0, a0, -0x7c
# NOGP: 00020537      lui     a0, 0x20
# NOGP: 00450513      addi    a0, a0, 0x4

#--- a.s
.global _start
_start:
  lui a0, %hi(foo)
  addi a0, a0, %lo(foo)
  lui a0, %hi(neg)
  addi a0, a0, %lo(neg)
  lui a0, %hi(bar)
  addi a0, a0, %lo(bar)

.section .sdata,"aw"
  .zero 32
.section .data,"aw"
foo:
  .word 0
.section .out,"aw"
neg:
  .word 0
.section .far,"aw"
bar:
  .word 0

#--- lds
SECTIONS {
  .text : {*(.text) }
  .sdata 0x1000 : { }
  .data 0x2004 : { }
  .out 0x1FF84 : { }
  .far 0x20004 : { }
}
