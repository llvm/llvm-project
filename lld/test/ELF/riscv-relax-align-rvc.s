# REQUIRES: riscv

# RUN: rm -rf %t && mkdir %t && cd %t

# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+c,+relax %s -o 32.o
# RUN: ld.lld -Ttext=0x10000 32.o -o 32
# RUN: llvm-objdump -td --no-show-raw-insn -M no-aliases 32 | FileCheck %s
## R_RISCV_ALIGN is handled regarldess of --no-relax.
# RUN: ld.lld -Ttext=0x10000 --no-relax 32.o -o 32.norelax
# RUN: llvm-objdump -td --no-show-raw-insn -M no-aliases 32.norelax | FileCheck %s

# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+c,+relax %s -o 64.o
# RUN: ld.lld -Ttext=0x10000 64.o -o 64
# RUN: llvm-objdump -td --no-show-raw-insn -M no-aliases 64 | FileCheck %s
# RUN: ld.lld -Ttext=0x10000 --no-relax 64.o -o 64.norelax
# RUN: llvm-objdump -td --no-show-raw-insn -M no-aliases 64.norelax | FileCheck %s

# CHECK-DAG: 00010002 l       .text  {{0*}}1e a
# CHECK-DAG: 00010010 l       .text  {{0*}}22 b
# CHECK-DAG: 00010012 l       .text  {{0*}}1e c
# CHECK-DAG: 00010020 l       .text  {{0*}}16 d
# CHECK-DAG: 00010000 g       .text  {{0*}}36 _start

# CHECK:      <_start>:
# CHECK-NEXT:           c.addi    a0, 0x1
# CHECK-EMPTY:
# CHECK-NEXT: <a>:
# CHECK-NEXT:           c.nop
# CHECK-NEXT:           addi    zero, zero, 0x0
# CHECK-NEXT:           addi    zero, zero, 0x0
# CHECK-NEXT:           addi    zero, zero, 0x0
# CHECK-EMPTY:
# CHECK-NEXT: <b>:
# CHECK-NEXT:   10010:  c.addi  a0, 0x2
# CHECK-EMPTY:
# CHECK-NEXT: <c>:
# CHECK-NEXT:           c.addi  a0, 0x3
# CHECK-NEXT:           addi    zero, zero, 0x0
# CHECK-NEXT:           addi    zero, zero, 0x0
# CHECK-NEXT:           addi    zero, zero, 0x0
# CHECK-EMPTY:
# CHECK-NEXT: <d>:
# CHECK-NEXT:   10020:  c.addi  a0, 0x4
# CHECK-NEXT:           c.addi  a0, 0x5
# CHECK-NEXT:           addi    zero, zero, 0x0
# CHECK-NEXT:           addi    zero, zero, 0x0
# CHECK-NEXT:           addi    zero, zero, 0x0
# CHECK-NEXT:   10030:  c.addi  a0, 0x6
# CHECK-NEXT:           c.addi  a0, 0x7
# CHECK-NEXT:           c.addi  a0, 0x8
# CHECK-EMPTY:

# CHECK:      <.text2>:
# CHECK-NEXT:           addi    a0, a1, 0x1
# CHECK-NEXT:           c.addi  a0, 0x1
# CHECK-NEXT:           c.nop
# CHECK-NEXT:           c.addi  a0, 0x2

.global _start
_start:
  c.addi a0, 0x1
a:
.balign 16
b:
  c.addi a0, 0x2
c:
  c.addi a0, 0x3
.balign 32
.size a, . - a
d:
  c.addi a0, 0x4
  c.addi a0, 0x5
.balign 16
.size c, . - c
  c.addi a0, 0x6
.size b, . - b
  c.addi a0, 0x7
.balign 4
  c.addi a0, 0x8
.size d, . - d
.size _start, . - _start

.section .text2,"ax"
.balign 16
  addi a0, a1, 0x1
  c.addi a0, 0x1
.balign 8
  c.addi a0, 0x2
