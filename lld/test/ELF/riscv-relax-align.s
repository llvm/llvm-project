# REQUIRES: riscv
## Test that we can handle R_RISCV_ALIGN.

# RUN: rm -rf %t && mkdir %t && cd %t

# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+relax %s -o 32.o
# RUN: ld.lld -Ttext=0x10000 32.o -o 32
# RUN: llvm-objdump -td --no-show-raw-insn -M no-aliases 32 | FileCheck %s
## R_RISCV_ALIGN is handled regarldess of --no-relax.
# RUN: ld.lld -Ttext=0x10000 --no-relax 32.o -o 32.norelax
# RUN: llvm-objdump -td --no-show-raw-insn -M no-aliases 32.norelax | FileCheck %s

# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+relax %s -o 64.o
# RUN: ld.lld -Ttext=0x10000 64.o -o 64
# RUN: llvm-objdump -td --no-show-raw-insn -M no-aliases 64 | FileCheck %s
# RUN: ld.lld -Ttext=0x10000 --no-relax 64.o -o 64.norelax
# RUN: llvm-objdump -td --no-show-raw-insn -M no-aliases 64.norelax | FileCheck %s

# RUN: ld.lld -Ttext=0x10000 --gc-sections 64.o -o 64.gc
# RUN: llvm-objdump -td --no-show-raw-insn -M no-aliases 64.gc | FileCheck %s --check-prefix=GC

## -r keeps section contents unchanged.
# RUN: ld.lld -r 64.o -o 64.r
# RUN: llvm-objdump -dr --no-show-raw-insn -M no-aliases 64.r | FileCheck %s --check-prefix=CHECKR

# CHECK-DAG: 00010004 l       .text  {{0*}}1c a
# CHECK-DAG: 00010008 l       .text  {{0*}}28 b
# CHECK-DAG: 00010014 l       .text  {{0*}}20 c
# CHECK-DAG: 00010000 g       .text  {{0*}}38 _start

# CHECK:       <_start>:
# CHECK-NEXT:            addi    a0, a0, 0x1
# CHECK-EMPTY:
# CHECK-NEXT:  <a>:
# CHECK-NEXT:            addi    a0, a0, 0x2
# CHECK-EMPTY:
# CHECK-NEXT:  <b>:
# CHECK-NEXT:            addi    zero, zero, 0x0
# CHECK-NEXT:            addi    zero, zero, 0x0
# CHECK-NEXT:    10010:  addi    a0, a0, 0x3
# CHECK-EMPTY:
# CHECK-NEXT:  <c>:
# CHECK-NEXT:            addi    a0, a0, 0x4
# CHECK-NEXT:            addi    a0, a0, 0x5
# CHECK-NEXT:            addi    zero, zero, 0x0
# CHECK-NEXT:    10020:  addi    a0, a0, 0x6
# CHECK-NEXT:            addi    a0, a0, 0x7
# CHECK-NEXT:            addi    zero, zero, 0x0
# CHECK-NEXT:            addi    zero, zero, 0x0
# CHECK-NEXT:    10030:  addi    a0, a0, 0x8
# CHECK-NEXT:            addi    a0, a0, 0x9
# CHECK-EMPTY:
# CHECK:       <e>:
# CHECK-NEXT:            addi    a0, a0, 0x1
# CHECK-EMPTY:
# CHECK-NEXT:  <f>:
# CHECK-NEXT:    10044:  addi    a0, a0, 0x2
# CHECK-NEXT:            addi    zero, zero, 0x0
# CHECK-NEXT:            addi    zero, zero, 0x0
# CHECK-NEXT:            addi    zero, zero, 0x0
# CHECK-NEXT:            addi    zero, zero, 0x0
# CHECK-NEXT:            addi    zero, zero, 0x0
# CHECK-NEXT:            addi    zero, zero, 0x0
# CHECK-NEXT:    10060:  addi    a0, a0, 0x3
# CHECK-EMPTY:

## _start-0x10070 = 0x10000-0x10070 = -112
# CHECK:      <.L1>:
# CHECK-NEXT:   10070:  auipc   a0, 0x0
# CHECK-NEXT:           addi    a0, a0, -0x70
# CHECK-NEXT:           addi    zero, zero, 0x0
# CHECK-NEXT:           addi    zero, zero, 0x0
# CHECK-NEXT:           auipc   a0, 0x0
# CHECK-NEXT:           addi    a0, a0, -0x70
# CHECK-EMPTY:

# GC-DAG:       00010004 l       .text  {{0*}}1c a
# GC-DAG:       00010008 l       .text  {{0*}}28 b
# GC-DAG:       00010014 l       .text  {{0*}}20 c
# GC-DAG:       00010000 g       .text  {{0*}}38 _start
# GC:           <_start>:
# GC-NOT:       <d>:

# CHECKR:       <_start>:
# CHECKR-NEXT:          addi    a0, a0, 0x1
# CHECKR-EMPTY:
# CHECKR-NEXT:  <a>:
# CHECKR-NEXT:          addi    a0, a0, 0x2
# CHECKR-EMPTY:
# CHECKR-NEXT:  <b>:
# CHECKR-NEXT:          addi    zero, zero, 0x0
# CHECKR-NEXT:          0000000000000008:  R_RISCV_ALIGN        *ABS*+0xc
# CHECKR-NEXT:          addi    zero, zero, 0x0
# CHECKR-NEXT:          addi    zero, zero, 0x0
# CHECKR-NEXT:          addi    a0, a0, 0x3
# CHECKR-EMPTY:
# CHECKR-NEXT:  <c>:
# CHECKR-NEXT:          addi    a0, a0, 0x4
# CHECKR-NEXT:          addi    a0, a0, 0x5
# CHECKR-NEXT:          addi    zero, zero, 0x0
# CHECKR-NEXT:          0000000000000020:  R_RISCV_ALIGN        *ABS*+0x1c
# CHECKR-NEXT:          addi    zero, zero, 0x0
# CHECKR-NEXT:          addi    zero, zero, 0x0
# CHECKR-NEXT:          addi    zero, zero, 0x0
# CHECKR-NEXT:          addi    zero, zero, 0x0
# CHECKR-NEXT:          addi    zero, zero, 0x0
# CHECKR-NEXT:          addi    zero, zero, 0x0
# CHECKR-NEXT:          addi    a0, a0, 0x6
# CHECKR-NEXT:          addi    a0, a0, 0x7
# CHECKR-NEXT:          addi    zero, zero, 0x0
# CHECKR-NEXT:          0000000000000044:  R_RISCV_ALIGN        *ABS*+0xc
# CHECKR-NEXT:          addi    zero, zero, 0x0
# CHECKR-NEXT:          addi    zero, zero, 0x0
# CHECKR-NEXT:          addi    a0, a0, 0x8
# CHECKR-NEXT:          addi    a0, a0, 0x9

.global _start
_start:
  addi a0, a0, 0x1
a:
  addi a0, a0, 0x2
b:
.balign 16
  addi a0, a0, 0x3
c:
  addi a0, a0, 0x4
  addi a0, a0, 0x5
.balign 32
.size a, . - a
  addi a0, a0, 0x6
  addi a0, a0, 0x7
.balign 16
.size b, . - b
  addi a0, a0, 0x8
.size c, . - c
  addi a0, a0, 0x9
.size _start, . - _start

## Test another text section.
.section .text2,"ax",@progbits
d:
e:
.balign 8
  addi a0, a0, 0x1
f:
  addi a0, a0, 0x2
.balign 32
.size d, . - d
  addi a0, a0, 0x3
.size e, . - e
.size f, . - f

## Test that matching HI20 can be found despite deleted bytes.
.section .pcrel,"ax",@progbits
.L1:
  auipc a0, %pcrel_hi(_start)
  addi a0, a0, %pcrel_lo(.L1)
.balign 16
.L2:
  auipc a0, %pcrel_hi(_start)
  addi a0, a0, %pcrel_lo(.L1)
