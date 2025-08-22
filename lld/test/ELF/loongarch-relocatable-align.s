# REQUIRES: loongarch

## Test LA64.
# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=loongarch64 -mattr=+relax a.s -o a.o
# RUN: llvm-mc -filetype=obj -triple=loongarch64 -mattr=+relax --defsym ELF64=1 b.s -o b.o
# RUN: llvm-mc -filetype=obj -triple=loongarch64 -mattr=+relax b1.s -o b1.o
# RUN: llvm-mc -filetype=obj -triple=loongarch64 -mattr=+relax c.s -o c.o
# RUN: llvm-mc -filetype=obj -triple=loongarch64 d.s -o d.o

## No RELAX. Don't synthesize ALIGN.
# RUN: ld.lld -r b.o d.o -o bd.ro
# RUN: llvm-readelf -r bd.ro | FileCheck %s --check-prefix=NOREL

# NOREL: no relocations

# RUN: ld.lld -r b.o b.o a.o b.o b1.o c.o d.o -o out.ro
# RUN: llvm-objdump -dr --no-show-raw-insn out.ro | FileCheck %s
# RUN: llvm-readelf -r out.ro | FileCheck %s --check-prefix=CHECK-REL

# CHECK:      <b0>:
# CHECK-NEXT:   0:    addi.d $a0, $a1, 1
# CHECK-NEXT:   4:    nop
# CHECK-EMPTY:
# CHECK-NEXT: <b0>:
# CHECK-NEXT:   8:    addi.d $a0, $a1, 1
# CHECK-EMPTY:
# CHECK-NEXT: <_start>:
# CHECK-NEXT:   c:    pcalau12i $a0, 0
# CHECK-NEXT:           000000000000000c:  R_LARCH_PCALA_HI20   .Ltext1_start
# CHECK-NEXT:           000000000000000c:  R_LARCH_RELAX        *ABS*
# CHECK-NEXT:   10:   addi.d $a0, $a0, 0
# CHECK-NEXT:           0000000000000010:  R_LARCH_PCALA_LO12   .Ltext1_start
# CHECK-NEXT:           0000000000000010:  R_LARCH_RELAX        *ABS*
# CHECK-NEXT:   14:   nop
# CHECK-NEXT:           0000000000000014:  R_LARCH_ALIGN        *ABS*+0x4
# CHECK-EMPTY:
# CHECK-NEXT: <b0>:
# CHECK-NEXT:   18:   addi.d $a0, $a1, 1
# CHECK-NEXT:   1c:   nop
# CHECK-NEXT:   20:   nop
# CHECK-NEXT:           0000000000000020:  R_LARCH_ALIGN        *ABS*+0x4
# CHECK-NEXT:   24:   nop
# CHECK-EMPTY:
# CHECK-NEXT: <b1>:
# CHECK-NEXT:   28:   addi.d $a0, $a1, 3
# CHECK-EMPTY:
# CHECK-NEXT: <c0>:
# CHECK-NEXT:   2c:   addi.d $a0, $a1, 4
# CHECK-NEXT:   30:   nop
# CHECK-NEXT:           0000000000000030:  R_LARCH_ALIGN        *ABS*+0x4
# CHECK-EMPTY:
# CHECK-NEXT: <d0>:
# CHECK-NEXT:   34:   addi.d $a0, $a1, 5

# CHECK-REL:  Relocation section '.rela.text' at offset {{.*}} contains 7 entries:
# CHECK-REL:  Relocation section '.rela.text1' at offset {{.*}} contains 5 entries:

## Test LA32.
# RUN: llvm-mc -filetype=obj -triple=loongarch32 -mattr=+relax a.s -o a.32.o
# RUN: llvm-mc -filetype=obj -triple=loongarch32 -mattr=+relax b.s -o b.32.o
# RUN: ld.lld -r a.32.o b.32.o -o out.32.ro
# RUN: ld.lld -Ttext=0x10000 out.32.ro -o out32
# RUN: llvm-objdump -dr --no-show-raw-insn out32 | FileCheck %s --check-prefix=CHECK32

# CHECK32:      <_start>:
# CHECK32-NEXT:   10000:    pcaddi $a0, 4
# CHECK32-NEXT:   10004:    nop
# CHECK32-EMPTY:
# CHECK32-NEXT: <b0>:
# CHECK32-NEXT:   10008:    addi.w $a0, $a1, 1
# CHECK32:      <.Ltext1_start>:
# CHECK32-NEXT:   10010:    pcaddi $a1, 0
# CHECK32-NEXT:   10014:    nop
# CHECK32-NEXT:   10018:    addi.w $a0, $a1, 2

## Test CREL.
# RUN: llvm-mc -filetype=obj -triple=loongarch64 -mattr=+relax --crel a.s -o acrel.o
# RUN: ld.lld -r acrel.o b.o -o out.crel.ro
# RUN: llvm-objdump -dr --no-show-raw-insn out.crel.ro | FileCheck %s --check-prefix=CHECKC

# CHECKC:      <_start>:
# CHECKC-NEXT:   0:    pcalau12i $a0, 0
# CHECKC-NEXT:           0000000000000000:  R_LARCH_PCALA_HI20   .Ltext1_start
# CHECKC-NEXT:           0000000000000000:  R_LARCH_RELAX        *ABS*
# CHECKC-NEXT:   4:    addi.d $a0, $a0, 0
# CHECKC-NEXT:           0000000000000004:  R_LARCH_PCALA_LO12   .Ltext1_start
# CHECKC-NEXT:           0000000000000004:  R_LARCH_RELAX        *ABS*
# CHECKC-NEXT:   8:    nop
# CHECKC-NEXT:           0000000000000008:  R_LARCH_ALIGN        *ABS*+0x4
# CHECKC-EMPTY:
# CHECKC-NEXT: <b0>:
# CHECKC-NEXT:   c:    addi.d $a0, $a1, 1

#--- a.s
.globl _start
_start:
  la.pcrel $a0, .Ltext1_start

.section .text1,"ax"
.Ltext1_start:
  la.pcrel $a1, .Ltext1_start

#--- b.s
.macro addi dst, src1, src2
.ifdef ELF64
  addi.d \dst, \src1, \src2
.else
  addi.w \dst, \src1, \src2
.endif
.endm

## Needs synthesized ALIGN.
.option push
.option norelax
.balign 8
b0:
  addi $a0, $a1, 1

.section .text1,"ax"
.balign 8
  addi $a0, $a1, 2

.option pop

#--- b1.s
# Starts with an ALIGN relocation, don't need synthesized ALIGN.
.option push
.option norelax
  .reloc ., R_LARCH_ALIGN, 4
  nop
.balign 8
b1:
  addi.d $a0, $a1, 3
.option pop

#--- c.s
## Alignment == 4, don't need synthesized ALIGN.
.balign 4
c0:
  addi.d $a0, $a1, 4

#--- d.s
## Needs synthesized ALIGN.
.balign 8
d0:
  addi.d $a0, $a1, 5
