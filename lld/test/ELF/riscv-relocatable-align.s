# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+c,+relax a.s -o ac.o
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+c,+relax b.s -o bc.o
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+c,+relax b1.s -o b1c.o
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+c,+relax c.s -o cc.o
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+c d.s -o dc.o

## No RELAX. Don't synthesize ALIGN.
# RUN: ld.lld -r bc.o dc.o -o bd.ro

# NOREL: no relocations

# RUN: ld.lld -r bc.o bc.o ac.o bc.o b1c.o cc.o dc.o -o out.ro
# RUN: llvm-objdump -dr -M no-aliases out.ro | FileCheck %s
# RUN: llvm-readelf -r out.ro | FileCheck %s --check-prefix=CHECK-REL

# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+relax a.s -o a.o
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+relax b.s -o b.o
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+relax d.s -o d.o
# RUN: ld.lld -r a.o b.o d.o -o out0.ro
# RUN: ld.lld -Ttext=0x10000 out0.ro -o out0
# RUN: llvm-objdump -dr -M no-aliases out0 | FileCheck %s --check-prefix=CHECK1

# CHECK:      <b0>:
# CHECK-NEXT:   0: 00158513             addi    a0, a1, 0x1
# CHECK-NEXT:   4: 0001                 c.nop
# CHECK-NEXT:   6: 0001                 c.nop
# CHECK-EMPTY:
# CHECK-NEXT: <b0>:
# CHECK-NEXT:   8: 00158513             addi    a0, a1, 0x1
# CHECK-EMPTY:
# CHECK-NEXT: <_start>:
# CHECK-NEXT:   c: 00000097             auipc   ra, 0x0
# CHECK-NEXT:           000000000000000c:  R_RISCV_CALL_PLT     foo
# CHECK-NEXT:           000000000000000c:  R_RISCV_RELAX        *ABS*
# CHECK-NEXT:  10: 000080e7             jalr    ra, 0x0(ra) <_start>
# CHECK-NEXT:  14: 0001                 c.nop
# CHECK-NEXT:           0000000000000014:  R_RISCV_ALIGN        *ABS*+0x6
# CHECK-NEXT:  16: 0001                 c.nop
# CHECK-NEXT:  18: 0001                 c.nop
# CHECK-EMPTY:
# CHECK-NEXT: <b0>:
# CHECK-NEXT:  1a: 00158513             addi    a0, a1, 0x1
# CHECK-NEXT:  1e: 0001                 c.nop
# CHECK-NEXT:  20: 0001                 c.nop
# CHECK-NEXT:           0000000000000020:  R_RISCV_ALIGN        *ABS*+0x6
# CHECK-NEXT:  22: 0001                 c.nop
# CHECK-NEXT:  24: 00000013             addi    zero, zero, 0x0
# CHECK-EMPTY:
# CHECK-NEXT: <b0>:
# CHECK-NEXT:  28: 00158513             addi    a0, a1, 0x1
# CHECK-EMPTY:
# CHECK-NEXT: <c0>:
# CHECK-NEXT:  2c: 00000097             auipc   ra, 0x0
# CHECK-NEXT:           000000000000002c:  R_RISCV_CALL_PLT     foo
# CHECK-NEXT:           000000000000002c:  R_RISCV_RELAX        *ABS*
# CHECK-NEXT:  30: 000080e7             jalr    ra, 0x0(ra) <c0>
# CHECK-NEXT:  34: 0001                 c.nop
# CHECK-NEXT:           0000000000000034:  R_RISCV_ALIGN        *ABS*+0x2
# CHECK-EMPTY:
# CHECK-NEXT: <d0>:
# CHECK-NEXT:  36: 00258513             addi    a0, a1, 0x2

# CHECK-REL:  Relocation section '.rela.text' at offset {{.*}} contains 7 entries:
# CHECK-REL:  Relocation section '.rela.text1' at offset {{.*}} contains 3 entries:

# CHECK1:      <_start>:
# CHECK1-NEXT:    010000ef      jal     ra, 0x10010 <foo>
# CHECK1-NEXT:    00000013      addi zero, zero, 0x0
# CHECK1-EMPTY:
# CHECK1-NEXT: <b0>:
# CHECK1-NEXT:    00158513      addi    a0, a1, 0x1
# CHECK1-EMPTY:
# CHECK1-NEXT: <d0>:
# CHECK1-NEXT:    00258513      addi    a0, a1, 0x2

## Test CREL.
# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+c,+relax --crel a.s -o acrel.o
# RUN: ld.lld -r acrel.o bc.o -o out1.ro
# RUN: llvm-objdump -dr -M no-aliases out1.ro | FileCheck %s --check-prefix=CHECK2

# CHECK2:      <_start>:
# CHECK2-NEXT:   0: 00000097             auipc   ra, 0x0
# CHECK2-NEXT:           0000000000000000:  R_RISCV_CALL_PLT     foo
# CHECK2-NEXT:           0000000000000000:  R_RISCV_RELAX        *ABS*
# CHECK2-NEXT:   4: 000080e7             jalr    ra, 0x0(ra) <_start>
# CHECK2-NEXT:   8: 0001                 c.nop
# CHECK2-NEXT:           0000000000000008:  R_RISCV_ALIGN        *ABS*+0x6
# CHECK2-NEXT:   a: 0001                 c.nop
# CHECK2-NEXT:   c: 0001                 c.nop
# CHECK2-EMPTY:
# CHECK2-NEXT: <b0>:
# CHECK2-NEXT:   e: 00158513             addi    a0, a1, 0x1

#--- a.s
.globl _start
_start:
  call foo

.section .text1,"ax"
.globl foo
foo:
  call foo

#--- b.s
## Needs synthesized ALIGN
.option push
.option norelax
.balign 8
b0:
  addi a0, a1, 1

.section .text1,"ax"
.balign 8
  addi a0, a1, 1

.option pop

#--- b1.s
.option push
.option norelax
  .reloc ., R_RISCV_ALIGN, 6
  addi x0, x0, 0
  c.nop
.balign 8
b0:
  addi a0, a1, 1
.option pop

#--- c.s
.balign 2
c0:
  call foo

#--- d.s
## Needs synthesized ALIGN
.balign 4
d0:
  addi a0, a1, 2
