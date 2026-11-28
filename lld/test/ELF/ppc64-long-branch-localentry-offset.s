# REQUIRES: ppc

# RUN: llvm-mc -filetype=obj -triple=ppc64le %s -o %t.o
# RUN: echo 'SECTIONS { .text 0x10000 : { *(.text) } .text_high 0x200fff8 : { *(.text_high) } }' > %t.lds
# RUN: ld.lld -T %t.lds %t.o -o %t
# RUN: llvm-nm %t | FileCheck %s

# CHECK-DAG: 0000000000010000 T _start
# CHECK-DAG: 0000000000010010 t __long_branch_callee
# CHECK-DAG: 000000000200fff8 T callee

# The bl instruction jumps to the local entry. The distance requires a long branch stub:
# localentry(callee) - _start = 0x200fff8+8 - 0x10000 = 0x2000000

# We used to compute globalentry(callee) - _start and caused a "R_PPC64_REL24
# out of range" error because we didn't create the stub.

.globl _start
_start:
  bl callee

.section .text_high,"ax",@progbits
.globl callee
callee:
.Lgep0:
  addis 2, 12, .TOC.-.Lgep0@ha
  addi 2, 2, .TOC.-.Lgep0@l
.Llep0:
  .localentry callee, .Llep0-.Lgep0
  blr
