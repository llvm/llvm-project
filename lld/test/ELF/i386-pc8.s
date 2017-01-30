# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=i386-pc-linux-gnu %s -o %t1.o
# RUN: llvm-mc -filetype=obj -triple=i386-pc-linux-gnu %S/Inputs/i386-pc8.s -o %t2.o
# RUN: ld.lld -Ttext 0x0 %t1.o %t2.o -o %t.out
# RUN: llvm-objdump -s -section=.text %t.out | FileCheck %s

# CHECK:      Contents of section .text:
# CHECK-NEXT:  0000 15253748

.byte und-.+0x11
.byte und-.+0x22
.byte und+0x33
.byte und+0x44
