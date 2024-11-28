# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o

# RUN: ld.lld %t.o -o %t.out
# RUN: llvm-readelf -x .text %t.out | FileCheck %s
# CHECK: Hex dump of section '.text':
# CHECK-NEXT: 0x00201120 010203

## --shuffle-padding= inserts segment offset padding and pre-section padding.
# RUN: ld.lld --shuffle-padding=1 %t.o -o %t.out
# RUN: llvm-readelf -x .text %t.out | FileCheck --check-prefix=PAD1 %s
# PAD1: Hex dump of section '.text':
# PAD1: 0x00201540 cccccccc cccccccc 0102cc03

## --shuffle-padding=  does not affect .text flags.
# RUN: llvm-readelf -S %t.out | FileCheck --check-prefix=HEADER %s
# HEADER: .text             PROGBITS        0000000000201120 000120 00042c 00  AX  0   0  4

.section .text.a,"ax"
.byte 1

.section .text.b,"ax"
.byte 2

.section .text.c,"ax"
.byte 3
