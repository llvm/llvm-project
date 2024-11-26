# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o

# RUN: ld.lld %t.o -o %t.out
# RUN: llvm-readelf -x .rodata %t.out | FileCheck %s
# CHECK: Hex dump of section '.rodata':
# CHECK-NEXT: 0x00201120 010203

## --shuffle-padding= inserts segment offset padding and pre-section padding.
# RUN: ld.lld --shuffle-padding=1 %t.o -o %t.out
# RUN: llvm-readelf -x .rodata %t.out | FileCheck --check-prefix=PAD1 %s
# PAD1: Hex dump of section '.rodata':
# PAD1-NEXT: 0x00201548 0102cc03

## Size of segment offset padding and location of pre-section padding is
## dependent on the seed.
# RUN: ld.lld --shuffle-padding=2 %t.o -o %t.out
# RUN: llvm-readelf -x .rodata %t.out | FileCheck --check-prefix=PAD2 %s
# PAD2: Hex dump of section '.rodata':
# PAD2-NEXT: 0x00201dc8 cc010203

.section .rodata.a,"ax"
.byte 1

.section .rodata.b,"ax"
.byte 2

.section .rodata.c,"ax"
.byte 3
