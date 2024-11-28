# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o

# RUN: ld.lld %t.o -o %t.out
# RUN: llvm-readelf -x .data %t.out | FileCheck %s
# CHECK: Hex dump of section '.data':
# CHECK-NEXT: 0x00202158 010203

## --shuffle-padding= inserts segment offset padding and pre-section padding.
# RUN: ld.lld --shuffle-padding=1 %t.o -o %t.out
# RUN: llvm-readelf -x .data %t.out | FileCheck --check-prefix=PAD1 %s
# PAD1: Hex dump of section '.data':
# PAD1: 0x00203500 00000000 00000000 00000000 01000203

## --shuffle-padding=  does not affect .rodata flags.
# RUN: llvm-readelf -S %t.out | FileCheck --check-prefix=HEADER %s
# HEADER: .data             PROGBITS        0000000000202580 000580 000f90 00  WA  0   0  1

## Size of segment offset padding and location of pre-section padding is
## dependent on the seed.
# RUN: ld.lld --shuffle-padding=2 %t.o -o %t.out
# RUN: llvm-readelf -x .data %t.out | FileCheck --check-prefix=PAD2 %s
# PAD2: Hex dump of section '.data':
# PAD2: 0x002037e0 00000000 00000000 00000000 00010203

.section .data.a,"aw",@progbits
a:
.byte 1

.section .data.b,"aw",@progbits
b:
.byte 2

.section .data.c,"aw",@progbits
c:
.byte 3
