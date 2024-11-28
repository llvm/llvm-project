# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o

# RUN: ld.lld %t.o -o %t.out
# RUN: llvm-readelf -x .rodata %t.out | FileCheck %s
# CHECK: Hex dump of section '.rodata':
# CHECK-NEXT: 0x00200120 010203

## --shuffle-padding= inserts segment offset padding and pre-section padding.
# RUN: ld.lld --shuffle-padding=1 %t.o -o %t.out
# RUN: llvm-readelf -x .rodata %t.out | FileCheck --check-prefix=PAD1 %s
# PAD1: Hex dump of section '.rodata':
# PAD1: 0x00200540 00000000 00010203

## --shuffle-padding=  does not affect .rodata flags.
# RUN: llvm-readelf -S %t.out | FileCheck --check-prefix=HEADER %s
# HEADER: .rodata           PROGBITS        0000000000200120 000120 000428 00   A  0   0  1

## Size of segment offset padding and location of pre-section padding is
## dependent on the seed.
# RUN: ld.lld --shuffle-padding=2 %t.o -o %t.out
# RUN: llvm-readelf -x .rodata %t.out | FileCheck --check-prefix=PAD2 %s
# PAD2: Hex dump of section '.rodata':
# PAD2: 0x00200dc0 00000000 00000000 01000203

.section .rodata.a,"a",@progbits
a:
.byte 1

.section .rodata.b,"a",@progbits
b:
.byte 2

.section .rodata.c,"a",@progbits
c:
.byte 3
