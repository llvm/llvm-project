# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o

## --randomize-section-padding= inserts segment offset padding and pre-section
## padding, and does not affect flags. Segment offset padding is only inserted
## when PT_LOAD changes, as shown by .bss size (.data and .bss share a PT_LOAD).

# RUN: ld.lld --randomize-section-padding=6 %t.o -o %t.out
# RUN: llvm-readelf -sS -x .rodata -x .text -x .data %t.out | FileCheck --check-prefix=PAD6 %s

# PAD6: .rodata           PROGBITS        0000000000200158 000158 000b8d 00   A  0   0  1
# PAD6: .text             PROGBITS        0000000000201ce8 000ce8 000270 00  AX  0   0  4
# PAD6: .data             PROGBITS        0000000000202f58 000f58 000941 00  WA  0   0  1
# PAD6: .bss              NOBITS          0000000000203899 001899 000003 00  WA  0   0  1

# PAD6: 0000000000203899     0 NOTYPE  LOCAL  DEFAULT     4 a
# PAD6: 000000000020389a     0 NOTYPE  LOCAL  DEFAULT     4 b
# PAD6: 000000000020389b     0 NOTYPE  LOCAL  DEFAULT     4 c

# PAD6: Hex dump of section '.rodata':
# PAD6: 0x00200cd8 00000000 00000000 00000102 03
# PAD6: Hex dump of section '.text':
# PAD6: 0x00201f48 cccccccc cccccccc cccccccc 0405cc06
# PAD6: Hex dump of section '.data':
# PAD6: 0x00203888 00000000 00000000 00000000 00000708
# PAD6: 0x00203898 09

## Size of segment offset padding and location of pre-section padding is
## dependent on the seed.

# RUN: ld.lld --randomize-section-padding=46 %t.o -o %t.out
# RUN: llvm-readelf -sS -x .rodata -x .text -x .data %t.out | FileCheck --check-prefix=PAD46 %s

# PAD46: .rodata           PROGBITS        0000000000200158 000158 000cc0 00   A  0   0  1
# PAD46: .text             PROGBITS        0000000000201e18 000e18 0009bf 00  AX  0   0  4
# PAD46: .data             PROGBITS        00000000002037d7 0017d7 000540 00  WA  0   0  1
# PAD46: .bss              NOBITS          0000000000203d17 001d17 000004 00  WA  0   0  1

# PAD46: 0000000000203d17     0 NOTYPE  LOCAL  DEFAULT     4 a
# PAD46: 0000000000203d18     0 NOTYPE  LOCAL  DEFAULT     4 b
# PAD46: 0000000000203d1a     0 NOTYPE  LOCAL  DEFAULT     4 c

# PAD46: Hex dump of section '.rodata':
# PAD46: 0x00200e08 00000000 00000000 00000000 00010203
# PAD46: Hex dump of section '.text':
# PAD46: 0x002027c8 cccccccc cccccccc cccccccc 040506
# PAD46: Hex dump of section '.data':
# PAD46: 0x00203d07 00000000 00000000 00000000 07000809

.section .rodata.a,"a",@progbits
.byte 1

.section .rodata.b,"a",@progbits
.byte 2

.section .rodata.c,"a",@progbits
.byte 3

.section .text.a,"ax",@progbits
.byte 4

.section .text.b,"ax",@progbits
.byte 5

.section .text.c,"ax",@progbits
.byte 6

.section .data.a,"aw",@progbits
.byte 7

.section .data.b,"aw",@progbits
.byte 8

.section .data.c,"aw",@progbits
.byte 9

.section .bss.a,"a",@nobits
a:
.zero 1

.section .bss.b,"a",@nobits
b:
.zero 1

.section .bss.c,"a",@nobits
c:
.zero 1

