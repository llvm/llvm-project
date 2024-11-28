# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o

## --shuffle-padding= inserts segment offset padding and pre-section padding,
## and does not affect .bss flags.
# RUN: ld.lld --shuffle-padding=1 %t.o -o %t.out
# RUN: llvm-readelf -sS %t.out | FileCheck --check-prefix=HEADER %s
# HEADER: .bss              NOBITS          0000000000202580 000580 000f90 00  WA  0   0  1
# HEADER: 1: 000000000020350c     0 NOTYPE  LOCAL  DEFAULT     2 a
# HEADER: 2: 000000000020350e     0 NOTYPE  LOCAL  DEFAULT     2 b
# HEADER: 3: 000000000020350f     0 NOTYPE  LOCAL  DEFAULT     2 c

.section .bss.a,"a",@nobits
a:
.zero 1

.section .bss.b,"a",@nobits
b:
.zero 1

.section .bss.c,"a",@nobits
c:
.zero 1
