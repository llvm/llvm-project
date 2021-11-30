# REQUIRES: ppc

# RUN: llvm-mc -filetype=obj -triple=powerpc64le-linux %s -o %t.o
# RUN: ld.lld %t.o -z separate-code -o %t.ppc64le
# RUN: llvm-readelf -Sl %t.ppc64le | FileCheck %s
# RUN: od -Ax -t x1 -N16 -j0x1fff0 %t.ppc64le | FileCheck %s -check-prefix=LE

# RUN: llvm-mc -filetype=obj -triple=powerpc64-linux %s -o %t.o
# RUN: ld.lld %t.o -z separate-code -o %t.ppc64
# RUN: llvm-readelf -Sl %t.ppc64 | FileCheck %s
# RUN: od -Ax -t x1 -N16 -j0x1fff0 %t.ppc64 | FileCheck %s -check-prefix=BE

# CHECK:      [Nr] Name              Type            Address          Off    Size   ES Flg Lk Inf Al
# CHECK-NEXT: [ 0]                   NULL            0000000000000000 000000 000000 00      0   0  0
# CHECK-NEXT: [ 1] .text             PROGBITS        0000000010010000 010000 000004 00  AX  0   0  4
## TODO Remove empty .branch_lt
# CHECK-NEXT: [ 2] .branch_lt        PROGBITS        0000000010020000 020000 000000 00  WA  0   0  8
# CHECK-NEXT: [ 3] .comment          PROGBITS        0000000000000000 020000 000008 01  MS  0   0  1

# CHECK:      Type           Offset   VirtAddr           PhysAddr           FileSiz  MemSiz   Flg Align
# CHECK-NEXT: PHDR           0x000040 0x0000000010000040 0x0000000010000040 0x000118 0x000118 R   0x8
# CHECK-NEXT: LOAD           0x000000 0x0000000010000000 0x0000000010000000 0x000158 0x000158 R   0x10000
# CHECK-NEXT: LOAD           0x010000 0x0000000010010000 0x0000000010010000 0x010000 0x010000 R E 0x10000

## Check that executable page is filled with traps at its end.
# LE: 01fff0 08 00 e0 7f 08 00 e0 7f 08 00 e0 7f 08 00 e0 7f
# BE: 01fff0 7f e0 00 08 7f e0 00 08 7f e0 00 08 7f e0 00 08

.globl _start
_start:
  nop
