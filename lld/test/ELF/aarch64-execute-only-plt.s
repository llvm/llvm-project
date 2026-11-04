// REQUIRES: aarch64

// RUN: llvm-mc -filetype=obj -triple=aarch64-linux-gnu %s -o %t.o
// RUN: llvm-mc -filetype=obj -triple=aarch64-linux-gnu %S/Inputs/plt-aarch64.s -o %t2.o
// RUN: ld.lld -shared %t2.o -soname=t2.so -o %t2.so
// RUN: ld.lld %t.o %t2.so -o %t
// RUN: llvm-readelf -S -l %t | FileCheck %s

// CHECK-LABEL: Section Headers:
// CHECK:         .text PROGBITS 00000000002102a0 0002a0 000010 00 AXy  0   0  4
// CHECK:         .plt  PROGBITS 00000000002102b0 0002b0 000030 00 AXy  0   0 16
// CHECK:         .iplt PROGBITS 00000000002102e0 0002e0 000010 00 AXy  0   0 16

// CHECK-LABEL: Program Headers:
// CHECK:         PHDR
// CHECK-NEXT:    LOAD
// CHECK-NEXT:    LOAD 0x0002a0 0x00000000002102a0 0x00000000002102a0 0x000050 0x000050   E 0x10000

// CHECK-LABEL: Section to Segment mapping:
/// Index should match the index of the LOAD segment above.
// CHECK:         02   .text .plt .iplt

.global bar

.section .text,"axy",@progbits,unique,0
.global _start
_start:
  bl foo
  bl bar
  ret

.globl foo
.type foo STT_GNU_IFUNC
foo:
  ret
