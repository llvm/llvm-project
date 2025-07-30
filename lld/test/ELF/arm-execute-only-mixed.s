// REQUIRES: arm
// RUN: rm -rf %t && split-file %s %t && cd %t

// RUN: llvm-mc -filetype=obj -triple=armv7 start.s -o start.o
// RUN: llvm-mc -filetype=obj -triple=armv7 xo.s -o xo.o
// RUN: llvm-mc -filetype=obj -triple=armv7 rx.s -o rx.o
// RUN: ld.lld start.o xo.o -o xo
// RUN: ld.lld start.o rx.o -o rx-default
// RUN: ld.lld --xosegment start.o rx.o -o rx-xosegment
// RUN: ld.lld --no-xosegment start.o rx.o -o rx-no-xosegment
// RUN: llvm-readelf -l xo | FileCheck --check-prefix=CHECK-XO %s
// RUN: llvm-readelf -l rx-default | FileCheck --check-prefix=CHECK-MERGED %s
// RUN: llvm-readelf -l rx-xosegment | FileCheck --check-prefix=CHECK-SEPARATE %s
// RUN: llvm-readelf -l rx-no-xosegment | FileCheck --check-prefix=CHECK-MERGED %s

// CHECK-XO:      PHDR
// CHECK-XO-NEXT: LOAD
// CHECK-XO-NEXT: LOAD 0x0000b4 0x000200b4 0x000200b4 0x0000c 0x0000c   E 0x10000
/// Index should match the index of the LOAD segment above.
// CHECK-XO:      02   .text .foo

// CHECK-MERGED:      PHDR
// CHECK-MERGED-NEXT: LOAD
// CHECK-MERGED-NEXT: LOAD 0x0000b4 0x000200b4 0x000200b4 0x0000c 0x0000c R E 0x10000
/// Index should match the index of the LOAD segment above.
// CHECK-MERGED:      02   .text .foo

// CHECK-SEPARATE:      PHDR
// CHECK-SEPARATE-NEXT: LOAD
// CHECK-SEPARATE-NEXT: LOAD 0x0000d4 0x000200d4 0x000200d4 0x00008 0x00008   E 0x10000
// CHECK-SEPARATE-NEXT: LOAD 0x0000dc 0x000300dc 0x000300dc 0x00004 0x00004 R E 0x10000
/// Index should match the index of the LOAD segment above.
// CHECK-SEPARATE:      02   .text
// CHECK-SEPARATE:      03   .foo

//--- start.s
.section .text,"axy",%progbits,unique,0
.global _start
_start:
  bl foo
  bx lr

//--- xo.s
.section .foo,"axy",%progbits,unique,0
.global foo
foo:
  bx lr

//--- rx.s
/// Ensure that the implicitly-created .text section has the SHF_ARM_PURECODE flag.
.section .text,"axy",%progbits,unique,0
.section .foo,"ax",%progbits,unique,0
.global foo
foo:
  bx lr
