// REQUIRES: aarch64
// RUN: rm -rf %t && split-file %s %t && cd %t

// RUN: llvm-mc -filetype=obj -triple=aarch64 start.s -o start.o
// RUN: llvm-mc -filetype=obj -triple=aarch64 xo.s -o xo.o
// RUN: llvm-mc -filetype=obj -triple=aarch64 rx.s -o rx.o
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
// CHECK-XO-NEXT: LOAD 0x000120 0x0000000000210120 0x0000000000210120 0x00000c 0x00000c   E 0x10000
/// Index should match the index of the LOAD segment above.
// CHECK-XO:      02   .text .foo

// CHECK-MERGED:      PHDR
// CHECK-MERGED-NEXT: LOAD
// CHECK-MERGED-NEXT: LOAD 0x000120 0x0000000000210120 0x0000000000210120 0x00000c 0x00000c R E 0x10000
/// Index should match the index of the LOAD segment above.
// CHECK-MERGED:      02   .text .foo

// CHECK-SEPARATE:      PHDR
// CHECK-SEPARATE-NEXT: LOAD
// CHECK-SEPARATE-NEXT: LOAD 0x000158 0x0000000000210158 0x0000000000210158 0x000008 0x000008   E 0x10000
// CHECK-SEPARATE-NEXT: LOAD 0x000160 0x0000000000220160 0x0000000000220160 0x000004 0x000004 R E 0x10000
/// Index should match the index of the LOAD segment above.
// CHECK-SEPARATE:      02   .text
// CHECK-SEPARATE:      03   .foo

//--- start.s
.section .text,"axy",@progbits,unique,0
.global _start
_start:
  bl foo
  ret

//--- xo.s
.section .foo,"axy",@progbits,unique,0
.global foo
foo:
  ret

//--- rx.s
/// Ensure that the implicitly-created .text section has the SHF_AARCH64_PURECODE flag.
.section .text,"axy",@progbits,unique,0
.section .foo,"ax",@progbits,unique,0
.global foo
foo:
  ret
