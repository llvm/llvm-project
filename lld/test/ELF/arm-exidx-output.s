// REQUIRES: arm
// RUN: llvm-mc -filetype=obj -triple=armv7a-none-linux-gnueabi %s -o %t.o
// RUN: ld.lld %t.o -o %t
// RUN: llvm-readelf -S %t | FileCheck %s

// RUN: echo 'SECTIONS { .text.f1 : { *(.text.f1) } .text.f2 : { *(.text.f2) } }' > %t.lds
// RUN: ld.lld -T %t.lds %t.o -o %t1
// RUN: llvm-readelf -S %t1 | FileCheck --check-prefix=MULTI %s

// Check that only a single .ARM.exidx output section is created when
// there are input sections of the form .ARM.exidx.<section-name>. The
// assembler creates the .ARM.exidx input sections with the .cantunwind
// directive

// CHECK:      [Nr] Name       Type      {{.*}} Flg Lk
// CHECK-NEXT: [ 0]
// CHECK-NEXT: [ 1] .ARM.exidx ARM_EXIDX {{.*}}  AL  2
// CHECK-NEXT: [ 2] .text      PROGBITS  {{.*}}  AX  0

// MULTI:      [Nr] Name       Type      {{.*}} Flg Lk
// MULTI-NEXT: [ 0]
// MULTI-NEXT: [ 1] .ARM.exidx ARM_EXIDX {{.*}}  AL  2
// MULTI-NEXT: [ 2] .text.f1   PROGBITS  {{.*}}  AX  0
// MULTI-NEXT: [ 3] .text.f2   PROGBITS  {{.*}}  AX  0
// MULTI-NEXT: [ 4] .text      PROGBITS  {{.*}}  AX  0

 .syntax unified
 .section .text, "ax",%progbits
 .globl _start
_start:
 .fnstart
 bx lr
 .cantunwind
 .fnend

 .section .text.f1, "ax", %progbits
 .globl f1
f1:
 .fnstart
 bx lr
 .cantunwind
 .fnend

 .section .text.f2, "ax", %progbits
 .globl f2
f2:
 .fnstart
 bx lr
 .cantunwind
 .fnend
