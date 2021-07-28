// REQUIRES: arm
// RUN: llvm-mc -arm-add-build-attributes -filetype=obj -triple=thumbv7a-none-linux-gnueabi %s -o %t
// RUN: echo "SECTIONS { \
// RUN:       .text 0x100000 : { *(SORT_BY_NAME(.text.*)) } \
// RUN:       }" > %t.script
// RUN: ld.lld --script %t.script %t -o %t2
// RUN: llvm-objdump -d %t2 --start-address=0x100000 --stop-address=0x100008 | FileCheck --check-prefix=CHECK1 %s
// RUN: llvm-objdump -d %t2 --start-address=0x1000004 --stop-address=0x100000e | FileCheck --check-prefix=CHECK2 %s

 .syntax unified

// Test that linkerscript sorting does not apply to Thunks, we expect that the
// sort will reverse the order of sections presented here.

// Define a function aligned on a megabyte boundary
 .macro FUNCTION suff
 .section .text.\suff\(), "ax", %progbits
 .thumb
 .balign 0x100000
 .globl tfunc\suff\()
 .type  tfunc\suff\(), %function
tfunc\suff\():
 bx lr
 .endm

 FUNCTION 31
 FUNCTION 30
 FUNCTION 29
 FUNCTION 28
 FUNCTION 27
 FUNCTION 26
 FUNCTION 25
 FUNCTION 24
 FUNCTION 23
 FUNCTION 22
 FUNCTION 21
 FUNCTION 20
 FUNCTION 19
 FUNCTION 18
 FUNCTION 17
 FUNCTION 16
 FUNCTION 15
// CHECK2: <__Thumbv7ABSLongThunk_tfunc31>:
// CHECK2-NEXT:  1000004:       ff f3 fc 97     b.w     0x2000000 <tfunc31>
 FUNCTION 14
 FUNCTION 13
 FUNCTION 12
 FUNCTION 11
 FUNCTION 10
 FUNCTION 09
 FUNCTION 08
 FUNCTION 07
 FUNCTION 06
 FUNCTION 05
 FUNCTION 04
 FUNCTION 03
 FUNCTION 02
 FUNCTION 01
 .section .text.00, "ax", %progbits
 .thumb
 .globl _start
_start:
// Expect no range extension needed for tfunc01 and an extension needed for
// tfunc31
 bl tfunc01
 bl tfunc31
// CHECK1: <_start>:
// CHECK1-NEXT:   100000:       ff f0 fe ff     bl      0x200000 <tfunc01>
// CHECK1-NEXT:   100004:       ff f2 fe d7     bl      0x1000004 <__Thumbv7ABSLongThunk_tfunc31>
