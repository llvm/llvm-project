// REQUIRES: arm
// RUN: llvm-mc -arm-add-build-attributes -filetype=obj -triple=thumbv7a-none-linux-gnueabi %s -o %t.o
// RUN: ld.lld %t.o -o %t
// The output file is large, most of it zeroes. We dissassemble only the
// parts we need to speed up the test and avoid a large output file
// RUN: llvm-objdump -d %t --print-imm-hex --start-address=0x80000 --stop-address=0x80010 --triple=thumbv7a-linux-gnueabihf | FileCheck --check-prefix=CHECK1 %s
// RUN: llvm-objdump -d %t --print-imm-hex --start-address=0x100000 --stop-address=0x100008 --triple=thumbv7a-linux-gnueabihf | FileCheck --check-prefix=CHECK2 %s
// RUN: llvm-objdump -d %t --print-imm-hex --start-address=0x180000 --stop-address=0x18000a --triple=thumbv7a-linux-gnueabihf | FileCheck --check-prefix=CHECK3 %s
// RUN: llvm-objdump -d %t --print-imm-hex --start-address=0x500004 --stop-address=0x500008 --triple=thumbv7a-linux-gnueabihf | FileCheck --check-prefix=CHECK4 %s
// RUN: llvm-objdump -d %t --print-imm-hex --start-address=0x580000 --stop-address=0x580006 --triple=thumbv7a-linux-gnueabihf | FileCheck --check-prefix=CHECK5 %s
// RUN: llvm-objdump -d %t --print-imm-hex --start-address=0x1000004 --stop-address=0x100000c --triple=thumbv7a-linux-gnueabihf | FileCheck --check-prefix=CHECK6 %s
// RUN: llvm-objdump -d %t --print-imm-hex --start-address=0x1100000 --stop-address=0x1100006 --triple=thumbv7a-linux-gnueabihf | FileCheck --check-prefix=CHECK7 %s
// Test Range extension Thunks for the Thumb conditional branch instruction.
// This instruction only has a range of 1Mb whereas all the other Thumb wide
// Branch instructions have 16Mb range. We still place our pre-created Thunk
// Sections at 16Mb intervals as conditional branches to a target defined
// in a different section are rare.
 .syntax unified
// Define a function aligned on a half megabyte boundary
 .macro FUNCTION suff
 .section .text.\suff\(), "ax", %progbits
 .thumb
 .balign 0x80000
 .globl tfunc\suff\()
 .type  tfunc\suff\(), %function
tfunc\suff\():
 bx lr
 .endm

 .globl _start
_start:
 FUNCTION 00
// Long Range Thunk needed for 16Mb range branch, can reach pre-created Thunk
// Section
 bl tfunc33
// CHECK1: Disassembly of section .text:
// CHECK1-EMPTY:
// CHECK1-NEXT: <tfunc00>:
// CHECK1-NEXT:    80000:       70 47   bx      lr
// CHECK1-NEXT:    80002:       7f f3 ff d7     bl      0x1000004 <__Thumbv7ABSLongThunk_tfunc33>
// CHECK1: <__Thumbv7ABSLongThunk_tfunc05>:
// CHECK1-NEXT:    80008:       7f f2 fa bf     b.w     0x300000 <tfunc05>
// CHECK1: <__Thumbv7ABSLongThunk_tfunc00>:
// CHECK1-NEXT:    8000c:       ff f7 f8 bf     b.w     0x80000 <tfunc00>
 FUNCTION 01
// tfunc02 is within range of tfunc02
 beq.w tfunc02
// tfunc05 is out of range, and we can't reach the pre-created Thunk Section
// create a new one.
 bne.w tfunc05
// CHECK2:  <tfunc01>:
// CHECK2-NEXT:   100000:       70 47   bx      lr
// CHECK2-NEXT:   100002:       3f f0 fd a7     beq.w   0x180000 <tfunc02>
// CHECK2-NEXT:   100006:       7f f4 ff a7     bne.w   0x80008 <__Thumbv7ABSLongThunk_tfunc05>
 FUNCTION 02
// We can reach the Thunk Section created for bne.w tfunc05
 bne.w tfunc05
 beq.w tfunc00
// CHECK3:        180000:       70 47   bx      lr
// CHECK3-NEXT:   180002:       40 f4 01 80     bne.w   0x80008 <__Thumbv7ABSLongThunk_tfunc05>
// CHECK3-NEXT:   180006:       00 f4 01 80     beq.w   0x8000c <__Thumbv7ABSLongThunk_tfunc00>
 FUNCTION 03
 FUNCTION 04
 FUNCTION 05
 FUNCTION 06
 FUNCTION 07
 FUNCTION 08
 FUNCTION 09
// CHECK4:  <__Thumbv7ABSLongThunk_tfunc03>:
// CHECK4-NEXT:   500004:       ff f4 fc bf     b.w     0x200000 <tfunc03>
 FUNCTION 10
// We can't reach any Thunk Section, create a new one
 beq.w tfunc03
// CHECK5: <tfunc10>:
// CHECK5-NEXT:   580000:       70 47   bx      lr
// CHECK5-NEXT:   580002:       3f f4 ff a7     beq.w   0x500004 <__Thumbv7ABSLongThunk_tfunc03>
 FUNCTION 11
 FUNCTION 12
 FUNCTION 13
 FUNCTION 14
 FUNCTION 15
 FUNCTION 16
 FUNCTION 17
 FUNCTION 18
 FUNCTION 19
 FUNCTION 20
 FUNCTION 21
 FUNCTION 22
 FUNCTION 23
 FUNCTION 24
 FUNCTION 25
 FUNCTION 26
 FUNCTION 27
 FUNCTION 28
 FUNCTION 29
 FUNCTION 30
 FUNCTION 31
// CHECK6:  <__Thumbv7ABSLongThunk_tfunc33>:
// CHECK6-NEXT:  1000004:       ff f0 fc bf     b.w     0x1100000 <tfunc33>
// CHECK6: <__Thumbv7ABSLongThunk_tfunc00>:
// CHECK6-NEXT:  1000008:       7f f4 fa 97     b.w     0x80000 <tfunc00>
 FUNCTION 32
 FUNCTION 33
 // We should be able to reach an existing ThunkSection.
 b.w tfunc00
// CHECK7: <tfunc33>:
// CHECK7-NEXT:  1100000:       70 47           bx      lr
// CHECK7-NEXT:  1100002:       00 f7 01 b8     b.w     0x1000008 <__Thumbv7ABSLongThunk_tfunc00>
