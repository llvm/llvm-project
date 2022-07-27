// REQUIRES: arm
// RUN: llvm-mc -arm-add-build-attributes -filetype=obj -triple=armv7a-none-linux-gnueabi %s -o %t
// RUN: ld.lld %t -o %t2
// The output file is large, most of it zeroes. We dissassemble only the
// parts we need to speed up the test and avoid a large output file
// RUN: llvm-objdump -d %t2 --start-address=0x100000 --stop-address=0x10001c --triple=armv7a-linux-gnueabihf | FileCheck --check-prefix=CHECK1 %s
// RUN: llvm-objdump -d %t2 --start-address=0x200000 --stop-address=0x20000a | FileCheck --check-prefix=CHECK2 %s
// RUN: llvm-objdump -d %t2 --start-address=0x1000004 --stop-address=0x1000010 --triple=armv7a-linux-gnueabihf | FileCheck --check-prefix=CHECK3 %s
// RUN: llvm-objdump -d %t2 --start-address=0x1000010 --stop-address=0x100001a | FileCheck --check-prefix=CHECK4 %s
// RUN: llvm-objdump -d %t2 --start-address=0x1f00004 --stop-address=0x1f0000e | FileCheck --check-prefix=CHECK5 %s
// RUN: llvm-objdump -d %t2 --start-address=0x2200000 --stop-address=0x2200006 | FileCheck --check-prefix=CHECK6 %s
// RUN: llvm-objdump -d %t2 --start-address=0x2300000 --stop-address=0x2300008 --triple=armv7a-linux-gnueabihf | FileCheck --check-prefix=CHECK7 %s
// RUN: llvm-objdump -d %t2 --start-address=0x2e00004 --stop-address=0x2e00010 --triple=armv7a-linux-gnueabihf | FileCheck --check-prefix=CHECK8 %s
// RUN: llvm-objdump -d %t2 --start-address=0x3300004 --stop-address=0x3300010 | FileCheck --check-prefix=CHECK9 %s
// RUN: llvm-objdump -d %t2 --start-address=0x4100000 --stop-address=0x410000c --triple=armv7a-linux-gnueabihf | FileCheck --check-prefix=CHECK10 %s
// RUN: llvm-objdump -d %t2 --start-address=0x4200000 --stop-address=0x4200008 | FileCheck --check-prefix=CHECK11 %s

// Test the Range extension Thunks for ARM and Thumb when all the code is in a
// single OutputSection. The ARM branches and branch and link instructions
// have a range of 32Mb, the Thumb unconditional branch and
// branch and link instructions have . We create a series of Functions a
// megabyte apart. We expect range extension thunks to be created when a
// branch is out of range. Thunks will be reused whenever they are in range
 .syntax unified

// Define a function aligned on a megabyte boundary
 .macro ARMFUNCTION suff
 .section .text.\suff\(), "ax", %progbits
 .arm
 .balign 0x100000
 .globl afunc\suff\()
 .type  afunc\suff\(), %function
afunc\suff\():
 bx lr
 .endm

// Define a function aligned on a megabyte boundary
 .macro THUMBFUNCTION suff
 .section .text.\suff\(), "ax", %progbits
 .thumb
 .balign 0x100000
 .globl tfunc\suff\()
 .type  tfunc\suff\(), %function
tfunc\suff\():
 bx lr
 .endm

 .section .text, "ax", %progbits
 .thumb
 .globl _start
_start:

 ARMFUNCTION 00
// Expect ARM bl to be in range (can use blx to change state)
 bl tfunc31
// ARM b and beq are in range but need Thunk to change state to Thumb
 b  tfunc31
 beq tfunc31
// afunc32 is out of range of ARM branch and branch and link
 bl afunc32
 b  afunc32
 bne afunc32
// CHECK1:  <afunc00>:
// CHECK1-NEXT:   100000:       e12fff1e        bx      lr
// CHECK1-NEXT:   100004:       fa7bfffd        blx     0x2000000 <tfunc31>
// CHECK1-NEXT:   100008:       ea3bfffd        b       0x1000004 <__ARMv7ABSLongThunk_tfunc31>
// CHECK1-NEXT:   10000c:       0a3bfffc        beq     0x1000004 <__ARMv7ABSLongThunk_tfunc31>
// CHECK1-NEXT:   100010:       eb7ffffa        bl      0x2100000 <afunc32>
// CHECK1-NEXT:   100014:       ea7ffff9        b       0x2100000 <afunc32>
// CHECK1-NEXT:   100018:       1a7ffff8        bne     0x2100000 <afunc32>
 THUMBFUNCTION 01
// Expect Thumb bl to be in range (can use blx to change state)
 bl afunc14
// In range but need thunk to change state to Thumb
 b.w afunc14
// CHECK2: <tfunc01>:
// CHECK2-NEXT:   200000:       4770    bx      lr
// CHECK2-NEXT:   200002:       f0ff c7fe       blx     0xf00000 <afunc14>
// CHECK2-NEXT:   200006:       f200 9003       b.w     0x1000010 <__Thumbv7ABSLongThunk_afunc14>

 ARMFUNCTION 02
 THUMBFUNCTION 03
 ARMFUNCTION 04
 THUMBFUNCTION 05
 ARMFUNCTION 06
 THUMBFUNCTION 07
 ARMFUNCTION 08
 THUMBFUNCTION 09
 ARMFUNCTION 10
 THUMBFUNCTION 11
 ARMFUNCTION 12
 THUMBFUNCTION 13
 ARMFUNCTION 14
// CHECK3:   <__ARMv7ABSLongThunk_tfunc31>:
// CHECK3-NEXT:  1000004:       e300c001        movw    r12, #1
// CHECK3-NEXT:  1000008:       e340c200        movt    r12, #512
// CHECK3-NEXT:  100000c:       e12fff1c        bx      r12
// CHECK4: <__Thumbv7ABSLongThunk_afunc14>:
// CHECK4-NEXT:  1000010:       f240 0c00       movw    r12, #0
// CHECK4-NEXT:  1000014:       f2c0 0cf0       movt    r12, #240
// CHECK4-NEXT:  1000018:       4760    bx      r12
 THUMBFUNCTION 15
 ARMFUNCTION 16
 THUMBFUNCTION 17
 ARMFUNCTION 18
 THUMBFUNCTION 19
 ARMFUNCTION 20
 THUMBFUNCTION 21
 ARMFUNCTION 22
 THUMBFUNCTION 23
 ARMFUNCTION 24
 THUMBFUNCTION 25
 ARMFUNCTION 26
 THUMBFUNCTION 27
 ARMFUNCTION 28
 THUMBFUNCTION 29
 ARMFUNCTION 30
// Expect precreated Thunk Section here
// CHECK5: <__Thumbv7ABSLongThunk_afunc00>:
// CHECK5-NEXT:  1f00004:       f240 0c00       movw    r12, #0
// CHECK5-NEXT:  1f00008:       f2c0 0c10       movt    r12, #16
// CHECK5-NEXT:  1f0000c:       4760    bx      r12
 THUMBFUNCTION 31
 ARMFUNCTION 32
 THUMBFUNCTION 33
// Out of range, can only reach closest Thunk Section
 bl afunc00
// CHECK6:  <tfunc33>:
// CHECK6-NEXT:  2200000:       4770    bx      lr
// CHECK6-NEXT:  2200002:       f4ff ffff       bl      0x1f00004 <__Thumbv7ABSLongThunk_afunc00>
 ARMFUNCTION 34
// Out of range, can reach earlier Thunk Section
// CHECK7:  <afunc34>:
// CHECK7-NEXT:  2300000:       e12fff1e        bx      lr
// CHECK7-NEXT:  2300004:       faeffffe        blx     0x1f00004 <__Thumbv7ABSLongThunk_afunc00>
 bl afunc00
 THUMBFUNCTION 35
 ARMFUNCTION 36
 THUMBFUNCTION 37
 ARMFUNCTION 38
 THUMBFUNCTION 39
 ARMFUNCTION 40
 THUMBFUNCTION 41
 ARMFUNCTION 42
 THUMBFUNCTION 43
 ARMFUNCTION 44
 THUMBFUNCTION 45
// Expect precreated Thunk Section here
// CHECK8: <__ARMv7ABSLongThunk_tfunc35>:
// CHECK8-NEXT:  2e00004:       e300c001        movw    r12, #1
// CHECK8-NEXT:  2e00008:       e340c240        movt    r12, #576
// CHECK8-NEXT:  2e0000c:       e12fff1c        bx      r12
 ARMFUNCTION 46
 THUMBFUNCTION 47
 ARMFUNCTION 48
 THUMBFUNCTION 49
 ARMFUNCTION 50
// Expect precreated Thunk Section here
// CHECK9: <__Thumbv7ABSLongThunk_afunc34>:
// CHECK9-NEXT:  3300004:       f240 0c00       movw    r12, #0
// CHECK9-NEXT:  3300008:       f2c0 2c30       movt    r12, #560
// CHECK9-NEXT:  330000c:       4760    bx      r12
// CHECK9: <__Thumbv7ABSLongThunk_tfunc35>:
// CHECK9-NEXT:  330000e:       f4ff 97f7       b.w     0x2400000 <tfunc35>
 THUMBFUNCTION 51
 ARMFUNCTION 52
 THUMBFUNCTION 53
 ARMFUNCTION 54
 THUMBFUNCTION 55
 ARMFUNCTION 56
 THUMBFUNCTION 57
 ARMFUNCTION 58
 THUMBFUNCTION 59
 ARMFUNCTION 60
 THUMBFUNCTION 61
 ARMFUNCTION 62
 THUMBFUNCTION 63
 ARMFUNCTION 64
// afunc34 is in range, as is tfunc35 but a branch needs a state change Thunk
 bl afunc34
 b  tfunc35
// CHECK10: <afunc64>:
// CHECK10-NEXT:  4100000:      e12fff1e        bx      lr
// CHECK10-NEXT:  4100004:      eb87fffd        bl      0x2300000 <afunc34>
// CHECK10-NEXT:  4100008:      eab3fffd        b       0x2e00004 <__ARMv7ABSLongThunk_tfunc35>
 THUMBFUNCTION 65
// afunc34 and tfunc35 are both out of range
 bl afunc34
 bl tfunc35
// CHECK11: <tfunc65>:
// CHECK11:  4200000:   4770    bx      lr
// CHECK11-NEXT:  4200002:      f4ff d7ff       bl      0x3300004 <__Thumbv7ABSLongThunk_afunc34>
// CHECK11-NEXT:  4200006:      f500 d002       bl      0x330000e <__Thumbv7ABSLongThunk_tfunc35>
