// REQUIRES: arm
// RUN: llvm-mc -arm-add-build-attributes -filetype=obj -triple=armv7a-none-linux-gnueabi %s -o %t
// RUN: echo "SECTIONS { \
// RUN:       . = SIZEOF_HEADERS; \
// RUN:       .R_ARM_JUMP24_callee_1 : { *(.R_ARM_JUMP24_callee_low) } \
// RUN:       .R_ARM_THM_JUMP_callee_1 : { *(.R_ARM_THM_JUMP_callee_low)} \
// RUN:       .text : { *(.text) } \
// RUN:       .arm_caller : { *(.arm_caller) } \
// RUN:       .thumb_caller : { *(.thumb_caller) } \
// RUN:       .R_ARM_JUMP24_callee_2 : { *(.R_ARM_JUMP24_callee_high) } \
// RUN:       .R_ARM_THM_JUMP_callee_2 : { *(.R_ARM_THM_JUMP_callee_high) } \
// RUN:       .got.plt 0x18b4 : {  }  } " > %t.script
// RUN: ld.lld --script %t.script %t -o %t2
// RUN: llvm-objdump --no-print-imm-hex -d --no-show-raw-insn %t2 | FileCheck --check-prefix=CHECK-THUMB --check-prefix=CHECK-ABS-THUMB %s
// RUN: llvm-objdump --no-print-imm-hex -d --no-show-raw-insn --triple=armv7a-none-linux-gnueabi %t2 | FileCheck --check-prefix=CHECK-ARM --check-prefix=CHECK-ARM-ABS-ARM %s
// RUN: ld.lld --script %t.script %t -pie -o %t3
// RUN: llvm-objdump --no-print-imm-hex -d --no-show-raw-insn %t3 | FileCheck --check-prefix=CHECK-THUMB --check-prefix=CHECK-PI-THUMB %s
// RUN: llvm-objdump --no-print-imm-hex -d --no-show-raw-insn --triple=armv7a-none-linux-gnueabi %t3 | FileCheck --check-prefix=CHECK-ARM --check-prefix=CHECK-PI-ARM %s
// RUN: ld.lld --script %t.script %t --shared -o %t4
// RUN: llvm-readobj -S -r %t4 | FileCheck -check-prefix=CHECK-DSO-REL %s
// RUN: llvm-objdump --no-print-imm-hex -d --no-show-raw-insn --triple=armv7a-none-linux-gnueabi %t4 | FileCheck -check-prefix=CHECK-ARM-PLT %s
/// Test ARM Thumb Interworking
/// The file is linked and checked 3 times to check the following contexts
/// - Absolute executables, absolute Thunks are used.
/// - Position independent executables, position independent Thunks are used.
/// - Shared object, position independent Thunks to PLT entries are used.

 .syntax unified

/// Target Sections for thunks at a lower address than the callers.
.section .R_ARM_JUMP24_callee_low, "ax", %progbits
 .thumb
 .balign 0x1000
 .globl thumb_callee1
 .type thumb_callee1, %function
thumb_callee1:
 bx lr

// CHECK-THUMB: Disassembly of section .R_ARM_JUMP24_callee_1:
// CHECK-THUMB-EMPTY:
// CHECK-THUMB: <thumb_callee1>:
// CHECK-THUMB: 1000: bx      lr
 .section .R_ARM_THM_JUMP_callee_low, "ax", %progbits
 .arm
 .balign 0x100
 .globl arm_callee1
 .type arm_callee1, %function
arm_callee1:
 bx lr
// Disassembly of section .R_ARM_THM_JUMP_callee_1:
// CHECK-ARM: <arm_callee1>:
// CHECK-ARM-NEXT: 1100: bx      lr

/// Calling sections
/// At present ARM and Thumb interworking thunks are always added to the calling
/// section.
 .section .arm_caller, "ax", %progbits
 .arm
 .balign 0x100
 .globl arm_caller
 .type arm_caller, %function
arm_caller:
/// If target supports BLX and target is in range we don't need an
/// interworking thunk for a BL or BLX instruction.
 bl thumb_callee1
 blx thumb_callee1
/// A B instruction can't be transformed into a BLX and needs an interworking
/// thunk.
 b thumb_callee1
/// As long as the thunk is in range it can be reused.
 b thumb_callee1
/// There can be more than one thunk associated with a section.
 b thumb_callee2
 b thumb_callee3
/// In range ARM targets do not require interworking thunks.
 b arm_callee1
 beq arm_callee2
 bne arm_callee3
 bx lr
// CHECK-ARM-ABS-ARM: Disassembly of section .arm_caller:
// CHECK-ARM-ABS-ARM-EMPTY:
// CHECK-ARM-ABS-ARM-NEXT: <arm_caller>:
// CHECK-ARM-ABS-ARM-NEXT: 1300: blx     0x1000 <thumb_callee1>
// CHECK-ARM-ABS-ARM-NEXT: 1304: blx     0x1000 <thumb_callee1>
// CHECK-ARM-ABS-ARM-NEXT: 1308: b       0x1328 <__ARMv7ABSLongThunk_thumb_callee1>
// CHECK-ARM-ABS-ARM-NEXT: 130c: b       0x1328 <__ARMv7ABSLongThunk_thumb_callee1>
// CHECK-ARM-ABS-ARM-NEXT: 1310: b       0x1334 <__ARMv7ABSLongThunk_thumb_callee2>
// CHECK-ARM-ABS-ARM-NEXT: 1314: b       0x1340 <__ARMv7ABSLongThunk_thumb_callee3>
// CHECK-ARM-ABS-ARM-NEXT: 1318: b       0x1100 <arm_callee1>
// CHECK-ARM-ABS-ARM-NEXT: 131c: beq     0x1600 <arm_callee2>
// CHECK-ARM-ABS-ARM-NEXT: 1320: bne     0x1604 <arm_callee3>
// CHECK-ARM-ABS-ARM-NEXT: 1324: bx      lr
// CHECK-ARM-ABS-ARM:      <__ARMv7ABSLongThunk_thumb_callee1>:
// 0x1001 = thumb_callee1
// CHECK-ARM-ABS-ARM-NEXT: 1328: movw    r12, #4097
// CHECK-ARM-ABS-ARM-NEXT: 132c: movt    r12, #0
// CHECK-ARM-ABS-ARM-NEXT: 1330: bx      r12
// 0x1501 = thumb_callee2
// CHECK-ARM-ABS-ARM:      <__ARMv7ABSLongThunk_thumb_callee2>:
// CHECK-ARM-ABS-ARM-NEXT: 1334: movw    r12, #5377
// CHECK-ARM-ABS-ARM-NEXT: 1338: movt    r12, #0
// CHECK-ARM-ABS-ARM-NEXT: 133c: bx      r12
// 0x1503 = thumb_callee3
// CHECK-ARM-ABS-ARM:      <__ARMv7ABSLongThunk_thumb_callee3>:
// CHECK-ARM-ABS-ARM-NEXT: 1340: movw    r12, #5379
// CHECK-ARM-ABS-ARM-NEXT: 1344: movt    r12, #0
// CHECK-ARM-ABS-ARM-NEXT: 1348: bx      r12

// CHECK-PI-ARM: Disassembly of section .arm_caller:
// CHECK-PI-ARM-EMPTY:
// CHECK-PI-ARM-NEXT: <arm_caller>:
// CHECK-PI-ARM-NEXT: 1300: blx     0x1000 <thumb_callee1>
// CHECK-PI-ARM-NEXT: 1304: blx     0x1000 <thumb_callee1>
// CHECK-PI-ARM-NEXT: 1308: b       0x1328 <__ARMV7PILongThunk_thumb_callee1>
// CHECK-PI-ARM-NEXT: 130c: b       0x1328 <__ARMV7PILongThunk_thumb_callee1>
// CHECK-PI-ARM-NEXT: 1310: b       0x1338 <__ARMV7PILongThunk_thumb_callee2>
// CHECK-PI-ARM-NEXT: 1314: b       0x1348 <__ARMV7PILongThunk_thumb_callee3>
// CHECK-PI-ARM-NEXT: 1318: b       0x1100 <arm_callee1>
// CHECK-PI-ARM-NEXT: 131c: beq     0x1600 <arm_callee2>
// CHECK-PI-ARM-NEXT: 1320: bne     0x1604 <arm_callee3>
// CHECK-PI-ARM-NEXT: 1324: bx      lr
// CHECK-PI-ARM: <__ARMV7PILongThunk_thumb_callee1>:
// CHECK-PI-ARM-NEXT: 1328: movw    r12, #64713
// CHECK-PI-ARM-NEXT: 132c: movt    r12, #65535
// CHECK-PI-ARM-NEXT: 1330: add     r12, r12, pc
// CHECK-PI-ARM-NEXT: 1334: bx      r12
// CHECK-PI-ARM: <__ARMV7PILongThunk_thumb_callee2>:
// CHECK-PI-ARM-NEXT: 1338: movw    r12, #441
// CHECK-PI-ARM-NEXT: 133c: movt    r12, #0
// CHECK-PI-ARM-NEXT: 1340: add     r12, r12, pc
// CHECK-PI-ARM-NEXT: 1344: bx      r12
// CHECK-PI-ARM: <__ARMV7PILongThunk_thumb_callee3>:
// CHECK-PI-ARM-NEXT: 1348: movw    r12, #427
// CHECK-PI-ARM-NEXT: 134c: movt    r12, #0
// CHECK-PI-ARM-NEXT: 1350: add     r12, r12, pc
// CHECK-PI-ARM-NEXT: 1354: bx      r12

/// All PLT entries are ARM, callers via PLT no need for interworking thunks.
// CHECK-ARM-PLT: Disassembly of section .arm_caller:
// CHECK-ARM-PLT-EMPTY:
// CHECK-ARM-PLT-NEXT: <arm_caller>:
// CHECK-ARM-PLT-NEXT: 1300:        bl      0x1630
// CHECK-ARM-PLT-NEXT: 1304:        bl      0x1630
// CHECK-ARM-PLT-NEXT: 1308:        b       0x1630
// CHECK-ARM-PLT-NEXT: 130c:        b       0x1630
// CHECK-ARM-PLT-NEXT: 1310:        b       0x1660
// CHECK-ARM-PLT-NEXT: 1314:        b       0x1670
// CHECK-ARM-PLT-NEXT: 1318:        b       0x1640
// CHECK-ARM-PLT-NEXT: 131c:        beq     0x1680
// CHECK-ARM-PLT-NEXT: 1320:        bne     0x1690
// CHECK-ARM-PLT-NEXT: 1324:        bx      lr

 .section .thumb_caller, "ax", %progbits
 .balign 0x100
 .thumb
 .globl thumb_caller
 .type thumb_caller, %function
thumb_caller:
/// If target supports BLX and target is in range we don't need an
/// interworking thunk for a BL or BLX instruction.
 bl arm_callee1
 blx arm_callee1
/// A B instruction can't be transformed into a BLX and needs an interworking
/// thunk
 b.w arm_callee1
/// As long as the thunk is in range it can be reused
 b.w arm_callee2
/// There can be more than one thunk associated with a section
 b.w arm_callee3
/// Conditional branches also require interworking thunks, they can use the
/// same interworking thunks.
 beq.w arm_callee1
 beq.w arm_callee2
 bne.w arm_callee3
// CHECK-ABS-THUMB: Disassembly of section .thumb_caller:
// CHECK-ABS-THUMB-EMPTY:
// CHECK-ABS-THUMB-NEXT: <thumb_caller>:
// CHECK-ABS-THUMB-NEXT: 1400: blx     0x1100 <arm_callee1>
// CHECK-ABS-THUMB-NEXT: 1404: blx     0x1100 <arm_callee1>
// CHECK-ABS-THUMB-NEXT: 1408: b.w     0x1420 <__Thumbv7ABSLongThunk_arm_callee1>
// CHECK-ABS-THUMB-NEXT: 140c: b.w     0x142a <__Thumbv7ABSLongThunk_arm_callee2>
// CHECK-ABS-THUMB-NEXT: 1410: b.w     0x1434 <__Thumbv7ABSLongThunk_arm_callee3>
// CHECK-ABS-THUMB-NEXT: 1414: beq.w   0x1420 <__Thumbv7ABSLongThunk_arm_callee1>
// CHECK-ABS-THUMB-NEXT: 1418: beq.w   0x142a <__Thumbv7ABSLongThunk_arm_callee2>
// CHECK-ABS-THUMB-NEXT: 141c: bne.w   0x1434 <__Thumbv7ABSLongThunk_arm_callee3>
// CHECK-ABS-THUMB: <__Thumbv7ABSLongThunk_arm_callee1>:
// CHECK-ABS-THUMB-NEXT: 1420: movw    r12, #4352
// CHECK-ABS-THUMB-NEXT: 1424: movt    r12, #0
// CHECK-ABS-THUMB-NEXT: 1428: bx      r12
// CHECK-ABS-THUMB: <__Thumbv7ABSLongThunk_arm_callee2>:
// CHECK-ABS-THUMB-NEXT: 142a: movw    r12, #5632
// CHECK-ABS-THUMB-NEXT: 142e: movt    r12, #0
// CHECK-ABS-THUMB-NEXT: 1432: bx      r12
// CHECK-ABS-THUMB: <__Thumbv7ABSLongThunk_arm_callee3>:
// CHECK-ABS-THUMB-NEXT: 1434: movw    r12, #5636
// CHECK-ABS-THUMB-NEXT: 1438: movt    r12, #0
// CHECK-ABS-THUMB-NEXT: 143c: bx      r12

// CHECK-PI-THUMB: Disassembly of section .thumb_caller:
// CHECK-PI-THUMB-EMPTY:
// CHECK-PI-THUMB-NEXT: <thumb_caller>:
// CHECK-PI-THUMB-NEXT: 1400: blx     0x1100 <arm_callee1>
// CHECK-PI-THUMB-NEXT: 1404: blx     0x1100 <arm_callee1>
// CHECK-PI-THUMB-NEXT: 1408: b.w     0x1420 <__ThumbV7PILongThunk_arm_callee1>
// CHECK-PI-THUMB-NEXT: 140c: b.w     0x142c <__ThumbV7PILongThunk_arm_callee2>
// CHECK-PI-THUMB-NEXT: 1410: b.w     0x1438 <__ThumbV7PILongThunk_arm_callee3>
// CHECK-PI-THUMB-NEXT: 1414: beq.w   0x1420 <__ThumbV7PILongThunk_arm_callee1>
// CHECK-PI-THUMB-NEXT: 1418: beq.w   0x142c <__ThumbV7PILongThunk_arm_callee2>
// CHECK-PI-THUMB-NEXT: 141c: bne.w   0x1438 <__ThumbV7PILongThunk_arm_callee3>
// CHECK-PI-THUMB: <__ThumbV7PILongThunk_arm_callee1>:
// CHECK-PI-THUMB-NEXT: 1420: movw    r12, #64724
// CHECK-PI-THUMB-NEXT: 1424: movt    r12, #65535
// CHECK-PI-THUMB-NEXT: 1428: add     r12, pc
// CHECK-PI-THUMB-NEXT: 142a: bx      r12
// CHECK-PI-THUMB: <__ThumbV7PILongThunk_arm_callee2>:
// CHECK-PI-THUMB-NEXT: 142c: movw    r12, #456
// CHECK-PI-THUMB-NEXT: 1430: movt    r12, #0
// CHECK-PI-THUMB-NEXT: 1434: add     r12, pc
// CHECK-PI-THUMB-NEXT: 1436: bx      r12
// CHECK-PI-THUMB: <__ThumbV7PILongThunk_arm_callee3>:
// CHECK-PI-THUMB-NEXT: 1438: movw    r12, #448
// CHECK-PI-THUMB-NEXT: 143c: movt    r12, #0
// CHECK-PI-THUMB-NEXT: 1440: add     r12, pc
// CHECK-PI-THUMB-NEXT: 1442: bx      r12

/// Thumb calls need to change state to reach PLT
/// bl can change to blx to PLT entries, branches
/// need a state change thunk.
// CHECK-ARM-PLT: Disassembly of section .thumb_caller:
// CHECK-ARM-PLT-EMPTY:
// CHECK-ARM-PLT-NEXT: <thumb_caller>:
// CHECK-ARM-PLT-NEXT: 1400: blx     0x1640
// CHECK-ARM-PLT-NEXT: 1404: blx     0x1640
// CHECK-ARM-PLT-NEXT: 1408: b.w     0x1420 <__ThumbV7PILongThunk_arm_callee1>
// CHECK-ARM-PLT-NEXT: 140c: b.w     0x142c <__ThumbV7PILongThunk_arm_callee2>
// CHECK-ARM-PLT-NEXT: 1410: b.w     0x1438 <__ThumbV7PILongThunk_arm_callee3>
// CHECK-ARM-PLT-NEXT: 1414: beq.w   0x1420 <__ThumbV7PILongThunk_arm_callee1>
// CHECK-ARM-PLT-NEXT: 1418: beq.w   0x142c <__ThumbV7PILongThunk_arm_callee2>
// CHECK-ARM-PLT-NEXT: 141c: bne.w   0x1438 <__ThumbV7PILongThunk_arm_callee3>

/// Target Sections for thunks at a higher address than the callers.
.section .R_ARM_JUMP24_callee_high, "ax", %progbits
 .thumb
 .balign 0x100
 .globl thumb_callee2
 .type thumb_callee2, %function
thumb_callee2:
 bx lr

 .globl thumb_callee3
 .type thumb_callee3, %function
thumb_callee3:
 bx lr
// CHECK-THUMB:  Disassembly of section .R_ARM_JUMP24_callee_2:
// CHECK-THUMB-EMPTY:
// CHECK-THUMB-NEXT: <thumb_callee2>:
// CHECK-THUMB-NEXT: 1500: bx      lr
// CHECK-THUMB: <thumb_callee3>:
// CHECK-THUMB-NEXT: 1502: bx      lr

 .section .R_ARM_THM_JUMP_callee_high, "ax", %progbits
 .arm
 .balign 0x100
 .globl arm_callee2
 .type arm_callee2, %function
arm_callee2:
 bx lr
 .globl arm_callee3
 .type arm_callee3, %function
arm_callee3:
 bx lr
// CHECK-ARM: Disassembly of section .R_ARM_THM_JUMP_callee_2:
// CHECK-ARM-EMPTY:
// CHECK-ARM-NEXT: <arm_callee2>:
// CHECK-ARM-NEXT: 1600: bx      lr
// CHECK-ARM: <arm_callee3>:
// CHECK-ARM-NEXT: 1604: bx      lr

/// _start section just calls the arm and thumb calling sections
 .text
 .arm
 .globl _start
 .balign 0x100
 .type _start, %function
_start:
 bl arm_caller
 bl thumb_caller
 bx lr

// CHECK-ARM-PLT: Disassembly of section .plt:
// CHECK-ARM-PLT-EMPTY:
// CHECK-ARM-PLT-NEXT: 00001610 <$a>:
// CHECK-ARM-PLT-NEXT:     1610:             str     lr, [sp, #-4]!
// CHECK-ARM-PLT-NEXT:     1614:             add     lr, pc, #0, #12
// CHECK-ARM-PLT-NEXT:     1618:             add     lr, lr, #0, #20
// CHECK-ARM-PLT-NEXT:     161c:             ldr     pc, [lr, #672]!
// CHECK-ARM-PLT-EMPTY:
// CHECK-ARM-PLT-NEXT: 00001620 <$d>:
// CHECK-ARM-PLT-NEXT:     1620:     d4 d4 d4 d4     .word   0xd4d4d4d4
// CHECK-ARM-PLT-NEXT:     1624:     d4 d4 d4 d4     .word   0xd4d4d4d4
// CHECK-ARM-PLT-NEXT:     1628:     d4 d4 d4 d4     .word   0xd4d4d4d4
// CHECK-ARM-PLT-NEXT:     162c:     d4 d4 d4 d4     .word   0xd4d4d4d4
// CHECK-ARM-PLT-EMPTY:
// CHECK-ARM-PLT-NEXT: 00001630 <$a>:
// CHECK-ARM-PLT-NEXT:     1630:             add     r12, pc, #0, #12
// CHECK-ARM-PLT-NEXT:     1634:             add     r12, r12, #0, #20
// CHECK-ARM-PLT-NEXT:     1638:             ldr     pc, [r12, #648]!
// CHECK-ARM-PLT-EMPTY:
// CHECK-ARM-PLT-NEXT: 0000163c <$d>:
// CHECK-ARM-PLT-NEXT:     163c:     d4 d4 d4 d4     .word   0xd4d4d4d4
// CHECK-ARM-PLT-EMPTY:
// CHECK-ARM-PLT-NEXT: 00001640 <$a>:
// CHECK-ARM-PLT-NEXT:     1640:             add     r12, pc, #0, #12
// CHECK-ARM-PLT-NEXT:     1644:             add     r12, r12, #0, #20
// CHECK-ARM-PLT-NEXT:     1648:             ldr     pc, [r12, #636]!
// CHECK-ARM-PLT-EMPTY:
// CHECK-ARM-PLT-NEXT: 0000164c <$d>:
// CHECK-ARM-PLT-NEXT:     164c:     d4 d4 d4 d4     .word   0xd4d4d4d4
// CHECK-ARM-PLT-EMPTY:
// CHECK-ARM-PLT-NEXT: 00001650 <$a>:
// CHECK-ARM-PLT-NEXT:     1650:             add     r12, pc, #0, #12
// CHECK-ARM-PLT-NEXT:     1654:             add     r12, r12, #0, #20
// CHECK-ARM-PLT-NEXT:     1658:             ldr     pc, [r12, #624]!
// CHECK-ARM-PLT-EMPTY:
// CHECK-ARM-PLT-NEXT: 0000165c <$d>:
// CHECK-ARM-PLT-NEXT:     165c:     d4 d4 d4 d4     .word   0xd4d4d4d4
// CHECK-ARM-PLT-EMPTY:
// CHECK-ARM-PLT-NEXT: 00001660 <$a>:
// CHECK-ARM-PLT-NEXT:     1660:             add     r12, pc, #0, #12
// CHECK-ARM-PLT-NEXT:     1664:             add     r12, r12, #0, #20
// CHECK-ARM-PLT-NEXT:     1668:             ldr     pc, [r12, #612]!
// CHECK-ARM-PLT-EMPTY:
// CHECK-ARM-PLT-NEXT: 0000166c <$d>:
// CHECK-ARM-PLT-NEXT:     166c:     d4 d4 d4 d4     .word   0xd4d4d4d4
// CHECK-ARM-PLT-EMPTY:
// CHECK-ARM-PLT-NEXT: 00001670 <$a>:
// CHECK-ARM-PLT-NEXT:     1670:             add     r12, pc, #0, #12
// CHECK-ARM-PLT-NEXT:     1674:             add     r12, r12, #0, #20
// CHECK-ARM-PLT-NEXT:     1678:             ldr     pc, [r12, #600]!
// CHECK-ARM-PLT-EMPTY:
// CHECK-ARM-PLT-NEXT: 0000167c <$d>:
// CHECK-ARM-PLT-NEXT:     167c:     d4 d4 d4 d4     .word   0xd4d4d4d4
// CHECK-ARM-PLT-EMPTY:
// CHECK-ARM-PLT-NEXT: 00001680 <$a>:
// CHECK-ARM-PLT-NEXT:     1680:             add     r12, pc, #0, #12
// CHECK-ARM-PLT-NEXT:     1684:             add     r12, r12, #0, #20
// CHECK-ARM-PLT-NEXT:     1688:             ldr     pc, [r12, #588]!
// CHECK-ARM-PLT-EMPTY:
// CHECK-ARM-PLT-NEXT: 0000168c <$d>:
// CHECK-ARM-PLT-NEXT:     168c:     d4 d4 d4 d4     .word   0xd4d4d4d4
// CHECK-ARM-PLT-EMPTY:
// CHECK-ARM-PLT-NEXT: 00001690 <$a>:
// CHECK-ARM-PLT-NEXT:     1690:             add     r12, pc, #0, #12
// CHECK-ARM-PLT-NEXT:     1694:             add     r12, r12, #0, #20
// CHECK-ARM-PLT-NEXT:     1698:             ldr     pc, [r12, #576]!
// CHECK-ARM-PLT-EMPTY:
// CHECK-ARM-PLT-NEXT: 0000169c <$d>:
// CHECK-ARM-PLT-NEXT:     169c:     d4 d4 d4 d4     .word   0xd4d4d4d4
// CHECK-ARM-PLT-EMPTY:
// CHECK-ARM-PLT-NEXT: 000016a0 <$a>:
// CHECK-ARM-PLT-NEXT:     16a0:             add     r12, pc, #0, #12
// CHECK-ARM-PLT-NEXT:     16a4:             add     r12, r12, #0, #20
// CHECK-ARM-PLT-NEXT:     16a8:             ldr     pc, [r12, #564]!
// CHECK-ARM-PLT-EMPTY:
// CHECK-ARM-PLT-NEXT: 000016ac <$d>:
// CHECK-ARM-PLT-NEXT:     16ac:     d4 d4 d4 d4     .word   0xd4d4d4d4

// CHECK-DSO-REL:      0x18C0 R_ARM_JUMP_SLOT thumb_callee1
// CHECK-DSO-REL-NEXT: 0x18C4 R_ARM_JUMP_SLOT arm_callee1
// CHECK-DSO-REL-NEXT: 0x18C8 R_ARM_JUMP_SLOT arm_caller
// CHECK-DSO-REL-NEXT: 0x18CC R_ARM_JUMP_SLOT thumb_callee2
// CHECK-DSO-REL-NEXT: 0x18D0 R_ARM_JUMP_SLOT thumb_callee3
// CHECK-DSO-REL-NEXT: 0x18D4 R_ARM_JUMP_SLOT arm_callee2
// CHECK-DSO-REL-NEXT: 0x18D8 R_ARM_JUMP_SLOT arm_callee3
// CHECK-DSO-REL-NEXT: 0x18DC R_ARM_JUMP_SLOT thumb_caller
