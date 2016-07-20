// RUN: llvm-mc -filetype=obj -triple=thumbv7a-none-linux-gnueabi %p/Inputs/arm-plt-reloc.s -o %t1
// RUN: llvm-mc -filetype=obj -triple=thumbv7a-none-linux-gnueabi %s -o %t2
// RUN: ld.lld %t1 %t2 -o %t
// RUN: llvm-objdump -triple=thumbv7a-none-linux-gnueabi -d %t | FileCheck %s
// RUN: ld.lld -shared %t1 %t2 -o %t3
// RUN: llvm-objdump -triple=thumbv7a-none-linux-gnueabi -d %t3 | FileCheck -check-prefix=DSOTHUMB %s
// RUN: llvm-objdump -triple=armv7a-none-linux-gnueabi -d %t3 | FileCheck -check-prefix=DSOARM %s
// RUN: llvm-readobj -s -r %t3 | FileCheck -check-prefix=DSOREL %s
// REQUIRES: arm
//
// Test PLT entry generation
 .syntax unified
 .text
 .align 2
 .globl _start
 .type  _start,%function
_start:
// FIXME, interworking is only supported for BL via BLX at the moment, when
// interworking thunks are available for b.w and b<cond>.w this can be altered
// to test the different forms of interworking.
 bl func1
 bl func2
 bl func3

// Executable, expect no PLT
// CHECK: Disassembly of section .text:
// CHECK-NEXT: func1:
// CHECK-NEXT:   11000: 70 47   bx      lr
// CHECK: func2:
// CHECK-NEXT:   11002: 70 47   bx      lr
// CHECK: func3:
// CHECK-NEXT:   11004: 70 47   bx      lr
// CHECK-NEXT:   11006: 00 00   movs    r0, r0
// CHECK: _start:
// 11008 + 4 -12 = 0x11000 = func1
// CHECK-NEXT:   11008: ff f7 fa ff     bl      #-12
// 1100c + 4 -14 = 0x11002 = func2
// CHECK-NEXT:   1100c: ff f7 f9 ff     bl      #-14
// 11010 + 4 -16 = 0x11004 = func3
// CHECK-NEXT:   11010: ff f7 f8 ff     bl      #-16

// Expect PLT entries as symbols can be preempted
// .text is Thumb and .plt is ARM, llvm-objdump can currently only disassemble
// as ARM or Thumb. Work around by disassembling twice.
// DSOTHUMB: Disassembly of section .text:
// DSOTHUMB: func1:
// DSOTHUMB-NEXT:    1000:       70 47   bx      lr
// DSOTHUMB: func2:
// DSOTHUMB-NEXT:    1002:       70 47   bx      lr
// DSOTHUMB: func3:
// DSOTHUMB-NEXT:    1004:       70 47   bx      lr
// DSOTHUMB-NEXT:    1006:       00 00   movs    r0, r0
// DSOTHUMB: _start:
// 0x1008 + 0x28 + 4 = 0x1034 = PLT func1
// DSOTHUMB-NEXT:    1008:       00 f0 14 e8     blx     #40
// 0x100c + 0x34 + 4 = 0x1044 = PLT func2
// DSOTHUMB-NEXT:    100c:       00 f0 1a e8     blx     #52
// 0x1010 + 0x40 + 4 = 0x1054 = PLT func3
// DSOTHUMB-NEXT:    1010:       00 f0 20 e8     blx     #64
// DSOARM: Disassembly of section .plt:
// DSOARM: .plt:
// DSOARM-NEXT:    1020:       04 e0 2d e5     str     lr, [sp, #-4]!
// DSOARM-NEXT:    1024:       04 e0 9f e5     ldr     lr, [pc, #4]
// DSOARM-NEXT:    1028:       0e e0 8f e0     add     lr, pc, lr
// DSOARM-NEXT:    102c:       08 f0 be e5     ldr     pc, [lr, #8]!
// DSOARM-NEXT:    1030:       d0 1f 00 00
// 0x1028 + 8 + 1fd0 = 0x3000
// DSOARM-NEXT:    1034:       04 c0 9f e5     ldr     r12, [pc, #4]
// DSOARM-NEXT:    1038:       0f c0 8c e0     add     r12, r12, pc
// DSOARM-NEXT:    103c:       00 f0 9c e5     ldr     pc, [r12]
// DSOARM-NEXT:    1040:       cc 1f 00 00
// 0x1038 + 8 + 1fcc = 0x300c
// DSOARM-NEXT:    1044:       04 c0 9f e5     ldr     r12, [pc, #4]
// DSOARM-NEXT:    1048:       0f c0 8c e0     add     r12, r12, pc
// DSOARM-NEXT:    104c:       00 f0 9c e5     ldr     pc, [r12]
// DSOARM-NEXT:    1050:       c0 1f 00 00
// 0x1048 + 8 + 1fc0 = 0x3010
// DSOARM-NEXT:    1054:       04 c0 9f e5     ldr     r12, [pc, #4]
// DSOARM-NEXT:    1058:       0f c0 8c e0     add     r12, r12, pc
// DSOARM-NEXT:    105c:       00 f0 9c e5     ldr     pc, [r12]
// DSOARM-NEXT:    1060:       b4 1f 00 00
// 0x1058 + 8 + 1fb4 = 0x3014

// DSOREL:    Name: .got.plt
// DSOREL-NEXT:    Type: SHT_PROGBITS
// DSOREL-NEXT:    Flags [
// DSOREL-NEXT:      SHF_ALLOC
// DSOREL-NEXT:      SHF_WRITE
// DSOREL-NEXT:    ]
// DSOREL-NEXT:    Address: 0x3000
// DSOREL-NEXT:    Offset:
// DSOREL-NEXT:    Size: 24
// DSOREL-NEXT:    Link:
// DSOREL-NEXT:    Info:
// DSOREL-NEXT:    AddressAlignment: 4
// DSOREL-NEXT:    EntrySize:
// DSOREL:  Relocations [
// DSOREL-NEXT:  Section (4) .rel.plt {
// DSOREL-NEXT:    0x300C R_ARM_JUMP_SLOT func1 0x0
// DSOREL-NEXT:    0x3010 R_ARM_JUMP_SLOT func2 0x0
// DSOREL-NEXT:    0x3014 R_ARM_JUMP_SLOT func3 0x0
