// REQUIRES: arm
// RUN: llvm-mc -filetype=obj -triple=thumbv7a-none-linux-gnueabi %p/Inputs/arm-plt-reloc.s -o %t1
// RUN: llvm-mc -filetype=obj -triple=thumbv7a-none-linux-gnueabi %s -o %t2
// RUN: ld.lld %t1 %t2 -o %t
// RUN: llvm-objdump --triple=thumbv7a-none-linux-gnueabi -d %t | FileCheck %s
// RUN: ld.lld -shared %t1 %t2 -o %t.so
// RUN: llvm-objdump --triple=thumbv7a-none-linux-gnueabi -d %t.so | FileCheck --check-prefix=DSO %s
// RUN: llvm-readobj -S -r %t.so | FileCheck -check-prefix=DSOREL %s
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
// CHECK-EMPTY:
// CHECK-NEXT: <func1>:
// CHECK-NEXT:   200b4: 70 47   bx      lr
// CHECK: <func2>:
// CHECK-NEXT:   200b6: 70 47   bx      lr
// CHECK: <func3>:
// CHECK-NEXT:   200b8: 70 47   bx      lr
// CHECK-NEXT:   200ba: d4 d4
// CHECK: <_start>:
// . + 4 -12 = 0x200b4 = func1
// CHECK-NEXT:   200bc: ff f7 fa ff     bl      #-12
// . + 4 -14 = 0x200b6 = func2
// CHECK-NEXT:   200c0: ff f7 f9 ff     bl      #-14
// . + 4 -16 = 0x200b8 = func3
// CHECK-NEXT:   200c4: ff f7 f8 ff     bl      #-16

// Expect PLT entries as symbols can be preempted
// .text is Thumb and .plt is ARM, llvm-objdump can currently only disassemble
// as ARM or Thumb. Work around by disassembling twice.
// DSO: Disassembly of section .text:
// DSO-EMPTY:
// DSO-NEXT: <func1>:
// DSO-NEXT:     10214:     70 47   bx      lr
// DSO: <func2>:
// DSO-NEXT:     10216:     70 47   bx      lr
// DSO: <func3>:
// DSO-NEXT:     10218:     70 47   bx      lr
// DSO-NEXT:     1021a:     d4 d4   bmi     #-88
// DSO: <_start>:
// . + 48 + 4 = 0x10250 = PLT func1
// DSO-NEXT:     1021c:     00 f0 18 e8     blx     #48
// . + 60 + 4 = 0x10260 = PLT func2
// DSO-NEXT:     10220:     00 f0 1e e8     blx     #60
// . + 72 + 4 = 0x10270 = PLT func3
// DSO-NEXT:     10224:     00 f0 24 e8     blx     #72
// DSO: Disassembly of section .plt:
// DSO-EMPTY:
// DSO-NEXT: <$a>:
// DSO-NEXT:     10230:       04 e0 2d e5     str     lr, [sp, #-4]!
// (0x10234 + 8) + (0 RoR 12) + 8192 + 164 = 0x32e0 = .got.plt[3]
// DSO-NEXT:     10234:       00 e6 8f e2     add     lr, pc, #0, #12
// DSO-NEXT:     10238:       20 ea 8e e2     add     lr, lr, #32
// DSO-NEXT:     1023c:       a4 f0 be e5     ldr     pc, [lr, #164]!
// DSO: <$d>:

// DSO-NEXT:     10240:       d4 d4 d4 d4     .word   0xd4d4d4d4
// DSO-NEXT:     10244:       d4 d4 d4 d4     .word   0xd4d4d4d4
// DSO-NEXT:     10248:       d4 d4 d4 d4     .word   0xd4d4d4d4
// DSO-NEXT:     1024c:       d4 d4 d4 d4     .word   0xd4d4d4d4
// DSO: <$a>:
// (0x10250 + 8) + (0 RoR 12) + 8192 + 140 = 0x32e4
// DSO-NEXT:     10250:       00 c6 8f e2     add     r12, pc, #0, #12
// DSO-NEXT:     10254:       20 ca 8c e2     add     r12, r12, #32
// DSO-NEXT:     10258:       8c f0 bc e5     ldr     pc, [r12, #140]!
// DSO: <$d>:
// DSO-NEXT:     1025c:       d4 d4 d4 d4     .word   0xd4d4d4d4
// DSO: <$a>:
// (0x10260 + 8) + (0 RoR 12) + 8192 + 128 = 0x32e8
// DSO-NEXT:     10260:       00 c6 8f e2     add     r12, pc, #0, #12
// DSO-NEXT:     10264:       20 ca 8c e2     add     r12, r12, #32
// DSO-NEXT:     10268:       80 f0 bc e5     ldr     pc, [r12, #128]!
// DSO: <$d>:
// DSO-NEXT:     1026c:       d4 d4 d4 d4     .word   0xd4d4d4d4
// DSO: <$a>:
// (0x10270 + 8) + (0 RoR 12) + 8192 + 116 = 0x32ec
// DSO-NEXT:     10270:       00 c6 8f e2     add     r12, pc, #0, #12
// DSO-NEXT:     10274:       20 ca 8c e2     add     r12, r12, #32
// DSO-NEXT:     10278:       74 f0 bc e5     ldr     pc, [r12, #116]!
// DSO: <$d>:
// DSO-NEXT:     1027c:       d4 d4 d4 d4     .word   0xd4d4d4d4

// DSOREL:    Name: .got.plt
// DSOREL-NEXT:    Type: SHT_PROGBITS
// DSOREL-NEXT:    Flags [
// DSOREL-NEXT:      SHF_ALLOC
// DSOREL-NEXT:      SHF_WRITE
// DSOREL-NEXT:    ]
// DSOREL-NEXT:    Address: 0x302D8
// DSOREL-NEXT:    Offset:
// DSOREL-NEXT:    Size: 24
// DSOREL-NEXT:    Link:
// DSOREL-NEXT:    Info:
// DSOREL-NEXT:    AddressAlignment: 4
// DSOREL-NEXT:    EntrySize:
// DSOREL:  Relocations [
// DSOREL-NEXT:  Section (5) .rel.plt {
// DSOREL-NEXT:    0x302E4 R_ARM_JUMP_SLOT func1
// DSOREL-NEXT:    0x302E8 R_ARM_JUMP_SLOT func2
// DSOREL-NEXT:    0x302EC R_ARM_JUMP_SLOT func3
