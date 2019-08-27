// REQUIRES: arm
// RUN: llvm-mc -filetype=obj -triple=thumbv7a-none-linux-gnueabi %p/Inputs/arm-plt-reloc.s -o %t1
// RUN: llvm-mc -filetype=obj -triple=thumbv7a-none-linux-gnueabi %s -o %t2
// RUN: ld.lld %t1 %t2 -o %t
// RUN: llvm-objdump -triple=thumbv7a-none-linux-gnueabi -d %t | FileCheck %s
// RUN: ld.lld -shared %t1 %t2 -o %t.so
// RUN: llvm-objdump -triple=thumbv7a-none-linux-gnueabi -d %t.so | FileCheck -check-prefix=DSO %s
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
// CHECK-NEXT: func1:
// CHECK-NEXT:   110b4: 70 47   bx      lr
// CHECK: func2:
// CHECK-NEXT:   110b6: 70 47   bx      lr
// CHECK: func3:
// CHECK-NEXT:   110b8: 70 47   bx      lr
// CHECK-NEXT:   110ba: d4 d4
// CHECK: _start:
// . + 4 -12 = 0x110b4 = func1
// CHECK-NEXT:   110bc: ff f7 fa ff     bl      #-12
// . + 4 -14 = 0x110b6 = func2
// CHECK-NEXT:   110c0: ff f7 f9 ff     bl      #-14
// . + 4 -16 = 0x110b8 = func3
// CHECK-NEXT:   110c4: ff f7 f8 ff     bl      #-16

// Expect PLT entries as symbols can be preempted
// .text is Thumb and .plt is ARM, llvm-objdump can currently only disassemble
// as ARM or Thumb. Work around by disassembling twice.
// DSO: Disassembly of section .text:
// DSO-EMPTY:
// DSO-NEXT: func1:
// DSO-NEXT:     1214:     70 47   bx      lr
// DSO: func2:
// DSO-NEXT:     1216:     70 47   bx      lr
// DSO: func3:
// DSO-NEXT:     1218:     70 47   bx      lr
// DSO-NEXT:     121a:     d4 d4   bmi     #-88
// DSO: _start:
// . + 48 + 4 = 0x1250 = PLT func1
// DSO-NEXT:     121c:     00 f0 18 e8     blx     #48
// . + 60 + 4 = 0x1260 = PLT func2
// DSO-NEXT:     1220:     00 f0 1e e8     blx     #60
// . + 72 + 4 = 0x1270 = PLT func3
// DSO-NEXT:     1224:     00 f0 24 e8     blx     #72
// DSO: Disassembly of section .plt:
// DSO-EMPTY:
// DSO-NEXT: $a:
// DSO-NEXT:     1230:       04 e0 2d e5     str     lr, [sp, #-4]!
// (0x1234 + 8) + (0 RoR 12) + 8192 + 164 = 0x32e0 = .got.plt[3]
// DSO-NEXT:     1234:       00 e6 8f e2     add     lr, pc, #0, #12
// DSO-NEXT:     1238:       02 ea 8e e2     add     lr, lr, #8192
// DSO-NEXT:     123c:       a4 f0 be e5     ldr     pc, [lr, #164]!
// DSO: $d:

// DSO-NEXT:     1240:       d4 d4 d4 d4     .word   0xd4d4d4d4
// DSO-NEXT:     1244:       d4 d4 d4 d4     .word   0xd4d4d4d4
// DSO-NEXT:     1248:       d4 d4 d4 d4     .word   0xd4d4d4d4
// DSO-NEXT:     124c:       d4 d4 d4 d4     .word   0xd4d4d4d4
// DSO: $a:
// (0x1250 + 8) + (0 RoR 12) + 8192 + 140 = 0x32e4
// DSO-NEXT:     1250:       00 c6 8f e2     add     r12, pc, #0, #12
// DSO-NEXT:     1254:       02 ca 8c e2     add     r12, r12, #8192
// DSO-NEXT:     1258:       8c f0 bc e5     ldr     pc, [r12, #140]!
// DSO: $d:
// DSO-NEXT:     125c:       d4 d4 d4 d4     .word   0xd4d4d4d4
// DSO: $a:
// (0x1260 + 8) + (0 RoR 12) + 8192 + 128 = 0x32e8
// DSO-NEXT:     1260:       00 c6 8f e2     add     r12, pc, #0, #12
// DSO-NEXT:     1264:       02 ca 8c e2     add     r12, r12, #8192
// DSO-NEXT:     1268:       80 f0 bc e5     ldr     pc, [r12, #128]!
// DSO: $d:
// DSO-NEXT:     126c:       d4 d4 d4 d4     .word   0xd4d4d4d4
// DSO: $a:
// (0x1270 + 8) + (0 RoR 12) + 8192 + 116 = 0x32ec
// DSO-NEXT:     1270:       00 c6 8f e2     add     r12, pc, #0, #12
// DSO-NEXT:     1274:       02 ca 8c e2     add     r12, r12, #8192
// DSO-NEXT:     1278:       74 f0 bc e5     ldr     pc, [r12, #116]!
// DSO: $d:
// DSO-NEXT:     127c:       d4 d4 d4 d4     .word   0xd4d4d4d4

// DSOREL:    Name: .got.plt
// DSOREL-NEXT:    Type: SHT_PROGBITS
// DSOREL-NEXT:    Flags [
// DSOREL-NEXT:      SHF_ALLOC
// DSOREL-NEXT:      SHF_WRITE
// DSOREL-NEXT:    ]
// DSOREL-NEXT:    Address: 0x32D8
// DSOREL-NEXT:    Offset:
// DSOREL-NEXT:    Size: 24
// DSOREL-NEXT:    Link:
// DSOREL-NEXT:    Info:
// DSOREL-NEXT:    AddressAlignment: 4
// DSOREL-NEXT:    EntrySize:
// DSOREL:  Relocations [
// DSOREL-NEXT:  Section (5) .rel.plt {
// DSOREL-NEXT:    0x32E4 R_ARM_JUMP_SLOT func1 0x0
// DSOREL-NEXT:    0x32E8 R_ARM_JUMP_SLOT func2 0x0
// DSOREL-NEXT:    0x32EC R_ARM_JUMP_SLOT func3 0x0
