// REQUIRES: arm
// RUN: llvm-mc -filetype=obj -triple=thumbv7a-none-linux-gnueabi %p/Inputs/arm-plt-reloc.s -o %t1
// RUN: llvm-mc -filetype=obj -triple=thumbv7a-none-linux-gnueabi %s -o %t2
// RUN: ld.lld %t1 %t2 -o %t
// RUN: llvm-objdump --no-print-imm-hex --triple=thumbv7a-none-linux-gnueabi -d %t | FileCheck %s
// RUN: ld.lld -shared %t1 %t2 -o %t.so
// RUN: llvm-objdump --no-print-imm-hex --triple=thumbv7a-none-linux-gnueabi -d %t.so | FileCheck --check-prefix=DSO %s
// RUN: llvm-readobj -S -r %t.so | FileCheck -check-prefix=DSOREL %s

// RUN: llvm-mc -filetype=obj -triple=thumbv7aeb-none-linux-gnueabi %p/Inputs/arm-plt-reloc.s -o %t1
// RUN: llvm-mc -filetype=obj -triple=thumbv7aeb-none-linux-gnueabi %s -o %t2
// RUN: ld.lld %t1 %t2 -o %t
// RUN: llvm-objdump --no-print-imm-hex --triple=thumbv7aeb-none-linux-gnueabi -d %t | FileCheck %s
// RUN: ld.lld -shared %t1 %t2 -o %t.so
// RUN: llvm-objdump --no-print-imm-hex --triple=thumbv7aeb-none-linux-gnueabi -d %t.so | FileCheck --check-prefix=DSO %s
// RUN: llvm-readobj -S -r %t.so | FileCheck -check-prefix=DSOREL %s

// RUN: ld.lld --be8 %t1 %t2 -o %t
// RUN: llvm-objdump --no-print-imm-hex --triple=thumbv7aeb-none-linux-gnueabi -d %t | FileCheck %s
// RUN: ld.lld --be8 -shared %t1 %t2 -o %t.so
// RUN: llvm-objdump --no-print-imm-hex --triple=thumbv7aeb-none-linux-gnueabi -d %t.so | FileCheck --check-prefix=DSO %s
// RUN: llvm-readobj -S -r %t.so | FileCheck -check-prefix=DSOREL %s

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
// CHECK-NEXT:   200b4: 4770    bx      lr
// CHECK: <func2>:
// CHECK-NEXT:   200b6: 4770    bx      lr
// CHECK: <func3>:
// CHECK-NEXT:   200b8: 4770    bx      lr
// CHECK-NEXT:   200ba: d4d4 
// CHECK: <_start>:
// CHECK-NEXT:   200bc: f7ff fffa       bl      0x200b4 <func1>
// CHECK-NEXT:   200c0: f7ff fff9       bl      0x200b6 <func2>
// CHECK-NEXT:   200c4: f7ff fff8       bl      0x200b8 <func3>

// Expect PLT entries as symbols can be preempted
// .text is Thumb and .plt is ARM, llvm-objdump can currently only disassemble
// as ARM or Thumb. Work around by disassembling twice.
// DSO: Disassembly of section .text:
// DSO-EMPTY:
// DSO-NEXT: <func1>:
// DSO-NEXT:     10214:     4770    bx      lr
// DSO: <func2>:
// DSO-NEXT:     10216:     4770    bx      lr
// DSO: <func3>:
// DSO-NEXT:     10218:     4770    bx      lr
// DSO-NEXT:     1021a:     d4d4 
// DSO: <_start>:
// 0x10250 = PLT func1
// DSO-NEXT:     1021c:     f000 e818       blx     0x10250
// 0x10260 = PLT func2
// DSO-NEXT:     10220:     f000 e81e       blx     0x10260
// 0x10270 = PLT func3
// DSO-NEXT:     10224:     f000 e824       blx     0x10270
// DSO: Disassembly of section .plt:
// DSO-EMPTY:
// DSO-NEXT: <$a>:
// DSO-NEXT:     10230:       e52de004        str     lr, [sp, #-4]!
// (0x10234 + 8) + (0 RoR 12) + (32 RoR 20 = 0x20000) + 164 = 0x302e0 = .got.plt[2]
// DSO-NEXT:     10234:       e28fe600        add     lr, pc, #0, #12
// DSO-NEXT:     10238:       e28eea20        add     lr, lr, #32, #20
// DSO-NEXT:     1023c:       e5bef0a4        ldr     pc, [lr, #164]!
// DSO: <$d>:

// DSO-NEXT:     10240:       d4 d4 d4 d4     .word   0xd4d4d4d4
// DSO-NEXT:     10244:       d4 d4 d4 d4     .word   0xd4d4d4d4
// DSO-NEXT:     10248:       d4 d4 d4 d4     .word   0xd4d4d4d4
// DSO-NEXT:     1024c:       d4 d4 d4 d4     .word   0xd4d4d4d4
// DSO: <$a>:
// (0x10250 + 8) + (0 RoR 12) + (32 RoR 20 = 0x20000) + 140 = 0x302e4
// DSO-NEXT:     10250:       e28fc600        add     r12, pc, #0, #12
// DSO-NEXT:     10254:       e28cca20        add     r12, r12, #32, #20
// DSO-NEXT:     10258:       e5bcf08c        ldr     pc, [r12, #140]!
// DSO: <$d>:
// DSO-NEXT:     1025c:       d4 d4 d4 d4     .word   0xd4d4d4d4
// DSO: <$a>:
// (0x10260 + 8) + (0 RoR 12) + (32 RoR 20 = 0x20000) + 128 = 0x302e8
// DSO-NEXT:     10260:       e28fc600        add     r12, pc, #0, #12
// DSO-NEXT:     10264:       e28cca20        add     r12, r12, #32, #20
// DSO-NEXT:     10268:       e5bcf080        ldr     pc, [r12, #128]!
// DSO: <$d>:
// DSO-NEXT:     1026c:       d4 d4 d4 d4     .word   0xd4d4d4d4
// DSO: <$a>:
// (0x10270 + 8) + (0 RoR 12) + (32 RoR 20 = 0x20000) + 116 = 0x302ec
// DSO-NEXT:     10270:       e28fc600        add     r12, pc, #0, #12
// DSO-NEXT:     10274:       e28cca20        add     r12, r12, #32, #20
// DSO-NEXT:     10278:       e5bcf074        ldr     pc, [r12, #116]!
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
