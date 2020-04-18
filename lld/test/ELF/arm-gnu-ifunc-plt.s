// REQUIRES: arm
// RUN: llvm-mc -filetype=obj -triple=armv7a-linux-gnueabihf %S/Inputs/arm-shared.s -o %t1.o
// RUN: ld.lld %t1.o --shared -soname=t.so -o %t.so
// RUN: llvm-mc -filetype=obj -triple=armv7a-linux-gnueabihf %s -o %t.o
// RUN: ld.lld %t.so %t.o -o %tout
// RUN: llvm-objdump --triple=armv7a-linux-gnueabihf -d --no-show-raw-insn %tout | FileCheck %s --check-prefix=DISASM
// RUN: llvm-objdump -s %tout | FileCheck %s --check-prefix=GOTPLT
// RUN: llvm-readobj -r --dynamic-table %tout | FileCheck %s

// Check that the IRELATIVE relocations are last in the .got
// CHECK: Relocations [
// CHECK-NEXT:   Section (5) .rel.dyn {
// CHECK-NEXT:     0x302E0 R_ARM_GLOB_DAT bar2 0x0
// CHECK-NEXT:     0x302E4 R_ARM_GLOB_DAT zed2 0x0
// CHECK-NEXT:     0x302E8 R_ARM_IRELATIVE - 0x0
// CHECK-NEXT:     0x302EC R_ARM_IRELATIVE - 0x0
// CHECK-NEXT:   }
// CHECK-NEXT:   Section (6) .rel.plt {
// CHECK-NEXT:     0x402FC R_ARM_JUMP_SLOT bar2 0x0
// CHECK-NEXT:     0x40300 R_ARM_JUMP_SLOT zed2 0x0
// CHECK-NEXT:   }
// CHECK-NEXT: ]

// Check that the GOT entries refer back to the ifunc resolver
// GOTPLT: Contents of section .got:
// GOTPLT-NEXT:  302e0 00000000 00000000 dc010200 e0010200
// GOTPLT: Contents of section .got.plt:
// GOTPLT-NEXT:  402f0 00000000 00000000 00000000 00020200
// GOTPLT-NEXT:  40300 00020200

// DISASM: Disassembly of section .text:
// DISASM-EMPTY:
// DISASM-NEXT: <foo>:
// DISASM-NEXT:    201dc:       bx      lr
// DISASM: <bar>:
// DISASM-NEXT:    201e0:       bx      lr
// DISASM: <_start>:
// DISASM-NEXT:    201e4:       bl      #84
// DISASM-NEXT:    201e8:       bl      #96
// DISASM: <$d.1>:
// DISASM-NEXT:    201ec:       00 00 00 00     .word   0x00000000
// DISASM-NEXT:    201f0:       04 00 00 00     .word   0x00000004
// DISASM:         201f4:       bl      #36
// DISASM-NEXT:    201f8:       bl      #48
// DISASM-EMPTY:
// DISASM-NEXT: Disassembly of section .plt:
// DISASM-EMPTY:
// DISASM-NEXT: <$a>:
// DISASM-NEXT:    20200:       str     lr, [sp, #-4]!
// DISASM-NEXT:    20204:       add     lr, pc, #0, #12
// DISASM-NEXT:    20208:       add     lr, lr, #32
// DISASM-NEXT:    2020c:       ldr     pc, [lr, #236]!
// DISASM: <$d>:
// DISASM-NEXT:    20210:       d4 d4 d4 d4     .word   0xd4d4d4d4
// DISASM-NEXT:    20214:       d4 d4 d4 d4     .word   0xd4d4d4d4
// DISASM-NEXT:    20218:       d4 d4 d4 d4     .word   0xd4d4d4d4
// DISASM-NEXT:    2021c:       d4 d4 d4 d4     .word   0xd4d4d4d4
// DISASM: <$a>:
// DISASM-NEXT:    20220:       add     r12, pc, #0, #12
// DISASM-NEXT:    20224:       add     r12, r12, #32
// DISASM-NEXT:    20228:       ldr     pc, [r12, #212]!
// DISASM: <$d>:
// DISASM-NEXT:    2022c:       d4 d4 d4 d4     .word   0xd4d4d4d4
// DISASM: <$a>:
// DISASM-NEXT:    20230:       add     r12, pc, #0, #12
// DISASM-NEXT:    20234:       add     r12, r12, #32
// DISASM-NEXT:    20238:       ldr     pc, [r12, #200]!
// DISASM: <$d>:
// DISASM-NEXT:    2023c:       d4 d4 d4 d4     .word   0xd4d4d4d4
// DISASM: <$a>:
// DISASM-NEXT:    20240:       add     r12, pc, #0, #12
// DISASM-NEXT:    20244:       add     r12, r12, #16
// DISASM-NEXT:    20248:       ldr     pc, [r12, #160]!
// DISASM: <$d>:
// DISASM-NEXT:    2024c:       d4 d4 d4 d4     .word   0xd4d4d4d4
// DISASM: <$a>:
// DISASM-NEXT:    20250:       add     r12, pc, #0, #12
// DISASM-NEXT:    20254:       add     r12, r12, #16
// DISASM-NEXT:    20258:       ldr     pc, [r12, #148]!
// DISASM: <$d>:
// DISASM-NEXT:    2025c:	d4 d4 d4 d4 	.word	0xd4d4d4d4

.syntax unified
.text
.type foo STT_GNU_IFUNC
.globl foo
foo:
 bx lr

.type bar STT_GNU_IFUNC
.globl bar
bar:
 bx lr

.globl _start
_start:
 bl foo
 bl bar
 // Create entries in the .got and .rel.dyn so that we don't just have
 // IRELATIVE
 .word bar2(got)
 .word zed2(got)
 bl bar2
 bl zed2
