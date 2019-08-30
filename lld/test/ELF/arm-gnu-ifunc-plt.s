// REQUIRES: arm
// RUN: llvm-mc -filetype=obj -triple=armv7a-linux-gnueabihf %S/Inputs/arm-shared.s -o %t1.o
// RUN: ld.lld %t1.o --shared -soname=t.so -o %t.so
// RUN: llvm-mc -filetype=obj -triple=armv7a-linux-gnueabihf %s -o %t.o
// RUN: ld.lld %t.so %t.o -o %tout
// RUN: llvm-objdump -triple=armv7a-linux-gnueabihf -d --no-show-raw-insn %tout | FileCheck %s --check-prefix=DISASM
// RUN: llvm-objdump -s %tout | FileCheck %s --check-prefix=GOTPLT
// RUN: llvm-readobj -r --dynamic-table %tout | FileCheck %s

// Check that the IRELATIVE relocations are last in the .got
// CHECK: Relocations [
// CHECK-NEXT:   Section (5) .rel.dyn {
// CHECK-NEXT:     0x122E0 R_ARM_GLOB_DAT bar2 0x0
// CHECK-NEXT:     0x122E4 R_ARM_GLOB_DAT zed2 0x0
// CHECK-NEXT:     0x122E8 R_ARM_IRELATIVE - 0x0
// CHECK-NEXT:     0x122EC R_ARM_IRELATIVE - 0x0
// CHECK-NEXT:   }
// CHECK-NEXT:   Section (6) .rel.plt {
// CHECK-NEXT:     0x132FC R_ARM_JUMP_SLOT bar2 0x0
// CHECK-NEXT:     0x13300 R_ARM_JUMP_SLOT zed2 0x0
// CHECK-NEXT:   }
// CHECK-NEXT: ]

// Check that the GOT entries refer back to the ifunc resolver
// GOTPLT: Contents of section .got:
// GOTPLT-NEXT:  122e0 00000000 00000000 dc110100 e0110100
// GOTPLT: Contents of section .got.plt:
// GOTPLT-NEXT:  132f0 00000000 00000000 00000000 00120100
// GOTPLT-NEXT:  13300 00120100

// DISASM: Disassembly of section .text:
// DISASM-EMPTY:
// DISASM-NEXT: foo:
// DISASM-NEXT:    111dc:       bx      lr
// DISASM: bar:
// DISASM-NEXT:    111e0:       bx      lr
// DISASM: _start:
// DISASM-NEXT:    111e4:       bl      #84
// DISASM-NEXT:    111e8:       bl      #96
// DISASM: $d.1:
// DISASM-NEXT:    111ec:       00 00 00 00     .word   0x00000000
// DISASM-NEXT:    111f0:       04 00 00 00     .word   0x00000004
// DISASM:         111f4:       bl      #36
// DISASM-NEXT:    111f8:       bl      #48
// DISASM-EMPTY:
// DISASM-NEXT: Disassembly of section .plt:
// DISASM-EMPTY:
// DISASM-NEXT: $a:
// DISASM-NEXT:    11200:       str     lr, [sp, #-4]!
// DISASM-NEXT:    11204:       add     lr, pc, #0, #12
// DISASM-NEXT:    11208:       add     lr, lr, #8192
// DISASM-NEXT:    1120c:       ldr     pc, [lr, #236]!
// DISASM: $d:
// DISASM-NEXT:    11210:       d4 d4 d4 d4     .word   0xd4d4d4d4
// DISASM-NEXT:    11214:       d4 d4 d4 d4     .word   0xd4d4d4d4
// DISASM-NEXT:    11218:       d4 d4 d4 d4     .word   0xd4d4d4d4
// DISASM-NEXT:    1121c:       d4 d4 d4 d4     .word   0xd4d4d4d4
// DISASM: $a:
// DISASM-NEXT:    11220:       add     r12, pc, #0, #12
// DISASM-NEXT:    11224:       add     r12, r12, #8192
// DISASM-NEXT:    11228:       ldr     pc, [r12, #212]!
// DISASM: $d:
// DISASM-NEXT:    1122c:       d4 d4 d4 d4     .word   0xd4d4d4d4
// DISASM: $a:
// DISASM-NEXT:    11230:       add     r12, pc, #0, #12
// DISASM-NEXT:    11234:       add     r12, r12, #8192
// DISASM-NEXT:    11238:       ldr     pc, [r12, #200]!
// DISASM: $d:
// DISASM-NEXT:    1123c:       d4 d4 d4 d4     .word   0xd4d4d4d4
// DISASM: $a:
// DISASM-NEXT:    11240:       add     r12, pc, #0, #12
// DISASM-NEXT:    11244:       add     r12, r12, #4096
// DISASM-NEXT:    11248:       ldr     pc, [r12, #160]!
// DISASM: $d:
// DISASM-NEXT:    1124c:       d4 d4 d4 d4     .word   0xd4d4d4d4
// DISASM: $a:
// DISASM-NEXT:    11250:       add     r12, pc, #0, #12
// DISASM-NEXT:    11254:       add     r12, r12, #4096
// DISASM-NEXT:    11258:       ldr     pc, [r12, #148]!
// DISASM: $d:
// DISASM-NEXT:    1125c:	d4 d4 d4 d4 	.word	0xd4d4d4d4

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
