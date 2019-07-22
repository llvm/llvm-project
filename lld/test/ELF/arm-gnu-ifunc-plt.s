// REQUIRES: arm
// RUN: llvm-mc -filetype=obj -triple=armv7a-linux-gnueabihf %S/Inputs/arm-shared.s -o %t1.o
// RUN: ld.lld %t1.o --shared -o %t.so
// RUN: llvm-mc -filetype=obj -triple=armv7a-linux-gnueabihf %s -o %t.o
// RUN: ld.lld --hash-style=sysv %t.so %t.o -o %tout
// RUN: llvm-objdump -triple=armv7a-linux-gnueabihf -d --no-show-raw-insn %tout | FileCheck %s --check-prefix=DISASM
// RUN: llvm-objdump -s %tout | FileCheck %s --check-prefix=GOTPLT
// RUN: llvm-readobj -r --dynamic-table %tout | FileCheck %s

// Check that the IRELATIVE relocations are last in the .got
// CHECK: Relocations [
// CHECK-NEXT:   Section (4) .rel.dyn {
// CHECK-NEXT:     0x12078 R_ARM_GLOB_DAT bar2 0x0
// CHECK-NEXT:     0x1207C R_ARM_GLOB_DAT zed2 0x0
// CHECK-NEXT:     0x12080 R_ARM_IRELATIVE - 0x0
// CHECK-NEXT:     0x12084 R_ARM_IRELATIVE - 0x0
// CHECK-NEXT:   }
// CHECK-NEXT:   Section (5) .rel.plt {
// CHECK-NEXT:     0x1300C R_ARM_JUMP_SLOT bar2 0x0
// CHECK-NEXT:     0x13010 R_ARM_JUMP_SLOT zed2 0x0
// CHECK-NEXT:   }
// CHECK-NEXT: ]

// Check that the GOT entries refer back to the ifunc resolver
// GOTPLT: Contents of section .got:
// GOTPLT-NEXT:  12078 00000000 00000000 00100100 04100100
// GOTPLT: Contents of section .got.plt:
// GOTPLT-NEXT:  13000 00000000 00000000 00000000 20100100
// GOTPLT-NEXT:  13010 20100100

// DISASM: Disassembly of section .text:
// DISASM-EMPTY:
// DISASM-NEXT: foo:
// DISASM-NEXT:    11000:       bx      lr
// DISASM: bar:
// DISASM-NEXT:    11004:       bx      lr
// DISASM: _start:
// DISASM-NEXT:    11008:       bl      #80
// DISASM-NEXT:    1100c:       bl      #92
// DISASM: $d.1:
// DISASM-NEXT:    11010:       00 00 00 00     .word   0x00000000
// DISASM-NEXT:    11014:       04 00 00 00     .word   0x00000004
// DISASM:         11018:       bl      #32
// DISASM-NEXT:    1101c:       bl      #44
// DISASM-EMPTY:
// DISASM-NEXT: Disassembly of section .plt:
// DISASM-EMPTY:
// DISASM-NEXT: $a:
// DISASM-NEXT:    11020:       str     lr, [sp, #-4]!
// DISASM-NEXT:    11024:       add     lr, pc, #0, #12
// DISASM-NEXT:    11028:       add     lr, lr, #4096
// DISASM-NEXT:    1102c:       ldr     pc, [lr, #4060]!
// DISASM: $d:
// DISASM-NEXT:    11030:       d4 d4 d4 d4     .word   0xd4d4d4d4
// DISASM-NEXT:    11034:       d4 d4 d4 d4     .word   0xd4d4d4d4
// DISASM-NEXT:    11038:       d4 d4 d4 d4     .word   0xd4d4d4d4
// DISASM-NEXT:    1103c:       d4 d4 d4 d4     .word   0xd4d4d4d4
// DISASM: $a:
// DISASM-NEXT:    11040:       add     r12, pc, #0, #12
// DISASM-NEXT:    11044:       add     r12, r12, #4096
// DISASM-NEXT:    11048:       ldr     pc, [r12, #4036]!
// DISASM: $d:
// DISASM-NEXT:    1104c:       d4 d4 d4 d4     .word   0xd4d4d4d4
// DISASM: $a:
// DISASM-NEXT:    11050:       add     r12, pc, #0, #12
// DISASM-NEXT:    11054:       add     r12, r12, #4096
// DISASM-NEXT:    11058:       ldr     pc, [r12, #4024]!
// DISASM: $d:
// DISASM-NEXT:    1105c:       d4 d4 d4 d4     .word   0xd4d4d4d4
// DISASM: $a:
// DISASM-NEXT:    11060:       add     r12, pc, #0, #12
// DISASM-NEXT:    11064:       add     r12, r12, #4096
// DISASM-NEXT:    11068:       ldr     pc, [r12, #24]!
// DISASM: $d:
// DISASM-NEXT:    1106c:       d4 d4 d4 d4     .word   0xd4d4d4d4
// DISASM: $a:
// DISASM-NEXT:    11070:       add     r12, pc, #0, #12
// DISASM-NEXT:    11074:       add     r12, r12, #4096
// DISASM-NEXT:    11078:       ldr     pc, [r12, #12]!
// DISASM: $d:
// DISASM-NEXT:   1107c:	d4 d4 d4 d4 	.word	0xd4d4d4d4

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
