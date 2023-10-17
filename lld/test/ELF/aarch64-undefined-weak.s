// REQUIRES: aarch64
// RUN: llvm-mc -filetype=obj -triple=aarch64-none-linux %s -o %t.o
// RUN: ld.lld --image-base=0x10000000 %t.o -o %t
// RUN: llvm-objdump -d --no-show-raw-insn %t | FileCheck %s

// Check that the ARM 64-bit ABI rules for undefined weak symbols are applied.
// Branch instructions are resolved to the next instruction. Undefined
// Symbols in relative are resolved to the place so S - P + A = A.
// We place the image-base at 0x10000000 to test that a range extensions thunk
// is not generated.
 .weak target

.weak undefweak2
.hidden undefweak2

 .text
 .global _start
_start:
// R_AARCH64_JUMP26
 b target
// R_AARCH64_CALL26
 bl target
// R_AARCH64_CONDBR19
 b.eq target
// R_AARCH64_TSTBR14
 cbz x1, target
// R_AARCH64_ADR_PREL_LO21
 adr x0, target
// R_AARCH64_ADR_PREL_PG_HI21
 adrp x0, target
// R_AARCH64_LD_PREL_LO19
 ldr x8, target
// R_AARCH64_PREL32
 .word target - .
// R_AARCH64_PREL64
 .xword target - .
// R_AARCH64_PREL16
 .hword target - .
// R_AARCH64_PLT32
 .word target@PLT - .

bl_undefweak2:
 bl undefweak2

// CHECK: Disassembly of section .text:
// CHECK-EMPTY:
// CHECK-NEXT: 0000000010010120 <_start>:
// CHECK-NEXT: 10010120: b       0x10010124
// CHECK-NEXT: 10010124: bl      0x10010128
// CHECK-NEXT: 10010128: b.eq    0x1001012c
// CHECK-NEXT: 1001012c: cbz     x1, 0x10010130
// CHECK-NEXT: 10010130: adr     x0, 0x10010130
// CHECK-NEXT: 10010134: adrp    x0, 0x10010000
// CHECK-NEXT: 10010138: ldr     x8, 0x10010138
// CHECK:      1001013c: 00 00 00 00     .word   0x00000000
// CHECK-NEXT: 10010140: 00 00 00 00     .word   0x00000000
// CHECK-NEXT: 10010144: 00 00 00 00     .word   0x00000000
// CHECK-NEXT: 10010148: 00 00 00 00     .word   0x00000000
// CHECK-NEXT: 1001014c: 00 00           .short  0x0000
// CHECK-LABEL: <bl_undefweak2>:
// CHECK-NEXT:    bl {{.*}} <bl_undefweak2+0x4>
