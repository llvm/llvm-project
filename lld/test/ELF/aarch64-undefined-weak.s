// RUN: llvm-mc -filetype=obj -triple=aarch64-none-linux %s -o %t
// RUN: ld.lld %t -o %t2 2>&1
// RUN: llvm-objdump -triple=aarch64-none-linux -d %t2 | FileCheck %s
// REQUIRES: aarch64

// Check that the ARM 64-bit ABI rules for undefined weak symbols are applied.
// Branch instructions are resolved to the next instruction. Undefined
// Symbols in relative are resolved to the place so S - P + A = A.

 .weak target

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
// R_AARCH64_PREL32
 .word target - .
// R_AARCH64_PREL64
 .xword target - .
// R_AARCH64_PREL16
 .hword target - .

// CHECK: Disassembly of section .text:
// 131076 = 0x20004
// CHECK:         20000:       01 80 00 14     b       #131076
// CHECK-NEXT:    20004:       02 80 00 94     bl      #131080
// CHECK-NEXT:    20008:       60 00 10 54     b.eq    #131084
// CHECK-NEXT:    2000c:       81 00 10 b4     cbz     x1, #131088
// CHECK-NEXT:    20010:       00 00 00 10     adr     x0, #0
// CHECK-NEXT:    20014:       00 00 00 90     adrp    x0, #0
// CHECK:         20018:       00 00 00 00     .word   0x00000000
// CHECK-NEXT:    2001c:       00 00 00 00     .word   0x00000000
// CHECK-NEXT:    20020:       00 00 00 00     .word   0x00000000
// CHECK-NEXT:    20024:       00 00           .short  0x0000
