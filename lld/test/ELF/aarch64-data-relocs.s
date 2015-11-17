// RUN: llvm-mc -filetype=obj -triple=aarch64-none-freebsd %s -o %t
// RUN: ld.lld2 -shared %t -o %t2
// RUN: llvm-objdump -d %t2 | FileCheck %s
// REQUIRES: aarch64

.section .R_AARCH64_ABS16, "ax",@progbits
  .hword sym+12

.section .R_AARCH64_ABS32, "ax",@progbits
  .word sym+24

.section .R_AARCH64_ABS64, "ax",@progbits
  .xword sym+36

// CHECK: Disassembly of section .R_AARCH64_ABS16:
// CHECK-NEXT: $d.0:
// CHECK-NEXT:     1000:       0c 00   .short
//                             ^-- A = 0xc

// CHECK-NEXT: Disassembly of section .R_AARCH64_ABS32:
// CHECK-NEXT: $d.1:
// CHECK-NEXT:     1002:       18 00 00 00     .word
//                             ^-- A = 0x18

// CHECK-NEXT: Disassembly of section .R_AARCH64_ABS64:
// CHECK-NEXT: $d.2:
// CHECK-NEXT:     1006:       24 00 00 00     .word
//                             ^-- A = 0x24
// CHECK-NEXT:     100a:       00 00 00 00     .word

.section .R_AARCH64_PREL16, "ax",@progbits
  .hword sym - . + 12
.section .R_AARCH64_PREL32, "ax",@progbits
  .word sym - . + 24
.section .R_AARCH64_PREL64, "ax",@progbits
  .xword sym - . + 36

// S + A = 0xc
// P = 0x100e
// SA - P = 0xeffe
// CHECK: Disassembly of section .R_AARCH64_PREL16:
// CHECK-NEXT: $d.3:
// CHECK-NEXT:     100e:       fe ef   .short

// S + A = 0x18
// P = 0x1010
// SA - P = 0xfffff016
// CHECK: Disassembly of section .R_AARCH64_PREL32:
// CHECK-NEXT: $d.4:
// CHECK-NEXT:     1010:       08 f0 ff ff     .word

// S + A = 0x24
// P = 0x1014
// SA - P = 0xfffffffffffff010
// CHECK: Disassembly of section .R_AARCH64_PREL64:
// CHECK-NEXT: $d.5:
// CHECK-NEXT:     1014:       10 f0 ff ff     .word
// CHECK-NEXT:     1018:       ff ff ff ff     .word
