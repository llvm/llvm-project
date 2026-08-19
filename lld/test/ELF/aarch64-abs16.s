// REQUIRES: aarch64
// RUN: rm -rf %t && mkdir %t && cd %t

// RUN: llvm-mc -filetype=obj -triple=aarch64 %s -o a.o
// RUN: llvm-mc -filetype=obj -triple=aarch64_be %s -o a.be.o

// RUN: ld.lld a.o --defsym foo=256 -o a
// RUN: llvm-objdump -s --section=.data a | FileCheck %s --check-prefixes=CHECK,LE
// RUN: ld.lld a.be.o --defsym foo=256 -o a.be
// RUN: llvm-objdump -s --section=.data a.be | FileCheck %s --check-prefixes=CHECK,BE

// CHECK: Contents of section .data:
// 220158: S = 0x100, A = 0xfeff
//         S + A = 0xffff
// 22015c: S = 0x100, A = -0x8100
//         S + A = 0x8000
// LE-NEXT: 220158 ffff0080
// BE-NEXT: 220158 ffff8000

// RUN: not ld.lld a.o --defsym foo=255 2>&1 | FileCheck %s --check-prefix=OVERFLOW1 --implicit-check-not=error:
// OVERFLOW1: error: a.o:(.data+0x2): relocation R_AARCH64_ABS16 out of range: -32769 is not in [-32768, 65535]; references 'foo'

// RUN: not ld.lld a.o --defsym foo=257 2>&1 | FileCheck %s --check-prefix=OVERFLOW2 --implicit-check-not=error:
// OVERFLOW2: error: a.o:(.data+0x0): relocation R_AARCH64_ABS16 out of range: 65536 is not in [-32768, 65535]; references 'foo'

.globl _start
_start:
.data
  .hword foo + 0xfeff
  .hword foo - 0x8100
