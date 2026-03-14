// REQUIRES: aarch64
// RUN: rm -rf %t && mkdir %t && cd %t

// RUN: llvm-mc -filetype=obj -triple=aarch64 %s -o a.o
// RUN: llvm-mc -filetype=obj -triple=aarch64_be %s -o a.be.o

// Note: If this test fails, it probably happens because of
//       the change of the address of the .data section.
//       You may found the correct address in the aarch64_abs16.s test,
//       if it is already fixed. Then, update addends accordingly.
// RUN: ld.lld -z max-page-size=4096 a.o --defsym foo=256 -o a
// RUN: llvm-objdump -s --section=.data a | FileCheck %s --check-prefixes=CHECK,LE
// RUN: ld.lld -z max-page-size=4096 a.be.o --defsym foo=256 -o a.be
// RUN: llvm-objdump -s --section=.data a.be | FileCheck %s --check-prefixes=CHECK,BE

// CHECK: Contents of section .data:
// 202158: S = 0x100, A = 0x212157, P = 0x202158
//         S + A - P = 0xffff
// 212a5a: S = 0x100, A = 0x1fa05a, P = 0x20215a
//         S + A - P = 0x8000
// LE-NEXT: 202158 ffff0080
// BE-NEXT: 202158 ffff8000

// RUN: not ld.lld -z max-page-size=4096 a.o --defsym foo=255 2>&1 | FileCheck %s --check-prefix=OVERFLOW1 --implicit-check-not=error:
// OVERFLOW1: error: a.o:(.data+0x2): relocation R_AARCH64_PREL16 out of range: -32769 is not in [-32768, 65535]; references 'foo'

// RUN: not ld.lld -z max-page-size=4096 a.o --defsym foo=257 2>&1 | FileCheck %s --check-prefix=OVERFLOW2 --implicit-check-not=error:
// OVERFLOW2: error: a.o:(.data+0x0): relocation R_AARCH64_PREL16 out of range: 65536 is not in [-32768, 65535]; references 'foo'

.globl _start
_start:
.data
  .hword foo - . + 0x212057
  .hword foo - . + 0x1fa05a
