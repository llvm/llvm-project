// REQUIRES: arm
// RUN: llvm-mc -filetype=obj -triple=armv7a-linux-gnueabihf --arm-add-build-attributes %s -o %t.o
// RUN: ld.lld --fix-cortex-a8 -verbose %t.o -o %t2 2>&1 | FileCheck %s
// RUN: llvm-objdump -d --no-show-raw-insn --start-address=0x21ffa --stop-address=0x22008 %t2 | FileCheck --check-prefix=CHECK-PATCH %s

/// Test that the patch can work on an unrelocated BLX. Neither clang or GCC
/// will emit these without a relocation, but they could be produced by ELF
/// processing tools.

// CHECK: ld.lld: detected cortex-a8-657419 erratum sequence starting at 21FFE in unpatched output.

 .syntax unified
 .text

 .type _start, %function
 .balign 4096
 .global _start
 .arm
_start:
 bx lr
 .space 4086
 .thumb
/// 32-bit Branch link and exchange spans 2 4KiB regions, preceded by a
/// 32-bit non branch instruction. Expect a patch.
 nop.w
/// Encoding for blx _start. Use .inst.n directives to avoid a relocation.
 .inst.n 0xf7ff
 .inst.n 0xe800

// CHECK-PATCH:         21ffa:          nop.w
// CHECK-PATCH-NEXT:    21ffe:          blx     0x22004 <__CortexA8657417_21FFE>
// CHECK-PATCH:      00022004 <__CortexA8657417_21FFE>:
// CHECK-PATCH-NEXT:    22004:          b       0x21000 <_start>
