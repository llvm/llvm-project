// REQUIRES: arm
// RUN: llvm-mc -filetype=obj -triple=armv7a-linux-gnueabihf --arm-add-build-attributes %s -o %t.o
// RUN: ld.lld --fix-cortex-a8 --shared %t.o -o %t2
// RUN: llvm-objdump --no-print-imm-hex -d --no-show-raw-insn %t2 | FileCheck %s

/// Test case that for an OutputSection larger than the ThunkSectionSpacing
/// --fix-cortex-a8 will cause the size of the ThunkSection to be rounded up to
/// the nearest 4KiB
 .thumb

 .section .text.01, "ax", %progbits
 .balign 4096
 .globl _start
 .type _start, %function
_start:
  /// state change thunk required
  b.w arm_func
thumb_target:
  .space 4096 - 10
  /// erratum patch needed
  nop.w
  b.w thumb_target

/// Expect thunk and patch to be inserted here
// CHECK:  00012004 <__ThumbV7PILongThunk_arm_func>:
// CHECK-NEXT: 12004: movw    r12, #4088
// CHECK-NEXT:        movt    r12, #256
// CHECK-NEXT:        add     r12, pc
// CHECK-NEXT:        bx      r12
// CHECK:  00013004 <__CortexA8657417_11FFE>:
// CHECK-NEXT: 13004: b.w     0x11004 <thumb_target>
 .section .text.02
 /// Take us over thunk section spacing
 .space 16 * 1024 * 1024

 .section .text.03, "ax", %progbits
 .arm
 .balign 4
 .type arm_func, %function
arm_func:
  bx lr
