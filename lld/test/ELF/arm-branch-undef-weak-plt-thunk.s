// REQUIRES: arm
// RUN: llvm-mc -arm-add-build-attributes -filetype=obj -triple=armv7a-none-linux-gnueabi %S/Inputs/arm-shared.s -o %t1.o
// RUN: ld.lld %t1.o --shared -soname=t1.so -o %t1.so
// RUN: llvm-mc -arm-add-build-attributes -filetype=obj -triple=armv7a-none-linux-gnueabi %s -o %t.o
// RUN: ld.lld %t.o %t1.so -o %t
// RUN: llvm-objdump --no-print-imm-hex -d --triple=armv7a-none-linux-gnueabi --start-address=0x201e4 --stop-address=0x20204 %t | FileCheck %s

/// When we are dynamic linking, undefined weak references have a PLT entry so
/// we must create a thunk for the branch to the PLT entry.

 .text
 .globl bar2
 .weak undefined_weak_we_expect_a_plt_entry_for
_start:
 .globl _start
 .type _start, %function
 b undefined_weak_we_expect_a_plt_entry_for
 bl bar2
/// Create 32 Mb gap between the call to the weak reference and the PLT so that
/// the b and bl need a range-extension thunk.
 .section .text.1, "ax", %progbits
 .space 32 * 1024 * 1024

// CHECK: Disassembly of section .text:
// CHECK-EMPTY:
// CHECK-NEXT: <_start>:
// CHECK-NEXT:    201e4:       ea000000        b       0x201ec <__ARMv7ABSLongThunk_undefined_weak_we_expect_a_plt_entry_for>
// CHECK-NEXT:    201e8:       eb000002        bl      0x201f8 <__ARMv7ABSLongThunk_bar2>
// CHECK: <__ARMv7ABSLongThunk_undefined_weak_we_expect_a_plt_entry_for>:
// CHECK-NEXT:    201ec:        e300c240        movw    r12, #576
// CHECK-NEXT:    201f0:        e340c202        movt    r12, #514
// CHECK-NEXT:    201f4:        e12fff1c        bx      r12
// CHECK: <__ARMv7ABSLongThunk_bar2>:
// CHECK-NEXT:    201f8:        e300c230        movw    r12, #560
// CHECK-NEXT:    201fc:        e340c202        movt    r12, #514
// CHECK-NEXT:    20200:        e12fff1c        bx      r12
