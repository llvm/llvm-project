// REQUIRES: arm
// RUN: llvm-mc -arm-add-build-attributes -filetype=obj -triple=armv7a-none-linux-gnueabi %S/Inputs/arm-shared.s -o %t1.o
// RUN: ld.lld %t1.o --shared -soname=t1.so -o %t1.so
// RUN: llvm-mc -arm-add-build-attributes -filetype=obj -triple=armv7a-none-linux-gnueabi %s -o %t.o
// RUN: ld.lld %t.o %t1.so -o %t
// RUN: llvm-objdump -d -triple=armv7a-none-linux-gnueabi -start-address=0x111e4 -stop-address=0x11204 %t | FileCheck %s

// When we are dynamic linking, undefined weak references have a PLT entry so
// we must create a thunk for the branch to the PLT entry.

 .text
 .globl bar2
 .weak undefined_weak_we_expect_a_plt_entry_for
_start:
 .globl _start
 .type _start, %function
 b undefined_weak_we_expect_a_plt_entry_for
 bl bar2
// Create 32 Mb gap between the call to the weak reference and the PLT so that
// the b and bl need a range-extension thunk.
 .section .text.1, "ax", %progbits
 .space 32 * 1024 * 1024

// CHECK: Disassembly of section .text:
// CHECK-EMPTY:
// CHECK-NEXT: _start:
// CHECK-NEXT:    111e4:       00 00 00 ea     b       #0 <__ARMv7ABSLongThunk_undefined_weak_we_expect_a_plt_entry_for>
// CHECK-NEXT:    111e8:       02 00 00 eb     bl      #8 <__ARMv7ABSLongThunk_bar2>
// CHECK: __ARMv7ABSLongThunk_undefined_weak_we_expect_a_plt_entry_for:
// CHECK-NEXT:    111ec:        30 c2 01 e3     movw    r12, #4656
// CHECK-NEXT:    111f0:        01 c2 40 e3     movt    r12, #513
// CHECK-NEXT:    111f4:        1c ff 2f e1     bx      r12
// CHECK: __ARMv7ABSLongThunk_bar2:
// CHECK-NEXT:    111f8:        40 c2 01 e3     movw    r12, #4672
// CHECK-NEXT:    111fc:        01 c2 40 e3     movt    r12, #513
// CHECK-NEXT:    11200:        1c ff 2f e1     bx      r12
