// REQUIRES: arm
// RUN: rm -rf %t && split-file %s %t && cd %t
// RUN: llvm-mc -arm-add-build-attributes -filetype=obj -triple=armv7a-none-linux-gnueabi shared.s -o shared.o
// RUN: ld.lld shared.o --shared -soname=t1.so -o shared.so
// RUN: llvm-mc -arm-add-build-attributes -filetype=obj -triple=armv7a-none-linux-gnueabi a.s -o a.o
// RUN: ld.lld a.o shared.so --script a.lds -o exe
// RUN: llvm-objdump --no-show-raw-insn -d --triple=armv7a-none-linux-gnueabi exe | FileCheck %s

/// When we are dynamic linking, undefined weak references have a PLT entry so
/// we must create a thunk for the branch to the PLT entry.

//--- a.lds
SECTIONS {
	.text 0x2000000 : AT(0x2000000) { *(.text) }
	.plt  0x4000100 : AT(0x4000100) { *(.plt) }
}
//--- shared.s

.syntax unified
 .global bar2
 .type bar2, %function
bar2:

 .global zed2
 .type zed2, %function
zed2:

//--- a.s

 .text
 .globl bar2
 .weak undefined_weak_we_expect_a_plt_entry_for
_start:
 .globl _start
 .type _start, %function
 b undefined_weak_we_expect_a_plt_entry_for
 bl bar2

// CHECK-LABEL: <_start>:
// CHECK-NEXT: 2000000: b  0x2000008 <__ARMv7ABSLongThunk_undefined_weak_we_expect_a_plt_entry_for>
// CHECK-NEXT:          bl 0x2000014 <__ARMv7ABSLongThunk_bar2>

// CHECK-LABEL: <__ARMv7ABSLongThunk_undefined_weak_we_expect_a_plt_entry_for>:
// CHECK-NEXT: 2000008: movw    r12, #0x130
// CHECK-NEXT:          movt    r12, #0x400
// CHECK-NEXT:          bx      r12

// CHECK-LABEL: <__ARMv7ABSLongThunk_bar2>:
// CHECK-NEXT: 2000014: movw    r12, #0x120
// CHECK-NEXT:          movt    r12, #0x400
// CHECK-NEXT:          bx      r12

// CHECK-LABEL:<bar2@plt>:
// CHECK-NEXT: 4000120: add     r12, pc, #0, #12

// CHECK-LABEL: <undefined_weak_we_expect_a_plt_entry_for@plt>:
// CHECK-NEXT: 4000130: add     r12, pc, #0, #12
