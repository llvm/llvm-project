// REQUIRES: arm
// RUN: llvm-mc --triple=armv7a-linux-gnueabihf -arm-add-build-attributes -filetype=obj -o %t.o %s
// RUN: ld.lld %t.o -o %t
// RUN: llvm-objdump --triple=armv7a-none-linux-gnueabi -d --no-show-raw-insn %t | FileCheck %s

/// A symbol assignment defined alias inherits st_type and gets the same treatment.
// RUN: llvm-mc --triple=armv7a-linux-gnueabihf -arm-add-build-attributes -filetype=obj --defsym ALIAS=1 -o %t1.o %s
// RUN: ld.lld --defsym foo=foo1 %t1.o -o %t1
// RUN: llvm-objdump --triple=armv7a-none-linux-gnueabi -d --no-show-raw-insn %t | FileCheck %s

/// Non-preemptible ifuncs are called via a PLT entry which is always Arm
/// state, expect the ARM callers to go direct to the PLT entry, Thumb
/// branches are indirected via state change thunks, the bl is changed to blx.

 .syntax unified
 .text
 .balign 0x1000
.ifdef ALIAS
 .type foo1 STT_GNU_IFUNC
 .globl foo1
foo1:
.else
 .type foo STT_GNU_IFUNC
 .globl foo
foo:
.endif
 bx lr

 .section .text.1, "ax", %progbits
 .arm
 .global _start
_start:
 b foo
 bl foo

 .section .text.2, "ax", %progbits
 .thumb
 .global thumb_caller
thumb_caller:
 b foo
 b.w foo
 bl foo

// CHECK:      00021004 <_start>:
// CHECK-NEXT: b       #36 <$a>
// CHECK-NEXT: bl      #32 <$a>

// CHECK:      0002100c <thumb_caller>:
// CHECK-NEXT: b.w     #8
// CHECK-NEXT: b.w     #4
// CHECK-NEXT: blx     #24

// CHECK:      00021018 <__Thumbv7ABSLongThunk_foo>:
// CHECK-NEXT: movw    r12, #4144
// CHECK-NEXT: movt    r12, #2
// CHECK-NEXT: bx      r12

// CHECK: Disassembly of section .iplt:

// CHECK:      00021030 <$a>:
// CHECK-NEXT: add     r12, pc, #0, #12
// CHECK-NEXT: add     r12, r12, #16, #20
// CHECK-NEXT: ldr     pc, [r12, #8]!
