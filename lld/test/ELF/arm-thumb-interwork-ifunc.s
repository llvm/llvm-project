// REQUIRES: arm
// RUN: llvm-mc --triple=armv7a-linux-gnueabihf -arm-add-build-attributes -filetype=obj -o %t.o %s
// RUN: ld.lld %t.o -o %t
// RUN: llvm-objdump -triple armv7a-none-linux-gnueabi -d --no-show-raw-insn %t

/// Non-preemptible ifuncs are called via a PLT entry which is always Arm
/// state, expect the ARM callers to go direct to the PLT entry, Thumb
/// branches are indirected via state change thunks, the bl is changed to blx.

 .syntax unified
 .text
 .balign 0x1000
 .type foo STT_GNU_IFUNC
 .globl foo
foo:
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

// CHECK: 00012004 _start:
// CHECK-NEXT: b       #36
// CHECK-NEXT: bl      #32

// CHECK: 0001200c thumb_caller:
// CHECK-NEXT: b.w     #8
// CHECK-NEXT: b.w     #4
// CHECK-NEXT: blx     #24

// CHECK: 00012018 __Thumbv7ABSLongThunk_foo:
// CHECK-NEXT: movw    r12, #8240
// CHECK-NEXT: movt    r12, #1
// CHECK-NEXT: bx      r12

// CHECK: Disassembly of section .iplt:

// CHECK: 00012030 $a:
// CHECK-NEXT: add     r12, pc, #0, #12
// CHECK-NEXT: add     r12, r12, #4096
// CHECK-NEXT: ldr     pc, [r12, #8]!
