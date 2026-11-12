// REQUIRES: arm
// RUN: llvm-mc -filetype=obj -triple=armv7a-none-linux-gnueabi %s -o %t
// RUN: ld.lld %t -o %t2
// RUN: llvm-readelf -s %t2 | FileCheck %s
// CHECK-NOT: $d.exidx.foo

/// Test that symbols which point to input .ARM.exidx sections are eliminated.
/// These symbols might be produced, for example, by GNU tools.

    .syntax unified
    .section .text.foo,"axG",%progbits,foo,comdat
foo:
    bx lr

/// GNU as adds mapping symbols "$d" for .ARM.exidx sections it generates.
/// llvm-mc does not do that, so reproduce that manually.
    .section .ARM.exidx.text.foo,"ao?",%0x70000001,.text.foo
$d.exidx.foo:
    .reloc 0, R_ARM_NONE, __aeabi_unwind_cpp_pr0
    .long .text.foo(PREL31)
    .long 0x80b0b0b0

    .section .text.h,"ax"
    .global __aeabi_unwind_cpp_pr0
__aeabi_unwind_cpp_pr0:
    bx lr
