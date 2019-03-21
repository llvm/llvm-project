// REQUIRES: arm
// RUN: llvm-mc -filetype=obj -triple=armv7a-none-linux-gnueabi --arm-add-build-attributes %s -o %t
// RUN: ld-lld %t -o %t2
// RUN: llvm-objdump -s %t2 | FileCheck %s

// We do not want to generate missing EXIDX_CANTUNWIND entries if there are no
// input .ARM.exidx sections.

// CHECK-NOT: .ARM.exidx
        .syntax unified
        .text
	.globl _start
	.type _start, %function
_start:
	bx lr

