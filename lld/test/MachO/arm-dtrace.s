# REQUIRES: arm
# RUN: rm -rf %t; split-file %s %t

# RUN: llvm-mc -filetype=obj -triple=armv4t-apple-darwin %t/armv4t-dtrace.s -o %t/armv4t-dtrace.o
# RUN: %lld -arch armv4t -o %t/armv4t-dtrace %t/armv4t-dtrace.o

## If references of dtrace symbols are handled by lld, their relocation should be replaced with the following instructions
# RUN: llvm-objdump --macho -D %t/armv4t-dtrace | FileCheck %s --check-prefix=CHECK-armv4t

# CHECK-armv4t: 00 00 20 e0  eor     r0, r0, r0

# CHECK-armv4t: 00 00 a0 e1  mov     r0, r0

# RUN: llvm-mc -filetype=obj -triple=thumbv7-apple-darwin %t/armv7-dtrace.s -o %t/armv7-dtrace.o
# RUN: %lld -arch armv7 -o %t/armv7-dtrace %t/armv7-dtrace.o

## If references of dtrace symbols are handled by lld, their relocation should be replaced with the following instructions
# RUN: llvm-objdump --macho -D %t/armv7-dtrace | FileCheck %s --check-prefix=CHECK-armv7

# CHECK-armv7:      40 40   eors    r0, r0
# CHECK-armv7-NEXT: c0 46   mov     r8, r8

# CHECK-armv7:      c0 46   mov     r8, r8
# CHECK-armv7-NEXT: c0 46   mov     r8, r8

;--- armv4t-dtrace.s
	.globl	_main
_main:
	bl	___dtrace_isenabled$Foo$added$v1
	.reference	___dtrace_typedefs$Foo$v2
	bl	___dtrace_probe$Foo$added$v1$696e74
	.reference	___dtrace_stability$Foo$v1$1_1_0_1_1_0_1_1_0_1_1_0_1_1_0

.subsections_via_symbols

;--- armv7-dtrace.s
	.globl	_main
	.thumb_func	_main
_main:
	bl	___dtrace_isenabled$Foo$added$v1
	.reference	___dtrace_typedefs$Foo$v2
	bl	___dtrace_probe$Foo$added$v1$696e74
	.reference	___dtrace_stability$Foo$v1$1_1_0_1_1_0_1_1_0_1_1_0_1_1_0

.subsections_via_symbols
