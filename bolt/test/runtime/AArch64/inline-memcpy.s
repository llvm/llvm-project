## This test checks that BOLT correctly inlines memcpy calls on AArch64.

# REQUIRES: system-linux, aarch64-registered-target

# RUN: llvm-mc -filetype=obj -triple aarch64-unknown-unknown %s -o %t.o
# RUN: %clang %cflags -no-pie %t.o -o %t.exe -Wl,-q
# RUN: llvm-bolt %t.exe --inline-memcpy -o %t.bolt 2>&1 | FileCheck %s --check-prefix=CHECK-INLINE
# RUN: llvm-objdump -d %t.bolt | FileCheck %s --check-prefix=CHECK-ASM

# Verify BOLT reports that it inlined memcpy calls (11 successful inlines out of 16 total calls)
# CHECK-INLINE: BOLT-INFO: inlined 11 memcpy() calls

# Each function should use optimal size-specific instructions and NO memcpy calls

# 1-byte copy should use single byte load/store (ldrb/strb)
# CHECK-ASM-LABEL: <test_1_byte_direct>:
# CHECK-ASM: ldrb{{.*}}w9, [x1]
# CHECK-ASM-NEXT: strb{{.*}}w9, [x0]
# CHECK-ASM-NOT: bl{{.*}}<memcpy

# 2-byte copy should use single 16-bit load/store (ldrh/strh)
# CHECK-ASM-LABEL: <test_2_byte_direct>:
# CHECK-ASM: ldrh{{.*}}w9, [x1]
# CHECK-ASM-NEXT: strh{{.*}}w9, [x0]
# CHECK-ASM-NOT: bl{{.*}}<memcpy

# 4-byte copy should use single 32-bit load/store (w register)
# CHECK-ASM-LABEL: <test_4_byte_direct>:
# CHECK-ASM: ldr{{.*}}w9, [x1]
# CHECK-ASM-NEXT: str{{.*}}w9, [x0]
# CHECK-ASM-NOT: bl{{.*}}<memcpy

# 8-byte copy should use single 64-bit load/store (x register)
# CHECK-ASM-LABEL: <test_8_byte_direct>:
# CHECK-ASM: ldr{{.*}}x9, [x1]
# CHECK-ASM-NEXT: str{{.*}}x9, [x0]
# CHECK-ASM-NOT: bl{{.*}}<memcpy

# 16-byte copy should use single 128-bit SIMD load/store (q register)
# CHECK-ASM-LABEL: <test_16_byte_direct>:
# CHECK-ASM: ldr{{.*}}q16, [x1]
# CHECK-ASM-NEXT: str{{.*}}q16, [x0]
# CHECK-ASM-NOT: bl{{.*}}<memcpy

# 32-byte copy should use two 128-bit SIMD operations
# CHECK-ASM-LABEL: <test_32_byte_direct>:
# CHECK-ASM: ldr{{.*}}q16, [x1]
# CHECK-ASM-NEXT: str{{.*}}q16, [x0]
# CHECK-ASM-NEXT: ldr{{.*}}q17, [x1, #0x10]
# CHECK-ASM-NEXT: str{{.*}}q17, [x0, #0x10]
# CHECK-ASM-NOT: bl{{.*}}<memcpy

# 37-byte copy should use greedy decomposition: (2*16) + (1*4) + (1*1)
# CHECK-ASM-LABEL: <test_37_byte_arbitrary>:
# CHECK-ASM: ldr{{.*}}q16, [x1]
# CHECK-ASM-NEXT: str{{.*}}q16, [x0]
# CHECK-ASM-NEXT: ldr{{.*}}q16, [x1, #0x10]
# CHECK-ASM-NEXT: str{{.*}}q16, [x0, #0x10]
# CHECK-ASM-NEXT: ldr{{.*}}w9, [x1, #0x20]
# CHECK-ASM-NEXT: str{{.*}}w9, [x0, #0x20]
# CHECK-ASM-NEXT: ldrb{{.*}}w9, [x1, #0x24]
# CHECK-ASM-NEXT: strb{{.*}}w9, [x0, #0x24]
# CHECK-ASM-NOT: bl{{.*}}<memcpy

# 0-byte copy should be inlined with no load/store instructions (nothing to copy)
# CHECK-ASM-LABEL: <test_0_byte>:
# CHECK-ASM-NOT: ldr
# CHECK-ASM-NOT: str
# CHECK-ASM-NOT: bl{{.*}}<memcpy

# Negative size should NOT be inlined (invalid size parameter)
# CHECK-ASM-LABEL: <test_negative_size>:
# CHECK-ASM: bl{{.*}}<memcpy

# 128-byte copy should NOT be inlined (too large, original call preserved)
# CHECK-ASM-LABEL: <test_128_byte_too_large>:
# CHECK-ASM: bl{{.*}}<memcpy

# ADD immediate with non-zero source should NOT be inlined (can't track mov+add chain)
# CHECK-ASM-LABEL: <test_4_byte_add_immediate>:
# CHECK-ASM: bl{{.*}}<memcpy

# Register move should NOT be inlined (size unknown at compile time)
# CHECK-ASM-LABEL: <test_register_move_negative>:
# CHECK-ASM: bl{{.*}}<memcpy

# Live-in parameter should NOT be inlined (size unknown at compile time)
# CHECK-ASM-LABEL: <test_live_in_negative>:
# CHECK-ASM: bl{{.*}}<memcpy

# _memcpy8 should be inlined with end-pointer return (dest+size)
# CHECK-ASM-LABEL: <test_memcpy8_4_byte>:
# CHECK-ASM: ldr{{.*}}w9, [x1]
# CHECK-ASM-NEXT: str{{.*}}w9, [x0]
# CHECK-ASM-NEXT: add{{.*}}x0, x0, #0x4
# CHECK-ASM-NOT: bl{{.*}}<_memcpy8

# Complex function with caller-saved X9 should inline 8-byte memcpy using X9 as temp register
# CHECK-ASM-LABEL: <complex_operation>:
# CHECK-ASM: ldr{{.*}}x9, [x1]
# CHECK-ASM-NEXT: str{{.*}}x9, [x0]
# CHECK-ASM-NOT: bl{{.*}}<memcpy

# Complex function with caller-saved Q16/Q17 should inline 64-byte memcpy using Q16 as temp register
# CHECK-ASM-LABEL: <complex_fp_operation>:
# CHECK-ASM: ldr{{.*}}q16, [x1]
# CHECK-ASM-NEXT: str{{.*}}q16, [x0]
# CHECK-ASM-NEXT: ldr{{.*}}q16, [x1, #0x10]
# CHECK-ASM-NEXT: str{{.*}}q16, [x0, #0x10]
# CHECK-ASM-NEXT: ldr{{.*}}q16, [x1, #0x20]
# CHECK-ASM-NEXT: str{{.*}}q16, [x0, #0x20]
# CHECK-ASM-NEXT: ldr{{.*}}q16, [x1, #0x30]
# CHECK-ASM-NEXT: str{{.*}}q16, [x0, #0x30]
# CHECK-ASM-NOT: bl{{.*}}<memcpy

	.text
	.globl	test_1_byte_direct
	.type	test_1_byte_direct,@function
test_1_byte_direct:
	stp	x29, x30, [sp, #-32]!
	mov	x29, sp
	add	x1, sp, #16
	add	x0, sp, #8
	mov	x2, #1
	bl	memcpy
	ldp	x29, x30, [sp], #32
	ret
	.size	test_1_byte_direct, .-test_1_byte_direct

	.globl	test_2_byte_direct
	.type	test_2_byte_direct,@function
test_2_byte_direct:
	stp	x29, x30, [sp, #-32]!
	mov	x29, sp
	add	x1, sp, #16
	add	x0, sp, #8
	mov	x2, #2
	bl	memcpy
	ldp	x29, x30, [sp], #32
	ret
	.size	test_2_byte_direct, .-test_2_byte_direct

	.globl	test_4_byte_direct
	.type	test_4_byte_direct,@function
test_4_byte_direct:
	stp	x29, x30, [sp, #-32]!
	mov	x29, sp
	add	x1, sp, #16
	add	x0, sp, #8
	mov	x2, #4
	bl	memcpy
	ldp	x29, x30, [sp], #32
	ret
	.size	test_4_byte_direct, .-test_4_byte_direct

	.globl	test_8_byte_direct
	.type	test_8_byte_direct,@function
test_8_byte_direct:
	stp	x29, x30, [sp, #-32]!
	mov	x29, sp
	add	x1, sp, #16
	add	x0, sp, #8
	mov	x2, #8
	bl	memcpy
	ldp	x29, x30, [sp], #32
	ret
	.size	test_8_byte_direct, .-test_8_byte_direct

	.globl	test_16_byte_direct
	.type	test_16_byte_direct,@function
test_16_byte_direct:
	stp	x29, x30, [sp, #-48]!
	mov	x29, sp
	add	x1, sp, #16
	add	x0, sp, #32
	mov	x2, #16
	bl	memcpy
	ldp	x29, x30, [sp], #48
	ret
	.size	test_16_byte_direct, .-test_16_byte_direct

	.globl	test_32_byte_direct
	.type	test_32_byte_direct,@function
test_32_byte_direct:
	stp	x29, x30, [sp, #-80]!
	mov	x29, sp
	add	x1, sp, #16
	add	x0, sp, #48
	mov	x2, #32
	bl	memcpy
	ldp	x29, x30, [sp], #80
	ret
	.size	test_32_byte_direct, .-test_32_byte_direct

	.globl	test_37_byte_arbitrary
	.type	test_37_byte_arbitrary,@function
test_37_byte_arbitrary:
	stp	x29, x30, [sp, #-96]!
	mov	x29, sp
	add	x1, sp, #16
	add	x0, sp, #56
	mov	x2, #37
	bl	memcpy
	ldp	x29, x30, [sp], #96
	ret
	.size	test_37_byte_arbitrary, .-test_37_byte_arbitrary

	.globl	test_0_byte
	.type	test_0_byte,@function
test_0_byte:
	stp	x29, x30, [sp, #-32]!
	mov	x29, sp
	add	x1, sp, #16
	add	x0, sp, #8
	mov	x2, #0
	bl	memcpy
	ldp	x29, x30, [sp], #32
	ret
	.size	test_0_byte, .-test_0_byte

	.globl	test_negative_size
	.type	test_negative_size,@function
test_negative_size:
	# Negative size should not be inlined
	stp	x29, x30, [sp, #-32]!
	mov	x29, sp
	add	x1, sp, #16
	add	x0, sp, #8
	mov	x2, #-1
	bl	memcpy
	ldp	x29, x30, [sp], #32
	ret
	.size	test_negative_size, .-test_negative_size

	.globl	test_128_byte_too_large
	.type	test_128_byte_too_large,@function
test_128_byte_too_large:
	stp	x29, x30, [sp, #-288]!
	mov	x29, sp
	add	x1, sp, #16
	add	x0, sp, #152
	mov	x2, #128
	bl	memcpy
	ldp	x29, x30, [sp], #288
	ret
	.size	test_128_byte_too_large, .-test_128_byte_too_large

	.globl	test_4_byte_add_immediate
	.type	test_4_byte_add_immediate,@function
test_4_byte_add_immediate:
	stp	x29, x30, [sp, #-32]!
	mov	x29, sp
	add	x1, sp, #16
	add	x0, sp, #8
	mov	x3, #0
	add	x2, x3, #4
	bl	memcpy
	ldp	x29, x30, [sp], #32
	ret
	.size	test_4_byte_add_immediate, .-test_4_byte_add_immediate

	.globl	test_register_move_negative
	.type	test_register_move_negative,@function
test_register_move_negative:
	stp	x29, x30, [sp, #-32]!
	mov	x29, sp
	add	x1, sp, #16
	add	x0, sp, #8
	mov	x6, #4
	mov	x2, x6
	bl	memcpy
	ldp	x29, x30, [sp], #32
	ret
	.size	test_register_move_negative, .-test_register_move_negative

	.globl	test_live_in_negative
	.type	test_live_in_negative,@function
test_live_in_negative:
	# x2 comes in as parameter, no instruction sets it (should NOT inline)
	stp	x29, x30, [sp, #-32]!
	mov	x29, sp
	add	x1, sp, #16
	add	x0, sp, #8
	# x2 is live-in, no size-setting instruction
	bl	memcpy
	ldp	x29, x30, [sp], #32
	ret
	.size	test_live_in_negative, .-test_live_in_negative

	.globl	test_memcpy8_4_byte
	.type	test_memcpy8_4_byte,@function
test_memcpy8_4_byte:
	stp	x29, x30, [sp, #-32]!
	mov	x29, sp
	add	x1, sp, #16
	add	x0, sp, #8
	mov	x2, #4
	bl	_memcpy8
	ldp	x29, x30, [sp], #32
	ret
	.size	test_memcpy8_4_byte, .-test_memcpy8_4_byte

	# Simple _memcpy8 implementation that calls memcpy and returns dest+size
	.globl	_memcpy8
	.type	_memcpy8,@function
_memcpy8:
	stp	x29, x30, [sp, #-16]!
	mov	x29, sp
	mov	x3, x0
	bl	memcpy
	add	x0, x3, x2
	ldp	x29, x30, [sp], #16
	ret
	.size	_memcpy8, .-_memcpy8

	.globl	complex_operation
	.type	complex_operation,@function
complex_operation:
	stp     x29, x30, [sp, #-32]!
	str     x19, [sp, #16]
	mov     x29, sp
	ldp     x9, x10, [x0]
	ldp     x11, x12, [x0, #16]
	mov     x19, x1
	mov     x8, x0
	add     x0, x1, #32
	madd    x9, x9, x2, x3
	and     x10, x10, x4
	asr     x12, x12, #2
	mov     w2, #8
	orr     x11, x12, x11, lsl #3
	eor     x12, x9, x10
	mul     x10, x11, x10
	eor     x12, x12, x11
	add     x13, x12, x9
	add     x9, x11, x9, asr #4
	stp     x13, x10, [x1]
	mov     w10, w12
	stp     x9, x10, [x1, #16]
	add     x1, x8, #32
	bl      memcpy
	ldr     x0, [x19, #16]
	ldr     x19, [sp, #16]
	ldp     x29, x30, [sp], #32
	b       use
	.size	complex_operation, .-complex_operation

	.globl	use
	.type	use,@function
use:
	ret
	.size	use, .-use

# Same as above but using FP caller-saved registers (Q16/17)
	.globl	complex_fp_operation
	.type	complex_fp_operation,@function
complex_fp_operation:
	stp     x29, x30, [sp, #-48]!
	stp     q8, q9, [sp, #16]
	mov     x29, sp
	ldr     q16, [x0]
	ldr     q17, [x0, #16]
	mov     x8, x0
	add     x0, x1, #32
	fadd    v16.4s, v16.4s, v17.4s
	fmul    v17.4s, v16.4s, v17.4s
	fsub    v16.2d, v16.2d, v17.2d
	mov     w2, #64
	fmax    v17.4s, v16.4s, v17.4s
	fmin    v16.2d, v16.2d, v17.2d
	str     q16, [x1]
	str     q17, [x1, #16]
	add     x1, x8, #32
	bl      memcpy
	ldp     q8, q9, [sp, #16]
	ldp     x29, x30, [sp], #48
	b       use_fp
	.size	complex_fp_operation, .-complex_fp_operation

	.globl	use_fp
	.type	use_fp,@function
use_fp:
	ret
	.size	use_fp, .-use_fp
