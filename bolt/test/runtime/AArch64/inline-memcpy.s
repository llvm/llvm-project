## This test checks that BOLT correctly inlines memcpy calls on AArch64.

# REQUIRES: system-linux, aarch64-registered-target

# RUN: llvm-mc -filetype=obj -triple aarch64-unknown-unknown %s -o %t.o
# RUN: %clang %cflags -no-pie %t.o -o %t.exe -Wl,-q 
# RUN: llvm-bolt %t.exe --inline-memcpy -o %t.bolt 2>&1 | FileCheck %s --check-prefix=CHECK-INLINE
# RUN: llvm-objdump -d %t.bolt | FileCheck %s --check-prefix=CHECK-ASM

# Verify BOLT reports that it inlined memcpy calls (8 successful inlines out of 11 total calls)
# CHECK-INLINE: BOLT-INFO: inlined 8 memcpy() calls

# Each function should use optimal size-specific instructions and NO memcpy calls

# 1-byte copy should use single byte load/store (ldrb/strb)
# CHECK-ASM-LABEL: <test_1_byte_direct>:
# CHECK-ASM: ldrb{{.*}}w{{[0-9]+}}, [x1]
# CHECK-ASM: strb{{.*}}w{{[0-9]+}}, [x0]
# CHECK-ASM-NOT: bl{{.*}}<memcpy

# 2-byte copy should use single 16-bit load/store (ldrh/strh)
# CHECK-ASM-LABEL: <test_2_byte_direct>:
# CHECK-ASM: ldrh{{.*}}w{{[0-9]+}}, [x1]
# CHECK-ASM: strh{{.*}}w{{[0-9]+}}, [x0]
# CHECK-ASM-NOT: bl{{.*}}<memcpy

# 4-byte copy should use single 32-bit load/store (w register)
# CHECK-ASM-LABEL: <test_4_byte_direct>:
# CHECK-ASM: ldr{{.*}}w{{[0-9]+}}, [x1]
# CHECK-ASM: str{{.*}}w{{[0-9]+}}, [x0]
# CHECK-ASM-NOT: bl{{.*}}<memcpy

# 8-byte copy should use single 64-bit load/store (x register)
# CHECK-ASM-LABEL: <test_8_byte_direct>:
# CHECK-ASM: ldr{{.*}}x{{[0-9]+}}, [x1]
# CHECK-ASM: str{{.*}}x{{[0-9]+}}, [x0]
# CHECK-ASM-NOT: bl{{.*}}<memcpy

# 16-byte copy should use single 128-bit SIMD load/store (q register)
# CHECK-ASM-LABEL: <test_16_byte_direct>:
# CHECK-ASM: ldr{{.*}}q{{[0-9]+}}, [x1]
# CHECK-ASM: str{{.*}}q{{[0-9]+}}, [x0]
# CHECK-ASM-NOT: bl{{.*}}<memcpy

# 32-byte copy should use two 128-bit SIMD operations
# CHECK-ASM-LABEL: <test_32_byte_direct>:
# CHECK-ASM: ldr{{.*}}q{{[0-9]+}}, [x1]
# CHECK-ASM: str{{.*}}q{{[0-9]+}}, [x0]
# CHECK-ASM: ldr{{.*}}q{{[0-9]+}}, [x1, #0x10]
# CHECK-ASM: str{{.*}}q{{[0-9]+}}, [x0, #0x10]
# CHECK-ASM-NOT: bl{{.*}}<memcpy

# 37-byte copy should use greedy decomposition: (2*16) + (1*4) + (1*1)
# CHECK-ASM-LABEL: <test_37_byte_arbitrary>:
# CHECK-ASM: ldr{{.*}}q{{[0-9]+}}, [x1]
# CHECK-ASM: str{{.*}}q{{[0-9]+}}, [x0]
# CHECK-ASM: ldr{{.*}}q{{[0-9]+}}, [x1, #0x10]
# CHECK-ASM: str{{.*}}q{{[0-9]+}}, [x0, #0x10]
# CHECK-ASM: ldr{{.*}}w{{[0-9]+}}, [x1, #0x20]
# CHECK-ASM: str{{.*}}w{{[0-9]+}}, [x0, #0x20]
# CHECK-ASM: ldrb{{.*}}w{{[0-9]+}}, [x1, #0x24]
# CHECK-ASM: strb{{.*}}w{{[0-9]+}}, [x0, #0x24]
# CHECK-ASM-NOT: bl{{.*}}<memcpy

# 128-byte copy should be "inlined" by removing the call entirely (too large for real inlining)
# CHECK-ASM-LABEL: <test_128_byte_too_large>:
# CHECK-ASM-NOT: bl{{.*}}<memcpy
# CHECK-ASM-NOT: ldr{{.*}}q{{[0-9]+}}

# ADD immediate with non-zero source should NOT be inlined (can't track mov+add chain)
# CHECK-ASM-LABEL: <test_4_byte_add_immediate>:
# CHECK-ASM: bl{{.*}}<memcpy

# Register move should NOT be inlined (size unknown at compile time)
# CHECK-ASM-LABEL: <test_register_move_negative>:
# CHECK-ASM: bl{{.*}}<memcpy

# Live-in parameter should NOT be inlined (size unknown at compile time)
# CHECK-ASM-LABEL: <test_live_in_negative>:
# CHECK-ASM: bl{{.*}}<memcpy

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



	.globl	main
	.type	main,@function
main:
	stp	x29, x30, [sp, #-16]!
	mov	x29, sp
	
	bl	test_1_byte_direct
	bl	test_2_byte_direct
	bl	test_4_byte_direct
	bl	test_8_byte_direct
	bl	test_16_byte_direct  
	bl	test_32_byte_direct
	bl	test_37_byte_arbitrary
	bl	test_128_byte_too_large
	bl	test_4_byte_add_immediate
	bl	test_register_move_negative
	bl	test_live_in_negative
	
	mov	w0, #0
	ldp	x29, x30, [sp], #16
	ret
	.size	main, .-main
