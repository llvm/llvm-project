/* strcat with 256-bit EVEX instructions.
   Copyright (C) 2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; if not, see
   <https://www.gnu.org/licenses/>.  */

#if IS_IN (libc)

# include <sysdep.h>

# ifndef STRCAT
#  define STRCAT  __strcat_evex
# endif

# define VMOVU		vmovdqu64
# define VMOVA		vmovdqa64

/* zero register */
# define XMMZERO	xmm16
# define YMMZERO	ymm16
# define YMM0		ymm17
# define YMM1		ymm18

# define USE_AS_STRCAT

/* Number of bytes in a vector register */
# define VEC_SIZE	32

	.section .text.evex,"ax",@progbits
ENTRY (STRCAT)
	mov	%rdi, %r9
# ifdef USE_AS_STRNCAT
	mov	%rdx, %r8
# endif

	xor	%eax, %eax
	mov	%edi, %ecx
	and	$((VEC_SIZE * 4) - 1), %ecx
	vpxorq	%XMMZERO, %XMMZERO, %XMMZERO
	cmp	$(VEC_SIZE * 3), %ecx
	ja	L(fourth_vector_boundary)
	vpcmpb	$0, (%rdi), %YMMZERO, %k0
	kmovd	%k0, %edx
	test	%edx, %edx
	jnz	L(exit_null_on_first_vector)
	mov	%rdi, %rax
	and	$-VEC_SIZE, %rax
	jmp	L(align_vec_size_start)
L(fourth_vector_boundary):
	mov	%rdi, %rax
	and	$-VEC_SIZE, %rax
	vpcmpb	$0, (%rax), %YMMZERO, %k0
	mov	$-1, %r10d
	sub	%rax, %rcx
	shl	%cl, %r10d
	kmovd	%k0, %edx
	and	%r10d, %edx
	jnz	L(exit)

L(align_vec_size_start):
	vpcmpb	$0, VEC_SIZE(%rax), %YMMZERO, %k0
	kmovd	%k0, %edx
	test	%edx, %edx
	jnz	L(exit_null_on_second_vector)

	vpcmpb	$0, (VEC_SIZE * 2)(%rax), %YMMZERO, %k1
	kmovd	%k1, %edx
	test	%edx, %edx
	jnz	L(exit_null_on_third_vector)

	vpcmpb	$0, (VEC_SIZE * 3)(%rax), %YMMZERO, %k2
	kmovd	%k2, %edx
	test	%edx, %edx
	jnz	L(exit_null_on_fourth_vector)

	vpcmpb	$0, (VEC_SIZE * 4)(%rax), %YMMZERO, %k3
	kmovd	%k3, %edx
	test	%edx, %edx
	jnz	L(exit_null_on_fifth_vector)

	vpcmpb	$0, (VEC_SIZE * 5)(%rax), %YMMZERO, %k4
	add	$(VEC_SIZE * 4), %rax
	kmovd	%k4, %edx
	test	%edx, %edx
	jnz	L(exit_null_on_second_vector)

	vpcmpb	$0, (VEC_SIZE * 2)(%rax), %YMMZERO, %k1
	kmovd	%k1, %edx
	test	%edx, %edx
	jnz	L(exit_null_on_third_vector)

	vpcmpb	$0, (VEC_SIZE * 3)(%rax), %YMMZERO, %k2
	kmovd	%k2, %edx
	test	%edx, %edx
	jnz	L(exit_null_on_fourth_vector)

	vpcmpb	$0, (VEC_SIZE * 4)(%rax), %YMMZERO, %k3
	kmovd	%k3, %edx
	test	%edx, %edx
	jnz	L(exit_null_on_fifth_vector)

	vpcmpb	$0, (VEC_SIZE * 5)(%rax), %YMMZERO, %k4
	kmovd	%k4, %edx
	add	$(VEC_SIZE * 4), %rax
	test	%edx, %edx
	jnz	L(exit_null_on_second_vector)

	vpcmpb	$0, (VEC_SIZE * 2)(%rax), %YMMZERO, %k1
	kmovd	%k1, %edx
	test	%edx, %edx
	jnz	L(exit_null_on_third_vector)

	vpcmpb	$0, (VEC_SIZE * 3)(%rax), %YMMZERO, %k2
	kmovd	%k2, %edx
	test	%edx, %edx
	jnz	L(exit_null_on_fourth_vector)

	vpcmpb	$0, (VEC_SIZE * 4)(%rax), %YMMZERO, %k3
	kmovd	%k3, %edx
	test	%edx, %edx
	jnz	L(exit_null_on_fifth_vector)

	vpcmpb	$0, (VEC_SIZE * 5)(%rax), %YMMZERO, %k4
	add	$(VEC_SIZE * 4), %rax
	kmovd	%k4, %edx
	test	%edx, %edx
	jnz	L(exit_null_on_second_vector)

	vpcmpb	$0, (VEC_SIZE * 2)(%rax), %YMMZERO, %k1
	kmovd	%k1, %edx
	test	%edx, %edx
	jnz	L(exit_null_on_third_vector)

	vpcmpb	$0, (VEC_SIZE * 3)(%rax), %YMMZERO, %k2
	kmovd	%k2, %edx
	test	%edx, %edx
	jnz	L(exit_null_on_fourth_vector)

	vpcmpb	$0, (VEC_SIZE * 4)(%rax), %YMMZERO, %k3
	kmovd	%k3, %edx
	test	%edx, %edx
	jnz	L(exit_null_on_fifth_vector)

	test	$((VEC_SIZE * 4) - 1), %rax
	jz	L(align_four_vec_loop)

	vpcmpb	$0, (VEC_SIZE * 5)(%rax), %YMMZERO, %k4
	add	$(VEC_SIZE * 5), %rax
	kmovd	%k4, %edx
	test	%edx, %edx
	jnz	L(exit)

	test	$((VEC_SIZE * 4) - 1), %rax
	jz	L(align_four_vec_loop)

	vpcmpb	$0, VEC_SIZE(%rax), %YMMZERO, %k0
	add	$VEC_SIZE, %rax
	kmovd	%k0, %edx
	test	%edx, %edx
	jnz	L(exit)

	test	$((VEC_SIZE * 4) - 1), %rax
	jz	L(align_four_vec_loop)

	vpcmpb	$0, VEC_SIZE(%rax), %YMMZERO, %k0
	add	$VEC_SIZE, %rax
	kmovd	%k0, %edx
	test	%edx, %edx
	jnz	L(exit)

	test	$((VEC_SIZE * 4) - 1), %rax
	jz	L(align_four_vec_loop)

	vpcmpb	$0, VEC_SIZE(%rax), %YMMZERO, %k1
	add	$VEC_SIZE, %rax
	kmovd	%k1, %edx
	test	%edx, %edx
	jnz	L(exit)

	add	$VEC_SIZE, %rax

	.p2align 4
L(align_four_vec_loop):
	VMOVA	(%rax), %YMM0
	VMOVA	(VEC_SIZE * 2)(%rax), %YMM1
	vpminub	VEC_SIZE(%rax), %YMM0, %YMM0
	vpminub	(VEC_SIZE * 3)(%rax), %YMM1, %YMM1
	vpminub	%YMM0, %YMM1, %YMM0
	/* If K0 != 0, there is a null byte.  */
	vpcmpb	$0, %YMM0, %YMMZERO, %k0
	add	$(VEC_SIZE * 4), %rax
	ktestd	%k0, %k0
	jz	L(align_four_vec_loop)

	vpcmpb	$0, -(VEC_SIZE * 4)(%rax), %YMMZERO, %k0
	sub	$(VEC_SIZE * 5), %rax
	kmovd	%k0, %edx
	test	%edx, %edx
	jnz	L(exit_null_on_second_vector)

	vpcmpb	$0, (VEC_SIZE * 2)(%rax), %YMMZERO, %k1
	kmovd	%k1, %edx
	test	%edx, %edx
	jnz	L(exit_null_on_third_vector)

	vpcmpb	$0, (VEC_SIZE * 3)(%rax), %YMMZERO, %k2
	kmovd	%k2, %edx
	test	%edx, %edx
	jnz	L(exit_null_on_fourth_vector)

	vpcmpb	$0, (VEC_SIZE * 4)(%rax), %YMMZERO, %k3
	kmovd	%k3, %edx
	sub	%rdi, %rax
	bsf	%rdx, %rdx
	add	%rdx, %rax
	add	$(VEC_SIZE * 4), %rax
	jmp	L(StartStrcpyPart)

	.p2align 4
L(exit):
	sub	%rdi, %rax
L(exit_null_on_first_vector):
	bsf	%rdx, %rdx
	add	%rdx, %rax
	jmp	L(StartStrcpyPart)

	.p2align 4
L(exit_null_on_second_vector):
	sub	%rdi, %rax
	bsf	%rdx, %rdx
	add	%rdx, %rax
	add	$VEC_SIZE, %rax
	jmp	L(StartStrcpyPart)

	.p2align 4
L(exit_null_on_third_vector):
	sub	%rdi, %rax
	bsf	%rdx, %rdx
	add	%rdx, %rax
	add	$(VEC_SIZE * 2), %rax
	jmp	L(StartStrcpyPart)

	.p2align 4
L(exit_null_on_fourth_vector):
	sub	%rdi, %rax
	bsf	%rdx, %rdx
	add	%rdx, %rax
	add	$(VEC_SIZE * 3), %rax
	jmp	L(StartStrcpyPart)

	.p2align 4
L(exit_null_on_fifth_vector):
	sub	%rdi, %rax
	bsf	%rdx, %rdx
	add	%rdx, %rax
	add	$(VEC_SIZE * 4), %rax

	.p2align 4
L(StartStrcpyPart):
	lea	(%r9, %rax), %rdi
	mov	%rsi, %rcx
	mov	%r9, %rax      /* save result */

# ifdef USE_AS_STRNCAT
	test	%r8, %r8
	jz	L(ExitZero)
#  define USE_AS_STRNCPY
# endif

# include "strcpy-evex.S"
#endif
