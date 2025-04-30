/* strrchr/wcsrchr optimized with 256-bit EVEX instructions.
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

# ifndef STRRCHR
#  define STRRCHR	__strrchr_evex
# endif

# define VMOVU		vmovdqu64
# define VMOVA		vmovdqa64

# ifdef USE_AS_WCSRCHR
#  define VPBROADCAST	vpbroadcastd
#  define VPCMP		vpcmpd
#  define SHIFT_REG	r8d
# else
#  define VPBROADCAST	vpbroadcastb
#  define VPCMP		vpcmpb
#  define SHIFT_REG	ecx
# endif

# define XMMZERO	xmm16
# define YMMZERO	ymm16
# define YMMMATCH	ymm17
# define YMM1		ymm18

# define VEC_SIZE	32

	.section .text.evex,"ax",@progbits
ENTRY (STRRCHR)
	movl	%edi, %ecx
	/* Broadcast CHAR to YMMMATCH.  */
	VPBROADCAST %esi, %YMMMATCH

	vpxorq	%XMMZERO, %XMMZERO, %XMMZERO

	/* Check if we may cross page boundary with one vector load.  */
	andl	$(2 * VEC_SIZE - 1), %ecx
	cmpl	$VEC_SIZE, %ecx
	ja	L(cros_page_boundary)

	VMOVU	(%rdi), %YMM1

	/* Each bit in K0 represents a null byte in YMM1.  */
	VPCMP	$0, %YMMZERO, %YMM1, %k0
	/* Each bit in K1 represents a CHAR in YMM1.  */
	VPCMP	$0, %YMMMATCH, %YMM1, %k1
	kmovd	%k0, %ecx
	kmovd	%k1, %eax

	addq	$VEC_SIZE, %rdi

	testl	%eax, %eax
	jnz	L(first_vec)

	testl	%ecx, %ecx
	jnz	L(return_null)

	andq	$-VEC_SIZE, %rdi
	xorl	%edx, %edx
	jmp	L(aligned_loop)

	.p2align 4
L(first_vec):
	/* Check if there is a null byte.  */
	testl	%ecx, %ecx
	jnz	L(char_and_nul_in_first_vec)

	/* Remember the match and keep searching.  */
	movl	%eax, %edx
	movq	%rdi, %rsi
	andq	$-VEC_SIZE, %rdi
	jmp	L(aligned_loop)

	.p2align 4
L(cros_page_boundary):
	andl	$(VEC_SIZE - 1), %ecx
	andq	$-VEC_SIZE, %rdi

# ifdef USE_AS_WCSRCHR
	/* NB: Divide shift count by 4 since each bit in K1 represent 4
	   bytes.  */
	movl	%ecx, %SHIFT_REG
	sarl	$2, %SHIFT_REG
# endif

	VMOVA	(%rdi), %YMM1

	/* Each bit in K0 represents a null byte in YMM1.  */
	VPCMP	$0, %YMMZERO, %YMM1, %k0
	/* Each bit in K1 represents a CHAR in YMM1.  */
	VPCMP	$0, %YMMMATCH, %YMM1, %k1
	kmovd	%k0, %edx
	kmovd	%k1, %eax

	shrxl	%SHIFT_REG, %edx, %edx
	shrxl	%SHIFT_REG, %eax, %eax
	addq	$VEC_SIZE, %rdi

	/* Check if there is a CHAR.  */
	testl	%eax, %eax
	jnz	L(found_char)

	testl	%edx, %edx
	jnz	L(return_null)

	jmp	L(aligned_loop)

	.p2align 4
L(found_char):
	testl	%edx, %edx
	jnz	L(char_and_nul)

	/* Remember the match and keep searching.  */
	movl	%eax, %edx
	leaq	(%rdi, %rcx), %rsi

	.p2align 4
L(aligned_loop):
	VMOVA	(%rdi), %YMM1
	addq	$VEC_SIZE, %rdi

	/* Each bit in K0 represents a null byte in YMM1.  */
	VPCMP	$0, %YMMZERO, %YMM1, %k0
	/* Each bit in K1 represents a CHAR in YMM1.  */
	VPCMP	$0, %YMMMATCH, %YMM1, %k1
	kmovd	%k0, %ecx
	kmovd	%k1, %eax
	orl	%eax, %ecx
	jnz	L(char_nor_null)

	VMOVA	(%rdi), %YMM1
	add	$VEC_SIZE, %rdi

	/* Each bit in K0 represents a null byte in YMM1.  */
	VPCMP	$0, %YMMZERO, %YMM1, %k0
	/* Each bit in K1 represents a CHAR in YMM1.  */
	VPCMP	$0, %YMMMATCH, %YMM1, %k1
	kmovd	%k0, %ecx
	kmovd	%k1, %eax
	orl	%eax, %ecx
	jnz	L(char_nor_null)

	VMOVA	(%rdi), %YMM1
	addq	$VEC_SIZE, %rdi

	/* Each bit in K0 represents a null byte in YMM1.  */
	VPCMP	$0, %YMMZERO, %YMM1, %k0
	/* Each bit in K1 represents a CHAR in YMM1.  */
	VPCMP	$0, %YMMMATCH, %YMM1, %k1
	kmovd	%k0, %ecx
	kmovd	%k1, %eax
	orl	%eax, %ecx
	jnz	L(char_nor_null)

	VMOVA	(%rdi), %YMM1
	addq	$VEC_SIZE, %rdi

	/* Each bit in K0 represents a null byte in YMM1.  */
	VPCMP	$0, %YMMZERO, %YMM1, %k0
	/* Each bit in K1 represents a CHAR in YMM1.  */
	VPCMP	$0, %YMMMATCH, %YMM1, %k1
	kmovd	%k0, %ecx
	kmovd	%k1, %eax
	orl	%eax, %ecx
	jz	L(aligned_loop)

	.p2align 4
L(char_nor_null):
	/* Find a CHAR or a null byte in a loop.  */
	testl	%eax, %eax
	jnz	L(match)
L(return_value):
	testl	%edx, %edx
	jz	L(return_null)
	movl	%edx, %eax
	movq	%rsi, %rdi
	bsrl	%eax, %eax
# ifdef USE_AS_WCSRCHR
	/* NB: Multiply wchar_t count by 4 to get the number of bytes.  */
	leaq	-VEC_SIZE(%rdi, %rax, 4), %rax
# else
	leaq	-VEC_SIZE(%rdi, %rax), %rax
# endif
	ret

	.p2align 4
L(match):
	/* Find a CHAR.  Check if there is a null byte.  */
	kmovd	%k0, %ecx
	testl	%ecx, %ecx
	jnz	L(find_nul)

	/* Remember the match and keep searching.  */
	movl	%eax, %edx
	movq	%rdi, %rsi
	jmp	L(aligned_loop)

	.p2align 4
L(find_nul):
	/* Mask out any matching bits after the null byte.  */
	movl	%ecx, %r8d
	subl	$1, %r8d
	xorl	%ecx, %r8d
	andl	%r8d, %eax
	testl	%eax, %eax
	/* If there is no CHAR here, return the remembered one.  */
	jz	L(return_value)
	bsrl	%eax, %eax
# ifdef USE_AS_WCSRCHR
	/* NB: Multiply wchar_t count by 4 to get the number of bytes.  */
	leaq	-VEC_SIZE(%rdi, %rax, 4), %rax
# else
	leaq	-VEC_SIZE(%rdi, %rax), %rax
# endif
	ret

	.p2align 4
L(char_and_nul):
	/* Find both a CHAR and a null byte.  */
	addq	%rcx, %rdi
	movl	%edx, %ecx
L(char_and_nul_in_first_vec):
	/* Mask out any matching bits after the null byte.  */
	movl	%ecx, %r8d
	subl	$1, %r8d
	xorl	%ecx, %r8d
	andl	%r8d, %eax
	testl	%eax, %eax
	/* Return null pointer if the null byte comes first.  */
	jz	L(return_null)
	bsrl	%eax, %eax
# ifdef USE_AS_WCSRCHR
	/* NB: Multiply wchar_t count by 4 to get the number of bytes.  */
	leaq	-VEC_SIZE(%rdi, %rax, 4), %rax
# else
	leaq	-VEC_SIZE(%rdi, %rax), %rax
# endif
	ret

	.p2align 4
L(return_null):
	xorl	%eax, %eax
	ret

END (STRRCHR)
#endif
