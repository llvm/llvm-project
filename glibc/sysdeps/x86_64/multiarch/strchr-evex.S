/* strchr/strchrnul optimized with 256-bit EVEX instructions.
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

# ifndef STRCHR
#  define STRCHR	__strchr_evex
# endif

# define VMOVU		vmovdqu64
# define VMOVA		vmovdqa64

# ifdef USE_AS_WCSCHR
#  define VPBROADCAST	vpbroadcastd
#  define VPCMP		vpcmpd
#  define VPMINU	vpminud
#  define CHAR_REG	esi
#  define SHIFT_REG	ecx
#  define CHAR_SIZE	4
# else
#  define VPBROADCAST	vpbroadcastb
#  define VPCMP		vpcmpb
#  define VPMINU	vpminub
#  define CHAR_REG	sil
#  define SHIFT_REG	edx
#  define CHAR_SIZE	1
# endif

# define XMMZERO	xmm16

# define YMMZERO	ymm16
# define YMM0		ymm17
# define YMM1		ymm18
# define YMM2		ymm19
# define YMM3		ymm20
# define YMM4		ymm21
# define YMM5		ymm22
# define YMM6		ymm23
# define YMM7		ymm24
# define YMM8		ymm25

# define VEC_SIZE 32
# define PAGE_SIZE 4096
# define CHAR_PER_VEC (VEC_SIZE / CHAR_SIZE)

	.section .text.evex,"ax",@progbits
ENTRY (STRCHR)
	/* Broadcast CHAR to YMM0.	*/
	VPBROADCAST	%esi, %YMM0
	movl	%edi, %eax
	andl	$(PAGE_SIZE - 1), %eax
	vpxorq	%XMMZERO, %XMMZERO, %XMMZERO

	/* Check if we cross page boundary with one vector load.
	   Otherwise it is safe to use an unaligned load.  */
	cmpl	$(PAGE_SIZE - VEC_SIZE), %eax
	ja	L(cross_page_boundary)

	/* Check the first VEC_SIZE bytes. Search for both CHAR and the
	   null bytes.  */
	VMOVU	(%rdi), %YMM1

	/* Leaves only CHARS matching esi as 0.  */
	vpxorq	%YMM1, %YMM0, %YMM2
	VPMINU	%YMM2, %YMM1, %YMM2
	/* Each bit in K0 represents a CHAR or a null byte in YMM1.  */
	VPCMP	$0, %YMMZERO, %YMM2, %k0
	kmovd	%k0, %eax
	testl	%eax, %eax
	jz	L(aligned_more)
	tzcntl	%eax, %eax
# ifdef USE_AS_WCSCHR
	/* NB: Multiply wchar_t count by 4 to get the number of bytes.
	 */
	leaq	(%rdi, %rax, CHAR_SIZE), %rax
# else
	addq	%rdi, %rax
# endif
# ifndef USE_AS_STRCHRNUL
	/* Found CHAR or the null byte.	 */
	cmp	(%rax), %CHAR_REG
	jne	L(zero)
# endif
	ret

	/* .p2align 5 helps keep performance more consistent if ENTRY()
	   alignment % 32 was either 16 or 0. As well this makes the
	   alignment % 32 of the loop_4x_vec fixed which makes tuning it
	   easier.  */
	.p2align 5
L(first_vec_x3):
	tzcntl	%eax, %eax
# ifndef USE_AS_STRCHRNUL
	/* Found CHAR or the null byte.	 */
	cmp	(VEC_SIZE * 3)(%rdi, %rax, CHAR_SIZE), %CHAR_REG
	jne	L(zero)
# endif
	/* NB: Multiply sizeof char type (1 or 4) to get the number of
	   bytes.  */
	leaq	(VEC_SIZE * 3)(%rdi, %rax, CHAR_SIZE), %rax
	ret

# ifndef USE_AS_STRCHRNUL
L(zero):
	xorl	%eax, %eax
	ret
# endif

	.p2align 4
L(first_vec_x4):
# ifndef USE_AS_STRCHRNUL
	/* Check to see if first match was CHAR (k0) or null (k1).  */
	kmovd	%k0, %eax
	tzcntl	%eax, %eax
	kmovd	%k1, %ecx
	/* bzhil will not be 0 if first match was null.  */
	bzhil	%eax, %ecx, %ecx
	jne	L(zero)
# else
	/* Combine CHAR and null matches.  */
	kord	%k0, %k1, %k0
	kmovd	%k0, %eax
	tzcntl	%eax, %eax
# endif
	/* NB: Multiply sizeof char type (1 or 4) to get the number of
	   bytes.  */
	leaq	(VEC_SIZE * 4)(%rdi, %rax, CHAR_SIZE), %rax
	ret

	.p2align 4
L(first_vec_x1):
	tzcntl	%eax, %eax
# ifndef USE_AS_STRCHRNUL
	/* Found CHAR or the null byte.	 */
	cmp	(VEC_SIZE)(%rdi, %rax, CHAR_SIZE), %CHAR_REG
	jne	L(zero)

# endif
	/* NB: Multiply sizeof char type (1 or 4) to get the number of
	   bytes.  */
	leaq	(VEC_SIZE)(%rdi, %rax, CHAR_SIZE), %rax
	ret

	.p2align 4
L(first_vec_x2):
# ifndef USE_AS_STRCHRNUL
	/* Check to see if first match was CHAR (k0) or null (k1).  */
	kmovd	%k0, %eax
	tzcntl	%eax, %eax
	kmovd	%k1, %ecx
	/* bzhil will not be 0 if first match was null.  */
	bzhil	%eax, %ecx, %ecx
	jne	L(zero)
# else
	/* Combine CHAR and null matches.  */
	kord	%k0, %k1, %k0
	kmovd	%k0, %eax
	tzcntl	%eax, %eax
# endif
	/* NB: Multiply sizeof char type (1 or 4) to get the number of
	   bytes.  */
	leaq	(VEC_SIZE * 2)(%rdi, %rax, CHAR_SIZE), %rax
	ret

	.p2align 4
L(aligned_more):
	/* Align data to VEC_SIZE.  */
	andq	$-VEC_SIZE, %rdi
L(cross_page_continue):
	/* Check the next 4 * VEC_SIZE. Only one VEC_SIZE at a time since
	   data is only aligned to VEC_SIZE. Use two alternating methods
	   for checking VEC to balance latency and port contention.  */

	/* This method has higher latency but has better port
	   distribution.  */
	VMOVA	(VEC_SIZE)(%rdi), %YMM1
	/* Leaves only CHARS matching esi as 0.  */
	vpxorq	%YMM1, %YMM0, %YMM2
	VPMINU	%YMM2, %YMM1, %YMM2
	/* Each bit in K0 represents a CHAR or a null byte in YMM1.  */
	VPCMP	$0, %YMMZERO, %YMM2, %k0
	kmovd	%k0, %eax
	testl	%eax, %eax
	jnz	L(first_vec_x1)

	/* This method has higher latency but has better port
	   distribution.  */
	VMOVA	(VEC_SIZE * 2)(%rdi), %YMM1
	/* Each bit in K0 represents a CHAR in YMM1.  */
	VPCMP	$0, %YMM1, %YMM0, %k0
	/* Each bit in K1 represents a CHAR in YMM1.  */
	VPCMP	$0, %YMM1, %YMMZERO, %k1
	kortestd	%k0, %k1
	jnz	L(first_vec_x2)

	VMOVA	(VEC_SIZE * 3)(%rdi), %YMM1
	/* Leaves only CHARS matching esi as 0.  */
	vpxorq	%YMM1, %YMM0, %YMM2
	VPMINU	%YMM2, %YMM1, %YMM2
	/* Each bit in K0 represents a CHAR or a null byte in YMM1.  */
	VPCMP	$0, %YMMZERO, %YMM2, %k0
	kmovd	%k0, %eax
	testl	%eax, %eax
	jnz	L(first_vec_x3)

	VMOVA	(VEC_SIZE * 4)(%rdi), %YMM1
	/* Each bit in K0 represents a CHAR in YMM1.  */
	VPCMP	$0, %YMM1, %YMM0, %k0
	/* Each bit in K1 represents a CHAR in YMM1.  */
	VPCMP	$0, %YMM1, %YMMZERO, %k1
	kortestd	%k0, %k1
	jnz	L(first_vec_x4)

	/* Align data to VEC_SIZE * 4 for the loop.  */
	addq	$VEC_SIZE, %rdi
	andq	$-(VEC_SIZE * 4), %rdi

	.p2align 4
L(loop_4x_vec):
	/* Check 4x VEC at a time. No penalty to imm32 offset with evex
	   encoding.  */
	VMOVA	(VEC_SIZE * 4)(%rdi), %YMM1
	VMOVA	(VEC_SIZE * 5)(%rdi), %YMM2
	VMOVA	(VEC_SIZE * 6)(%rdi), %YMM3
	VMOVA	(VEC_SIZE * 7)(%rdi), %YMM4

	/* For YMM1 and YMM3 use xor to set the CHARs matching esi to
	   zero.  */
	vpxorq	%YMM1, %YMM0, %YMM5
	/* For YMM2 and YMM4 cmp not equals to CHAR and store result in
	   k register. Its possible to save either 1 or 2 instructions
	   using cmp no equals method for either YMM1 or YMM1 and YMM3
	   respectively but bottleneck on p5 makes it not worth it.  */
	VPCMP	$4, %YMM0, %YMM2, %k2
	vpxorq	%YMM3, %YMM0, %YMM7
	VPCMP	$4, %YMM0, %YMM4, %k4

	/* Use min to select all zeros from either xor or end of string).
	 */
	VPMINU	%YMM1, %YMM5, %YMM1
	VPMINU	%YMM3, %YMM7, %YMM3

	/* Use min + zeromask to select for zeros. Since k2 and k4 will
	   have 0 as positions that matched with CHAR which will set
	   zero in the corresponding destination bytes in YMM2 / YMM4.
	 */
	VPMINU	%YMM1, %YMM2, %YMM2{%k2}{z}
	VPMINU	%YMM3, %YMM4, %YMM4
	VPMINU	%YMM2, %YMM4, %YMM4{%k4}{z}

	VPCMP	$0, %YMMZERO, %YMM4, %k1
	kmovd	%k1, %ecx
	subq	$-(VEC_SIZE * 4), %rdi
	testl	%ecx, %ecx
	jz	L(loop_4x_vec)

	VPCMP	$0, %YMMZERO, %YMM1, %k0
	kmovd	%k0, %eax
	testl	%eax, %eax
	jnz	L(last_vec_x1)

	VPCMP	$0, %YMMZERO, %YMM2, %k0
	kmovd	%k0, %eax
	testl	%eax, %eax
	jnz	L(last_vec_x2)

	VPCMP	$0, %YMMZERO, %YMM3, %k0
	kmovd	%k0, %eax
	/* Combine YMM3 matches (eax) with YMM4 matches (ecx).  */
# ifdef USE_AS_WCSCHR
	sall	$8, %ecx
	orl	%ecx, %eax
	tzcntl	%eax, %eax
# else
	salq	$32, %rcx
	orq	%rcx, %rax
	tzcntq	%rax, %rax
# endif
# ifndef USE_AS_STRCHRNUL
	/* Check if match was CHAR or null.  */
	cmp	(VEC_SIZE * 2)(%rdi, %rax, CHAR_SIZE), %CHAR_REG
	jne	L(zero_end)
# endif
	/* NB: Multiply sizeof char type (1 or 4) to get the number of
	   bytes.  */
	leaq	(VEC_SIZE * 2)(%rdi, %rax, CHAR_SIZE), %rax
	ret

# ifndef USE_AS_STRCHRNUL
L(zero_end):
	xorl	%eax, %eax
	ret
# endif

	.p2align 4
L(last_vec_x1):
	tzcntl	%eax, %eax
# ifndef USE_AS_STRCHRNUL
	/* Check if match was null.  */
	cmp	(%rdi, %rax, CHAR_SIZE), %CHAR_REG
	jne	L(zero_end)
# endif
	/* NB: Multiply sizeof char type (1 or 4) to get the number of
	   bytes.  */
	leaq	(%rdi, %rax, CHAR_SIZE), %rax
	ret

	.p2align 4
L(last_vec_x2):
	tzcntl	%eax, %eax
# ifndef USE_AS_STRCHRNUL
	/* Check if match was null.  */
	cmp	(VEC_SIZE)(%rdi, %rax, CHAR_SIZE), %CHAR_REG
	jne	L(zero_end)
# endif
	/* NB: Multiply sizeof char type (1 or 4) to get the number of
	   bytes.  */
	leaq	(VEC_SIZE)(%rdi, %rax, CHAR_SIZE), %rax
	ret

	/* Cold case for crossing page with first load.	 */
	.p2align 4
L(cross_page_boundary):
	movq	%rdi, %rdx
	/* Align rdi.  */
	andq	$-VEC_SIZE, %rdi
	VMOVA	(%rdi), %YMM1
	/* Leaves only CHARS matching esi as 0.  */
	vpxorq	%YMM1, %YMM0, %YMM2
	VPMINU	%YMM2, %YMM1, %YMM2
	/* Each bit in K0 represents a CHAR or a null byte in YMM1.  */
	VPCMP	$0, %YMMZERO, %YMM2, %k0
	kmovd	%k0, %eax
	/* Remove the leading bits.	 */
# ifdef USE_AS_WCSCHR
	movl	%edx, %SHIFT_REG
	/* NB: Divide shift count by 4 since each bit in K1 represent 4
	   bytes.  */
	sarl	$2, %SHIFT_REG
	andl	$(CHAR_PER_VEC - 1), %SHIFT_REG
# endif
	sarxl	%SHIFT_REG, %eax, %eax
	/* If eax is zero continue.  */
	testl	%eax, %eax
	jz	L(cross_page_continue)
	tzcntl	%eax, %eax
# ifndef USE_AS_STRCHRNUL
	/* Check to see if match was CHAR or null.  */
	cmp	(%rdx, %rax, CHAR_SIZE), %CHAR_REG
	jne	L(zero_end)
# endif
# ifdef USE_AS_WCSCHR
	/* NB: Multiply wchar_t count by 4 to get the number of
	   bytes.  */
	leaq	(%rdx, %rax, CHAR_SIZE), %rax
# else
	addq	%rdx, %rax
# endif
	ret

END (STRCHR)
# endif
