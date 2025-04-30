/* strcmp/wcscmp/strncmp/wcsncmp optimized with 256-bit EVEX instructions.
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

# ifndef STRCMP
#  define STRCMP	__strcmp_evex
# endif

# define PAGE_SIZE	4096

/* VEC_SIZE = Number of bytes in a ymm register */
# define VEC_SIZE	32

/* Shift for dividing by (VEC_SIZE * 4).  */
# define DIVIDE_BY_VEC_4_SHIFT	7
# if (VEC_SIZE * 4) != (1 << DIVIDE_BY_VEC_4_SHIFT)
#  error (VEC_SIZE * 4) != (1 << DIVIDE_BY_VEC_4_SHIFT)
# endif

# define VMOVU		vmovdqu64
# define VMOVA		vmovdqa64

# ifdef USE_AS_WCSCMP
/* Compare packed dwords.  */
#  define VPCMP		vpcmpd
#  define SHIFT_REG32	r8d
#  define SHIFT_REG64	r8
/* 1 dword char == 4 bytes.  */
#  define SIZE_OF_CHAR	4
# else
/* Compare packed bytes.  */
#  define VPCMP		vpcmpb
#  define SHIFT_REG32	ecx
#  define SHIFT_REG64	rcx
/* 1 byte char == 1 byte.  */
#  define SIZE_OF_CHAR	1
# endif

# define XMMZERO	xmm16
# define XMM0		xmm17
# define XMM1		xmm18

# define YMMZERO	ymm16
# define YMM0		ymm17
# define YMM1		ymm18
# define YMM2		ymm19
# define YMM3		ymm20
# define YMM4		ymm21
# define YMM5		ymm22
# define YMM6		ymm23
# define YMM7		ymm24

/* Warning!
           wcscmp/wcsncmp have to use SIGNED comparison for elements.
           strcmp/strncmp have to use UNSIGNED comparison for elements.
*/

/* The main idea of the string comparison (byte or dword) using 256-bit
   EVEX instructions consists of comparing (VPCMP) two ymm vectors. The
   latter can be on either packed bytes or dwords depending on
   USE_AS_WCSCMP. In order to check the null char, algorithm keeps the
   matched bytes/dwords, requiring 5 EVEX instructions (3 VPCMP and 2
   KORD). In general, the costs of comparing VEC_SIZE bytes (32-bytes)
   are 3 VPCMP and 2 KORD instructions, together with VMOVU and ktestd
   instructions.  Main loop (away from from page boundary) compares 4
   vectors are a time, effectively comparing 4 x VEC_SIZE bytes (128
   bytes) on each loop.

   The routine strncmp/wcsncmp (enabled by defining USE_AS_STRNCMP) logic
   is the same as strcmp, except that an a maximum offset is tracked.  If
   the maximum offset is reached before a difference is found, zero is
   returned.  */

	.section .text.evex,"ax",@progbits
ENTRY (STRCMP)
# ifdef USE_AS_STRNCMP
	/* Check for simple cases (0 or 1) in offset.  */
	cmp	$1, %RDX_LP
	je	L(char0)
	jb	L(zero)
#  ifdef USE_AS_WCSCMP
	/* Convert units: from wide to byte char.  */
	shl	$2, %RDX_LP
#  endif
	/* Register %r11 tracks the maximum offset.  */
	mov	%RDX_LP, %R11_LP
# endif
	movl	%edi, %eax
	xorl	%edx, %edx
	/* Make %XMMZERO (%YMMZERO) all zeros in this function.  */
	vpxorq	%XMMZERO, %XMMZERO, %XMMZERO
	orl	%esi, %eax
	andl	$(PAGE_SIZE - 1), %eax
	cmpl	$(PAGE_SIZE - (VEC_SIZE * 4)), %eax
	jg	L(cross_page)
	/* Start comparing 4 vectors.  */
	VMOVU	(%rdi), %YMM0
	VMOVU	(%rsi), %YMM1

	/* Each bit in K0 represents a mismatch in YMM0 and YMM1.  */
	VPCMP	$4, %YMM0, %YMM1, %k0

	/* Check for NULL in YMM0.  */
	VPCMP	$0, %YMMZERO, %YMM0, %k1
	/* Check for NULL in YMM1.  */
	VPCMP	$0, %YMMZERO, %YMM1, %k2
	/* Each bit in K1 represents a NULL in YMM0 or YMM1.  */
	kord	%k1, %k2, %k1

	/* Each bit in K1 represents:
	   1. A mismatch in YMM0 and YMM1.  Or
	   2. A NULL in YMM0 or YMM1.
	 */
	kord	%k0, %k1, %k1

	ktestd	%k1, %k1
	je	L(next_3_vectors)
	kmovd	%k1, %ecx
	tzcntl	%ecx, %edx
# ifdef USE_AS_WCSCMP
	/* NB: Multiply wchar_t count by 4 to get the number of bytes.  */
	sall	$2, %edx
# endif
# ifdef USE_AS_STRNCMP
	/* Return 0 if the mismatched index (%rdx) is after the maximum
	   offset (%r11).   */
	cmpq	%r11, %rdx
	jae	L(zero)
# endif
# ifdef USE_AS_WCSCMP
	xorl	%eax, %eax
	movl	(%rdi, %rdx), %ecx
	cmpl	(%rsi, %rdx), %ecx
	je	L(return)
L(wcscmp_return):
	setl	%al
	negl	%eax
	orl	$1, %eax
L(return):
# else
	movzbl	(%rdi, %rdx), %eax
	movzbl	(%rsi, %rdx), %edx
	subl	%edx, %eax
# endif
	ret

	.p2align 4
L(return_vec_size):
	kmovd	%k1, %ecx
	tzcntl	%ecx, %edx
# ifdef USE_AS_WCSCMP
	/* NB: Multiply wchar_t count by 4 to get the number of bytes.  */
	sall	$2, %edx
# endif
# ifdef USE_AS_STRNCMP
	/* Return 0 if the mismatched index (%rdx + VEC_SIZE) is after
	   the maximum offset (%r11).  */
	addq	$VEC_SIZE, %rdx
	cmpq	%r11, %rdx
	jae	L(zero)
#  ifdef USE_AS_WCSCMP
	xorl	%eax, %eax
	movl	(%rdi, %rdx), %ecx
	cmpl	(%rsi, %rdx), %ecx
	jne	L(wcscmp_return)
#  else
	movzbl	(%rdi, %rdx), %eax
	movzbl	(%rsi, %rdx), %edx
	subl	%edx, %eax
#  endif
# else
#  ifdef USE_AS_WCSCMP
	xorl	%eax, %eax
	movl	VEC_SIZE(%rdi, %rdx), %ecx
	cmpl	VEC_SIZE(%rsi, %rdx), %ecx
	jne	L(wcscmp_return)
#  else
	movzbl	VEC_SIZE(%rdi, %rdx), %eax
	movzbl	VEC_SIZE(%rsi, %rdx), %edx
	subl	%edx, %eax
#  endif
# endif
	ret

	.p2align 4
L(return_2_vec_size):
	kmovd	%k1, %ecx
	tzcntl	%ecx, %edx
# ifdef USE_AS_WCSCMP
	/* NB: Multiply wchar_t count by 4 to get the number of bytes.  */
	sall	$2, %edx
# endif
# ifdef USE_AS_STRNCMP
	/* Return 0 if the mismatched index (%rdx + 2 * VEC_SIZE) is
	   after the maximum offset (%r11).  */
	addq	$(VEC_SIZE * 2), %rdx
	cmpq	%r11, %rdx
	jae	L(zero)
#  ifdef USE_AS_WCSCMP
	xorl	%eax, %eax
	movl	(%rdi, %rdx), %ecx
	cmpl	(%rsi, %rdx), %ecx
	jne	L(wcscmp_return)
#  else
	movzbl	(%rdi, %rdx), %eax
	movzbl	(%rsi, %rdx), %edx
	subl	%edx, %eax
#  endif
# else
#  ifdef USE_AS_WCSCMP
	xorl	%eax, %eax
	movl	(VEC_SIZE * 2)(%rdi, %rdx), %ecx
	cmpl	(VEC_SIZE * 2)(%rsi, %rdx), %ecx
	jne	L(wcscmp_return)
#  else
	movzbl	(VEC_SIZE * 2)(%rdi, %rdx), %eax
	movzbl	(VEC_SIZE * 2)(%rsi, %rdx), %edx
	subl	%edx, %eax
#  endif
# endif
	ret

	.p2align 4
L(return_3_vec_size):
	kmovd	%k1, %ecx
	tzcntl	%ecx, %edx
# ifdef USE_AS_WCSCMP
	/* NB: Multiply wchar_t count by 4 to get the number of bytes.  */
	sall	$2, %edx
# endif
# ifdef USE_AS_STRNCMP
	/* Return 0 if the mismatched index (%rdx + 3 * VEC_SIZE) is
	   after the maximum offset (%r11).  */
	addq	$(VEC_SIZE * 3), %rdx
	cmpq	%r11, %rdx
	jae	L(zero)
#  ifdef USE_AS_WCSCMP
	xorl	%eax, %eax
	movl	(%rdi, %rdx), %ecx
	cmpl	(%rsi, %rdx), %ecx
	jne	L(wcscmp_return)
#  else
	movzbl	(%rdi, %rdx), %eax
	movzbl	(%rsi, %rdx), %edx
	subl	%edx, %eax
#  endif
# else
#  ifdef USE_AS_WCSCMP
	xorl	%eax, %eax
	movl	(VEC_SIZE * 3)(%rdi, %rdx), %ecx
	cmpl	(VEC_SIZE * 3)(%rsi, %rdx), %ecx
	jne	L(wcscmp_return)
#  else
	movzbl	(VEC_SIZE * 3)(%rdi, %rdx), %eax
	movzbl	(VEC_SIZE * 3)(%rsi, %rdx), %edx
	subl	%edx, %eax
#  endif
# endif
	ret

	.p2align 4
L(next_3_vectors):
	VMOVU	VEC_SIZE(%rdi), %YMM0
	VMOVU	VEC_SIZE(%rsi), %YMM1
	/* Each bit in K0 represents a mismatch in YMM0 and YMM1.  */
	VPCMP	$4, %YMM0, %YMM1, %k0
	VPCMP	$0, %YMMZERO, %YMM0, %k1
	VPCMP	$0, %YMMZERO, %YMM1, %k2
	/* Each bit in K1 represents a NULL in YMM0 or YMM1.  */
	kord	%k1, %k2, %k1
	/* Each bit in K1 represents a NULL or a mismatch.  */
	kord	%k0, %k1, %k1
	ktestd	%k1, %k1
	jne	L(return_vec_size)

	VMOVU	(VEC_SIZE * 2)(%rdi), %YMM2
	VMOVU	(VEC_SIZE * 3)(%rdi), %YMM3
	VMOVU	(VEC_SIZE * 2)(%rsi), %YMM4
	VMOVU	(VEC_SIZE * 3)(%rsi), %YMM5

	/* Each bit in K0 represents a mismatch in YMM2 and YMM4.  */
	VPCMP	$4, %YMM2, %YMM4, %k0
	VPCMP	$0, %YMMZERO, %YMM2, %k1
	VPCMP	$0, %YMMZERO, %YMM4, %k2
	/* Each bit in K1 represents a NULL in YMM2 or YMM4.  */
	kord	%k1, %k2, %k1
	/* Each bit in K1 represents a NULL or a mismatch.  */
	kord	%k0, %k1, %k1
	ktestd	%k1, %k1
	jne	L(return_2_vec_size)

	/* Each bit in K0 represents a mismatch in YMM3 and YMM5.  */
	VPCMP	$4, %YMM3, %YMM5, %k0
	VPCMP	$0, %YMMZERO, %YMM3, %k1
	VPCMP	$0, %YMMZERO, %YMM5, %k2
	/* Each bit in K1 represents a NULL in YMM3 or YMM5.  */
	kord	%k1, %k2, %k1
	/* Each bit in K1 represents a NULL or a mismatch.  */
	kord	%k0, %k1, %k1
	ktestd	%k1, %k1
	jne	L(return_3_vec_size)
L(main_loop_header):
	leaq	(VEC_SIZE * 4)(%rdi), %rdx
	movl	$PAGE_SIZE, %ecx
	/* Align load via RAX.  */
	andq	$-(VEC_SIZE * 4), %rdx
	subq	%rdi, %rdx
	leaq	(%rdi, %rdx), %rax
# ifdef USE_AS_STRNCMP
	/* Starting from this point, the maximum offset, or simply the
	   'offset', DECREASES by the same amount when base pointers are
	   moved forward.  Return 0 when:
	     1) On match: offset <= the matched vector index.
	     2) On mistmach, offset is before the mistmatched index.
	 */
	subq	%rdx, %r11
	jbe	L(zero)
# endif
	addq	%rsi, %rdx
	movq	%rdx, %rsi
	andl	$(PAGE_SIZE - 1), %esi
	/* Number of bytes before page crossing.  */
	subq	%rsi, %rcx
	/* Number of VEC_SIZE * 4 blocks before page crossing.  */
	shrq	$DIVIDE_BY_VEC_4_SHIFT, %rcx
	/* ESI: Number of VEC_SIZE * 4 blocks before page crossing.   */
	movl	%ecx, %esi
	jmp	L(loop_start)

	.p2align 4
L(loop):
# ifdef USE_AS_STRNCMP
	/* Base pointers are moved forward by 4 * VEC_SIZE.  Decrease
	   the maximum offset (%r11) by the same amount.  */
	subq	$(VEC_SIZE * 4), %r11
	jbe	L(zero)
# endif
	addq	$(VEC_SIZE * 4), %rax
	addq	$(VEC_SIZE * 4), %rdx
L(loop_start):
	testl	%esi, %esi
	leal	-1(%esi), %esi
	je	L(loop_cross_page)
L(back_to_loop):
	/* Main loop, comparing 4 vectors are a time.  */
	VMOVA	(%rax), %YMM0
	VMOVA	VEC_SIZE(%rax), %YMM2
	VMOVA	(VEC_SIZE * 2)(%rax), %YMM4
	VMOVA	(VEC_SIZE * 3)(%rax), %YMM6
	VMOVU	(%rdx), %YMM1
	VMOVU	VEC_SIZE(%rdx), %YMM3
	VMOVU	(VEC_SIZE * 2)(%rdx), %YMM5
	VMOVU	(VEC_SIZE * 3)(%rdx), %YMM7

	VPCMP	$4, %YMM0, %YMM1, %k0
	VPCMP	$0, %YMMZERO, %YMM0, %k1
	VPCMP	$0, %YMMZERO, %YMM1, %k2
	kord	%k1, %k2, %k1
	/* Each bit in K4 represents a NULL or a mismatch in YMM0 and
	   YMM1.  */
	kord	%k0, %k1, %k4

	VPCMP	$4, %YMM2, %YMM3, %k0
	VPCMP	$0, %YMMZERO, %YMM2, %k1
	VPCMP	$0, %YMMZERO, %YMM3, %k2
	kord	%k1, %k2, %k1
	/* Each bit in K5 represents a NULL or a mismatch in YMM2 and
	   YMM3.  */
	kord	%k0, %k1, %k5

	VPCMP	$4, %YMM4, %YMM5, %k0
	VPCMP	$0, %YMMZERO, %YMM4, %k1
	VPCMP	$0, %YMMZERO, %YMM5, %k2
	kord	%k1, %k2, %k1
	/* Each bit in K6 represents a NULL or a mismatch in YMM4 and
	   YMM5.  */
	kord	%k0, %k1, %k6

	VPCMP	$4, %YMM6, %YMM7, %k0
	VPCMP	$0, %YMMZERO, %YMM6, %k1
	VPCMP	$0, %YMMZERO, %YMM7, %k2
	kord	%k1, %k2, %k1
	/* Each bit in K7 represents a NULL or a mismatch in YMM6 and
	   YMM7.  */
	kord	%k0, %k1, %k7

	kord	%k4, %k5, %k0
	kord	%k6, %k7, %k1

	/* Test each mask (32 bits) individually because for VEC_SIZE
	   == 32 is not possible to OR the four masks and keep all bits
	   in a 64-bit integer register, differing from SSE2 strcmp
	   where ORing is possible.  */
	kortestd %k0, %k1
	je	L(loop)
	ktestd	%k4, %k4
	je	L(test_vec)
	kmovd	%k4, %edi
	tzcntl	%edi, %ecx
# ifdef USE_AS_WCSCMP
	/* NB: Multiply wchar_t count by 4 to get the number of bytes.  */
	sall	$2, %ecx
# endif
# ifdef USE_AS_STRNCMP
	cmpq	%rcx, %r11
	jbe	L(zero)
#  ifdef USE_AS_WCSCMP
	movq	%rax, %rsi
	xorl	%eax, %eax
	movl	(%rsi, %rcx), %edi
	cmpl	(%rdx, %rcx), %edi
	jne	L(wcscmp_return)
#  else
	movzbl	(%rax, %rcx), %eax
	movzbl	(%rdx, %rcx), %edx
	subl	%edx, %eax
#  endif
# else
#  ifdef USE_AS_WCSCMP
	movq	%rax, %rsi
	xorl	%eax, %eax
	movl	(%rsi, %rcx), %edi
	cmpl	(%rdx, %rcx), %edi
	jne	L(wcscmp_return)
#  else
	movzbl	(%rax, %rcx), %eax
	movzbl	(%rdx, %rcx), %edx
	subl	%edx, %eax
#  endif
# endif
	ret

	.p2align 4
L(test_vec):
# ifdef USE_AS_STRNCMP
	/* The first vector matched.  Return 0 if the maximum offset
	   (%r11) <= VEC_SIZE.  */
	cmpq	$VEC_SIZE, %r11
	jbe	L(zero)
# endif
	ktestd	%k5, %k5
	je	L(test_2_vec)
	kmovd	%k5, %ecx
	tzcntl	%ecx, %edi
# ifdef USE_AS_WCSCMP
	/* NB: Multiply wchar_t count by 4 to get the number of bytes.  */
	sall	$2, %edi
# endif
# ifdef USE_AS_STRNCMP
	addq	$VEC_SIZE, %rdi
	cmpq	%rdi, %r11
	jbe	L(zero)
#  ifdef USE_AS_WCSCMP
	movq	%rax, %rsi
	xorl	%eax, %eax
	movl	(%rsi, %rdi), %ecx
	cmpl	(%rdx, %rdi), %ecx
	jne	L(wcscmp_return)
#  else
	movzbl	(%rax, %rdi), %eax
	movzbl	(%rdx, %rdi), %edx
	subl	%edx, %eax
#  endif
# else
#  ifdef USE_AS_WCSCMP
	movq	%rax, %rsi
	xorl	%eax, %eax
	movl	VEC_SIZE(%rsi, %rdi), %ecx
	cmpl	VEC_SIZE(%rdx, %rdi), %ecx
	jne	L(wcscmp_return)
#  else
	movzbl	VEC_SIZE(%rax, %rdi), %eax
	movzbl	VEC_SIZE(%rdx, %rdi), %edx
	subl	%edx, %eax
#  endif
# endif
	ret

	.p2align 4
L(test_2_vec):
# ifdef USE_AS_STRNCMP
	/* The first 2 vectors matched.  Return 0 if the maximum offset
	   (%r11) <= 2 * VEC_SIZE.  */
	cmpq	$(VEC_SIZE * 2), %r11
	jbe	L(zero)
# endif
	ktestd	%k6, %k6
	je	L(test_3_vec)
	kmovd	%k6, %ecx
	tzcntl	%ecx, %edi
# ifdef USE_AS_WCSCMP
	/* NB: Multiply wchar_t count by 4 to get the number of bytes.  */
	sall	$2, %edi
# endif
# ifdef USE_AS_STRNCMP
	addq	$(VEC_SIZE * 2), %rdi
	cmpq	%rdi, %r11
	jbe	L(zero)
#  ifdef USE_AS_WCSCMP
	movq	%rax, %rsi
	xorl	%eax, %eax
	movl	(%rsi, %rdi), %ecx
	cmpl	(%rdx, %rdi), %ecx
	jne	L(wcscmp_return)
#  else
	movzbl	(%rax, %rdi), %eax
	movzbl	(%rdx, %rdi), %edx
	subl	%edx, %eax
#  endif
# else
#  ifdef USE_AS_WCSCMP
	movq	%rax, %rsi
	xorl	%eax, %eax
	movl	(VEC_SIZE * 2)(%rsi, %rdi), %ecx
	cmpl	(VEC_SIZE * 2)(%rdx, %rdi), %ecx
	jne	L(wcscmp_return)
#  else
	movzbl	(VEC_SIZE * 2)(%rax, %rdi), %eax
	movzbl	(VEC_SIZE * 2)(%rdx, %rdi), %edx
	subl	%edx, %eax
#  endif
# endif
	ret

	.p2align 4
L(test_3_vec):
# ifdef USE_AS_STRNCMP
	/* The first 3 vectors matched.  Return 0 if the maximum offset
	   (%r11) <= 3 * VEC_SIZE.  */
	cmpq	$(VEC_SIZE * 3), %r11
	jbe	L(zero)
# endif
	kmovd	%k7, %esi
	tzcntl	%esi, %ecx
# ifdef USE_AS_WCSCMP
	/* NB: Multiply wchar_t count by 4 to get the number of bytes.  */
	sall	$2, %ecx
# endif
# ifdef USE_AS_STRNCMP
	addq	$(VEC_SIZE * 3), %rcx
	cmpq	%rcx, %r11
	jbe	L(zero)
#  ifdef USE_AS_WCSCMP
	movq	%rax, %rsi
	xorl	%eax, %eax
	movl	(%rsi, %rcx), %esi
	cmpl	(%rdx, %rcx), %esi
	jne	L(wcscmp_return)
#  else
	movzbl	(%rax, %rcx), %eax
	movzbl	(%rdx, %rcx), %edx
	subl	%edx, %eax
#  endif
# else
#  ifdef USE_AS_WCSCMP
	movq	%rax, %rsi
	xorl	%eax, %eax
	movl	(VEC_SIZE * 3)(%rsi, %rcx), %esi
	cmpl	(VEC_SIZE * 3)(%rdx, %rcx), %esi
	jne	L(wcscmp_return)
#  else
	movzbl	(VEC_SIZE * 3)(%rax, %rcx), %eax
	movzbl	(VEC_SIZE * 3)(%rdx, %rcx), %edx
	subl	%edx, %eax
#  endif
# endif
	ret

	.p2align 4
L(loop_cross_page):
	xorl	%r10d, %r10d
	movq	%rdx, %rcx
	/* Align load via RDX.  We load the extra ECX bytes which should
	   be ignored.  */
	andl	$((VEC_SIZE * 4) - 1), %ecx
	/* R10 is -RCX.  */
	subq	%rcx, %r10

	/* This works only if VEC_SIZE * 2 == 64. */
# if (VEC_SIZE * 2) != 64
#  error (VEC_SIZE * 2) != 64
# endif

	/* Check if the first VEC_SIZE * 2 bytes should be ignored.  */
	cmpl	$(VEC_SIZE * 2), %ecx
	jge	L(loop_cross_page_2_vec)

	VMOVU	(%rax, %r10), %YMM2
	VMOVU	VEC_SIZE(%rax, %r10), %YMM3
	VMOVU	(%rdx, %r10), %YMM4
	VMOVU	VEC_SIZE(%rdx, %r10), %YMM5

	VPCMP	$4, %YMM4, %YMM2, %k0
	VPCMP	$0, %YMMZERO, %YMM2, %k1
	VPCMP	$0, %YMMZERO, %YMM4, %k2
	kord	%k1, %k2, %k1
	/* Each bit in K1 represents a NULL or a mismatch in YMM2 and
	   YMM4.  */
	kord	%k0, %k1, %k1

	VPCMP	$4, %YMM5, %YMM3, %k3
	VPCMP	$0, %YMMZERO, %YMM3, %k4
	VPCMP	$0, %YMMZERO, %YMM5, %k5
	kord	%k4, %k5, %k4
	/* Each bit in K3 represents a NULL or a mismatch in YMM3 and
	   YMM5.  */
	kord	%k3, %k4, %k3

# ifdef USE_AS_WCSCMP
	/* NB: Each bit in K1/K3 represents 4-byte element.  */
	kshiftlw $8, %k3, %k2
	/* NB: Divide shift count by 4 since each bit in K1 represent 4
	   bytes.  */
	movl	%ecx, %SHIFT_REG32
	sarl	$2, %SHIFT_REG32
# else
	kshiftlq $32, %k3, %k2
# endif

	/* Each bit in K1 represents a NULL or a mismatch.  */
	korq	%k1, %k2, %k1
	kmovq	%k1, %rdi

	/* Since ECX < VEC_SIZE * 2, simply skip the first ECX bytes.  */
	shrxq	%SHIFT_REG64, %rdi, %rdi
	testq	%rdi, %rdi
	je	L(loop_cross_page_2_vec)
	tzcntq	%rdi, %rcx
# ifdef USE_AS_WCSCMP
	/* NB: Multiply wchar_t count by 4 to get the number of bytes.  */
	sall	$2, %ecx
# endif
# ifdef USE_AS_STRNCMP
	cmpq	%rcx, %r11
	jbe	L(zero)
#  ifdef USE_AS_WCSCMP
	movq	%rax, %rsi
	xorl	%eax, %eax
	movl	(%rsi, %rcx), %edi
	cmpl	(%rdx, %rcx), %edi
	jne	L(wcscmp_return)
#  else
	movzbl	(%rax, %rcx), %eax
	movzbl	(%rdx, %rcx), %edx
	subl	%edx, %eax
#  endif
# else
#  ifdef USE_AS_WCSCMP
	movq	%rax, %rsi
	xorl	%eax, %eax
	movl	(%rsi, %rcx), %edi
	cmpl	(%rdx, %rcx), %edi
	jne	L(wcscmp_return)
#  else
	movzbl	(%rax, %rcx), %eax
	movzbl	(%rdx, %rcx), %edx
	subl	%edx, %eax
#  endif
# endif
	ret

	.p2align 4
L(loop_cross_page_2_vec):
	/* The first VEC_SIZE * 2 bytes match or are ignored.  */
	VMOVU	(VEC_SIZE * 2)(%rax, %r10), %YMM0
	VMOVU	(VEC_SIZE * 3)(%rax, %r10), %YMM1
	VMOVU	(VEC_SIZE * 2)(%rdx, %r10), %YMM2
	VMOVU	(VEC_SIZE * 3)(%rdx, %r10), %YMM3

	VPCMP	$4, %YMM0, %YMM2, %k0
	VPCMP	$0, %YMMZERO, %YMM0, %k1
	VPCMP	$0, %YMMZERO, %YMM2, %k2
	kord	%k1, %k2, %k1
	/* Each bit in K1 represents a NULL or a mismatch in YMM0 and
	   YMM2.  */
	kord	%k0, %k1, %k1

	VPCMP	$4, %YMM1, %YMM3, %k3
	VPCMP	$0, %YMMZERO, %YMM1, %k4
	VPCMP	$0, %YMMZERO, %YMM3, %k5
	kord	%k4, %k5, %k4
	/* Each bit in K3 represents a NULL or a mismatch in YMM1 and
	   YMM3.  */
	kord	%k3, %k4, %k3

# ifdef USE_AS_WCSCMP
	/* NB: Each bit in K1/K3 represents 4-byte element.  */
	kshiftlw $8, %k3, %k2
# else
	kshiftlq $32, %k3, %k2
# endif

	/* Each bit in K1 represents a NULL or a mismatch.  */
	korq	%k1, %k2, %k1
	kmovq	%k1, %rdi

	xorl	%r8d, %r8d
	/* If ECX > VEC_SIZE * 2, skip ECX - (VEC_SIZE * 2) bytes.  */
	subl	$(VEC_SIZE * 2), %ecx
	jle	1f
	/* R8 has number of bytes skipped.  */
	movl	%ecx, %r8d
# ifdef USE_AS_WCSCMP
	/* NB: Divide shift count by 4 since each bit in K1 represent 4
	   bytes.  */
	sarl	$2, %ecx
# endif
	/* Skip ECX bytes.  */
	shrq	%cl, %rdi
1:
	/* Before jumping back to the loop, set ESI to the number of
	   VEC_SIZE * 4 blocks before page crossing.  */
	movl	$(PAGE_SIZE / (VEC_SIZE * 4) - 1), %esi

	testq	%rdi, %rdi
# ifdef USE_AS_STRNCMP
	/* At this point, if %rdi value is 0, it already tested
	   VEC_SIZE*4+%r10 byte starting from %rax. This label
	   checks whether strncmp maximum offset reached or not.  */
	je	L(string_nbyte_offset_check)
# else
	je	L(back_to_loop)
# endif
	tzcntq	%rdi, %rcx
# ifdef USE_AS_WCSCMP
	/* NB: Multiply wchar_t count by 4 to get the number of bytes.  */
	sall	$2, %ecx
# endif
	addq	%r10, %rcx
	/* Adjust for number of bytes skipped.  */
	addq	%r8, %rcx
# ifdef USE_AS_STRNCMP
	addq	$(VEC_SIZE * 2), %rcx
	subq	%rcx, %r11
	jbe	L(zero)
#  ifdef USE_AS_WCSCMP
	movq	%rax, %rsi
	xorl	%eax, %eax
	movl	(%rsi, %rcx), %edi
	cmpl	(%rdx, %rcx), %edi
	jne	L(wcscmp_return)
#  else
	movzbl	(%rax, %rcx), %eax
	movzbl	(%rdx, %rcx), %edx
	subl	%edx, %eax
#  endif
# else
#  ifdef USE_AS_WCSCMP
	movq	%rax, %rsi
	xorl	%eax, %eax
	movl	(VEC_SIZE * 2)(%rsi, %rcx), %edi
	cmpl	(VEC_SIZE * 2)(%rdx, %rcx), %edi
	jne	L(wcscmp_return)
#  else
	movzbl	(VEC_SIZE * 2)(%rax, %rcx), %eax
	movzbl	(VEC_SIZE * 2)(%rdx, %rcx), %edx
	subl	%edx, %eax
#  endif
# endif
	ret

# ifdef USE_AS_STRNCMP
L(string_nbyte_offset_check):
	leaq	(VEC_SIZE * 4)(%r10), %r10
	cmpq	%r10, %r11
	jbe	L(zero)
	jmp	L(back_to_loop)
# endif

	.p2align 4
L(cross_page_loop):
	/* Check one byte/dword at a time.  */
# ifdef USE_AS_WCSCMP
	cmpl	%ecx, %eax
# else
	subl	%ecx, %eax
# endif
	jne	L(different)
	addl	$SIZE_OF_CHAR, %edx
	cmpl	$(VEC_SIZE * 4), %edx
	je	L(main_loop_header)
# ifdef USE_AS_STRNCMP
	cmpq	%r11, %rdx
	jae	L(zero)
# endif
# ifdef USE_AS_WCSCMP
	movl	(%rdi, %rdx), %eax
	movl	(%rsi, %rdx), %ecx
# else
	movzbl	(%rdi, %rdx), %eax
	movzbl	(%rsi, %rdx), %ecx
# endif
	/* Check null char.  */
	testl	%eax, %eax
	jne	L(cross_page_loop)
	/* Since %eax == 0, subtract is OK for both SIGNED and UNSIGNED
	   comparisons.  */
	subl	%ecx, %eax
# ifndef USE_AS_WCSCMP
L(different):
# endif
	ret

# ifdef USE_AS_WCSCMP
	.p2align 4
L(different):
	/* Use movl to avoid modifying EFLAGS.  */
	movl	$0, %eax
	setl	%al
	negl	%eax
	orl	$1, %eax
	ret
# endif

# ifdef USE_AS_STRNCMP
	.p2align 4
L(zero):
	xorl	%eax, %eax
	ret

	.p2align 4
L(char0):
#  ifdef USE_AS_WCSCMP
	xorl	%eax, %eax
	movl	(%rdi), %ecx
	cmpl	(%rsi), %ecx
	jne	L(wcscmp_return)
#  else
	movzbl	(%rsi), %ecx
	movzbl	(%rdi), %eax
	subl	%ecx, %eax
#  endif
	ret
# endif

	.p2align 4
L(last_vector):
	addq	%rdx, %rdi
	addq	%rdx, %rsi
# ifdef USE_AS_STRNCMP
	subq	%rdx, %r11
# endif
	tzcntl	%ecx, %edx
# ifdef USE_AS_WCSCMP
	/* NB: Multiply wchar_t count by 4 to get the number of bytes.  */
	sall	$2, %edx
# endif
# ifdef USE_AS_STRNCMP
	cmpq	%r11, %rdx
	jae	L(zero)
# endif
# ifdef USE_AS_WCSCMP
	xorl	%eax, %eax
	movl	(%rdi, %rdx), %ecx
	cmpl	(%rsi, %rdx), %ecx
	jne	L(wcscmp_return)
# else
	movzbl	(%rdi, %rdx), %eax
	movzbl	(%rsi, %rdx), %edx
	subl	%edx, %eax
# endif
	ret

	/* Comparing on page boundary region requires special treatment:
	   It must done one vector at the time, starting with the wider
	   ymm vector if possible, if not, with xmm. If fetching 16 bytes
	   (xmm) still passes the boundary, byte comparison must be done.
	 */
	.p2align 4
L(cross_page):
	/* Try one ymm vector at a time.  */
	cmpl	$(PAGE_SIZE - VEC_SIZE), %eax
	jg	L(cross_page_1_vector)
L(loop_1_vector):
	VMOVU	(%rdi, %rdx), %YMM0
	VMOVU	(%rsi, %rdx), %YMM1

	/* Each bit in K0 represents a mismatch in YMM0 and YMM1.  */
	VPCMP	$4, %YMM0, %YMM1, %k0
	VPCMP	$0, %YMMZERO, %YMM0, %k1
	VPCMP	$0, %YMMZERO, %YMM1, %k2
	/* Each bit in K1 represents a NULL in YMM0 or YMM1.  */
	kord	%k1, %k2, %k1
	/* Each bit in K1 represents a NULL or a mismatch.  */
	kord	%k0, %k1, %k1
	kmovd	%k1, %ecx
	testl	%ecx, %ecx
	jne	L(last_vector)

	addl	$VEC_SIZE, %edx

	addl	$VEC_SIZE, %eax
# ifdef USE_AS_STRNCMP
	/* Return 0 if the current offset (%rdx) >= the maximum offset
	   (%r11).  */
	cmpq	%r11, %rdx
	jae	L(zero)
# endif
	cmpl	$(PAGE_SIZE - VEC_SIZE), %eax
	jle	L(loop_1_vector)
L(cross_page_1_vector):
	/* Less than 32 bytes to check, try one xmm vector.  */
	cmpl	$(PAGE_SIZE - 16), %eax
	jg	L(cross_page_1_xmm)
	VMOVU	(%rdi, %rdx), %XMM0
	VMOVU	(%rsi, %rdx), %XMM1

	/* Each bit in K0 represents a mismatch in XMM0 and XMM1.  */
	VPCMP	$4, %XMM0, %XMM1, %k0
	VPCMP	$0, %XMMZERO, %XMM0, %k1
	VPCMP	$0, %XMMZERO, %XMM1, %k2
	/* Each bit in K1 represents a NULL in XMM0 or XMM1.  */
	korw	%k1, %k2, %k1
	/* Each bit in K1 represents a NULL or a mismatch.  */
	korw	%k0, %k1, %k1
	kmovw	%k1, %ecx
	testl	%ecx, %ecx
	jne	L(last_vector)

	addl	$16, %edx
# ifndef USE_AS_WCSCMP
	addl	$16, %eax
# endif
# ifdef USE_AS_STRNCMP
	/* Return 0 if the current offset (%rdx) >= the maximum offset
	   (%r11).  */
	cmpq	%r11, %rdx
	jae	L(zero)
# endif

L(cross_page_1_xmm):
# ifndef USE_AS_WCSCMP
	/* Less than 16 bytes to check, try 8 byte vector.  NB: No need
	   for wcscmp nor wcsncmp since wide char is 4 bytes.   */
	cmpl	$(PAGE_SIZE - 8), %eax
	jg	L(cross_page_8bytes)
	vmovq	(%rdi, %rdx), %XMM0
	vmovq	(%rsi, %rdx), %XMM1

	/* Each bit in K0 represents a mismatch in XMM0 and XMM1.  */
	VPCMP	$4, %XMM0, %XMM1, %k0
	VPCMP	$0, %XMMZERO, %XMM0, %k1
	VPCMP	$0, %XMMZERO, %XMM1, %k2
	/* Each bit in K1 represents a NULL in XMM0 or XMM1.  */
	kord	%k1, %k2, %k1
	/* Each bit in K1 represents a NULL or a mismatch.  */
	kord	%k0, %k1, %k1
	kmovd	%k1, %ecx

# ifdef USE_AS_WCSCMP
	/* Only last 2 bits are valid.  */
	andl	$0x3, %ecx
# else
	/* Only last 8 bits are valid.  */
	andl	$0xff, %ecx
# endif

	testl	%ecx, %ecx
	jne	L(last_vector)

	addl	$8, %edx
	addl	$8, %eax
#  ifdef USE_AS_STRNCMP
	/* Return 0 if the current offset (%rdx) >= the maximum offset
	   (%r11).  */
	cmpq	%r11, %rdx
	jae	L(zero)
#  endif

L(cross_page_8bytes):
	/* Less than 8 bytes to check, try 4 byte vector.  */
	cmpl	$(PAGE_SIZE - 4), %eax
	jg	L(cross_page_4bytes)
	vmovd	(%rdi, %rdx), %XMM0
	vmovd	(%rsi, %rdx), %XMM1

	/* Each bit in K0 represents a mismatch in XMM0 and XMM1.  */
	VPCMP	$4, %XMM0, %XMM1, %k0
	VPCMP	$0, %XMMZERO, %XMM0, %k1
	VPCMP	$0, %XMMZERO, %XMM1, %k2
	/* Each bit in K1 represents a NULL in XMM0 or XMM1.  */
	kord	%k1, %k2, %k1
	/* Each bit in K1 represents a NULL or a mismatch.  */
	kord	%k0, %k1, %k1
	kmovd	%k1, %ecx

# ifdef USE_AS_WCSCMP
	/* Only the last bit is valid.  */
	andl	$0x1, %ecx
# else
	/* Only last 4 bits are valid.  */
	andl	$0xf, %ecx
# endif

	testl	%ecx, %ecx
	jne	L(last_vector)

	addl	$4, %edx
#  ifdef USE_AS_STRNCMP
	/* Return 0 if the current offset (%rdx) >= the maximum offset
	   (%r11).  */
	cmpq	%r11, %rdx
	jae	L(zero)
#  endif

L(cross_page_4bytes):
# endif
	/* Less than 4 bytes to check, try one byte/dword at a time.  */
# ifdef USE_AS_STRNCMP
	cmpq	%r11, %rdx
	jae	L(zero)
# endif
# ifdef USE_AS_WCSCMP
	movl	(%rdi, %rdx), %eax
	movl	(%rsi, %rdx), %ecx
# else
	movzbl	(%rdi, %rdx), %eax
	movzbl	(%rsi, %rdx), %ecx
# endif
	testl	%eax, %eax
	jne	L(cross_page_loop)
	subl	%ecx, %eax
	ret
END (STRCMP)
#endif
