/* strlen/strnlen/wcslen/wcsnlen optimized with 256-bit EVEX instructions.
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

# ifndef STRLEN
#  define STRLEN	__strlen_evex
# endif

# define VMOVA		vmovdqa64

# ifdef USE_AS_WCSLEN
#  define VPCMP		vpcmpd
#  define VPMINU	vpminud
#  define SHIFT_REG ecx
#  define CHAR_SIZE	4
# else
#  define VPCMP		vpcmpb
#  define VPMINU	vpminub
#  define SHIFT_REG edx
#  define CHAR_SIZE	1
# endif

# define XMMZERO	xmm16
# define YMMZERO	ymm16
# define YMM1		ymm17
# define YMM2		ymm18
# define YMM3		ymm19
# define YMM4		ymm20
# define YMM5		ymm21
# define YMM6		ymm22

# define VEC_SIZE 32
# define PAGE_SIZE 4096
# define CHAR_PER_VEC (VEC_SIZE / CHAR_SIZE)

	.section .text.evex,"ax",@progbits
ENTRY (STRLEN)
# ifdef USE_AS_STRNLEN
	/* Check zero length.  */
	test	%RSI_LP, %RSI_LP
	jz	L(zero)
#  ifdef __ILP32__
	/* Clear the upper 32 bits.  */
	movl	%esi, %esi
#  endif
	mov	%RSI_LP, %R8_LP
# endif
	movl	%edi, %eax
	vpxorq	%XMMZERO, %XMMZERO, %XMMZERO
	/* Clear high bits from edi. Only keeping bits relevant to page
	   cross check.  */
	andl	$(PAGE_SIZE - 1), %eax
	/* Check if we may cross page boundary with one vector load.  */
	cmpl	$(PAGE_SIZE - VEC_SIZE), %eax
	ja	L(cross_page_boundary)

	/* Check the first VEC_SIZE bytes.  Each bit in K0 represents a
	   null byte.  */
	VPCMP	$0, (%rdi), %YMMZERO, %k0
	kmovd	%k0, %eax
# ifdef USE_AS_STRNLEN
	/* If length < CHAR_PER_VEC handle special.  */
	cmpq	$CHAR_PER_VEC, %rsi
	jbe	L(first_vec_x0)
# endif
	testl	%eax, %eax
	jz	L(aligned_more)
	tzcntl	%eax, %eax
	ret
# ifdef USE_AS_STRNLEN
L(zero):
	xorl	%eax, %eax
	ret

	.p2align 4
L(first_vec_x0):
	/* Set bit for max len so that tzcnt will return min of max len
	   and position of first match.  */
	btsq	%rsi, %rax
	tzcntl	%eax, %eax
	ret
# endif

	.p2align 4
L(first_vec_x1):
	tzcntl	%eax, %eax
	/* Safe to use 32 bit instructions as these are only called for
	   size = [1, 159].  */
# ifdef USE_AS_STRNLEN
	/* Use ecx which was computed earlier to compute correct value.
	 */
	leal	-(CHAR_PER_VEC * 4 + 1)(%rcx, %rax), %eax
# else
	subl	%edx, %edi
#  ifdef USE_AS_WCSLEN
	/* NB: Divide bytes by 4 to get the wchar_t count.  */
	sarl	$2, %edi
#  endif
	leal	CHAR_PER_VEC(%rdi, %rax), %eax
# endif
	ret

	.p2align 4
L(first_vec_x2):
	tzcntl	%eax, %eax
	/* Safe to use 32 bit instructions as these are only called for
	   size = [1, 159].  */
# ifdef USE_AS_STRNLEN
	/* Use ecx which was computed earlier to compute correct value.
	 */
	leal	-(CHAR_PER_VEC * 3 + 1)(%rcx, %rax), %eax
# else
	subl	%edx, %edi
#  ifdef USE_AS_WCSLEN
	/* NB: Divide bytes by 4 to get the wchar_t count.  */
	sarl	$2, %edi
#  endif
	leal	(CHAR_PER_VEC * 2)(%rdi, %rax), %eax
# endif
	ret

	.p2align 4
L(first_vec_x3):
	tzcntl	%eax, %eax
	/* Safe to use 32 bit instructions as these are only called for
	   size = [1, 159].  */
# ifdef USE_AS_STRNLEN
	/* Use ecx which was computed earlier to compute correct value.
	 */
	leal	-(CHAR_PER_VEC * 2 + 1)(%rcx, %rax), %eax
# else
	subl	%edx, %edi
#  ifdef USE_AS_WCSLEN
	/* NB: Divide bytes by 4 to get the wchar_t count.  */
	sarl	$2, %edi
#  endif
	leal	(CHAR_PER_VEC * 3)(%rdi, %rax), %eax
# endif
	ret

	.p2align 4
L(first_vec_x4):
	tzcntl	%eax, %eax
	/* Safe to use 32 bit instructions as these are only called for
	   size = [1, 159].  */
# ifdef USE_AS_STRNLEN
	/* Use ecx which was computed earlier to compute correct value.
	 */
	leal	-(CHAR_PER_VEC + 1)(%rcx, %rax), %eax
# else
	subl	%edx, %edi
#  ifdef USE_AS_WCSLEN
	/* NB: Divide bytes by 4 to get the wchar_t count.  */
	sarl	$2, %edi
#  endif
	leal	(CHAR_PER_VEC * 4)(%rdi, %rax), %eax
# endif
	ret

	.p2align 5
L(aligned_more):
	movq	%rdi, %rdx
	/* Align data to VEC_SIZE.  */
	andq	$-(VEC_SIZE), %rdi
L(cross_page_continue):
	/* Check the first 4 * VEC_SIZE.  Only one VEC_SIZE at a time
	   since data is only aligned to VEC_SIZE.  */
# ifdef USE_AS_STRNLEN
	/* + CHAR_SIZE because it simplies the logic in
	   last_4x_vec_or_less.  */
	leaq	(VEC_SIZE * 5 + CHAR_SIZE)(%rdi), %rcx
	subq	%rdx, %rcx
#  ifdef USE_AS_WCSLEN
	/* NB: Divide bytes by 4 to get the wchar_t count.  */
	sarl	$2, %ecx
#  endif
# endif
	/* Load first VEC regardless.  */
	VPCMP	$0, VEC_SIZE(%rdi), %YMMZERO, %k0
# ifdef USE_AS_STRNLEN
	/* Adjust length. If near end handle specially.  */
	subq	%rcx, %rsi
	jb	L(last_4x_vec_or_less)
# endif
	kmovd	%k0, %eax
	testl	%eax, %eax
	jnz	L(first_vec_x1)

	VPCMP	$0, (VEC_SIZE * 2)(%rdi), %YMMZERO, %k0
	kmovd	%k0, %eax
	test	%eax, %eax
	jnz	L(first_vec_x2)

	VPCMP	$0, (VEC_SIZE * 3)(%rdi), %YMMZERO, %k0
	kmovd	%k0, %eax
	testl	%eax, %eax
	jnz	L(first_vec_x3)

	VPCMP	$0, (VEC_SIZE * 4)(%rdi), %YMMZERO, %k0
	kmovd	%k0, %eax
	testl	%eax, %eax
	jnz	L(first_vec_x4)

	addq	$VEC_SIZE, %rdi
# ifdef USE_AS_STRNLEN
	/* Check if at last VEC_SIZE * 4 length.  */
	cmpq	$(CHAR_PER_VEC * 4 - 1), %rsi
	jbe	L(last_4x_vec_or_less_load)
	movl	%edi, %ecx
	andl	$(VEC_SIZE * 4 - 1), %ecx
#  ifdef USE_AS_WCSLEN
	/* NB: Divide bytes by 4 to get the wchar_t count.  */
	sarl	$2, %ecx
#  endif
	/* Readjust length.  */
	addq	%rcx, %rsi
# endif
	/* Align data to VEC_SIZE * 4.  */
	andq	$-(VEC_SIZE * 4), %rdi

	/* Compare 4 * VEC at a time forward.  */
	.p2align 4
L(loop_4x_vec):
	/* Load first VEC regardless.  */
	VMOVA	(VEC_SIZE * 4)(%rdi), %YMM1
# ifdef USE_AS_STRNLEN
	/* Break if at end of length.  */
	subq	$(CHAR_PER_VEC * 4), %rsi
	jb	L(last_4x_vec_or_less_cmpeq)
# endif
	/* Save some code size by microfusing VPMINU with the load. Since
	   the matches in ymm2/ymm4 can only be returned if there where no
	   matches in ymm1/ymm3 respectively there is no issue with overlap.
	 */
	VPMINU	(VEC_SIZE * 5)(%rdi), %YMM1, %YMM2
	VMOVA	(VEC_SIZE * 6)(%rdi), %YMM3
	VPMINU	(VEC_SIZE * 7)(%rdi), %YMM3, %YMM4

	VPCMP	$0, %YMM2, %YMMZERO, %k0
	VPCMP	$0, %YMM4, %YMMZERO, %k1
	subq	$-(VEC_SIZE * 4), %rdi
	kortestd	%k0, %k1
	jz	L(loop_4x_vec)

	/* Check if end was in first half.  */
	kmovd	%k0, %eax
	subq	%rdx, %rdi
# ifdef USE_AS_WCSLEN
	shrq	$2, %rdi
# endif
	testl	%eax, %eax
	jz	L(second_vec_return)

	VPCMP	$0, %YMM1, %YMMZERO, %k2
	kmovd	%k2, %edx
	/* Combine VEC1 matches (edx) with VEC2 matches (eax).  */
# ifdef USE_AS_WCSLEN
	sall	$CHAR_PER_VEC, %eax
	orl	%edx, %eax
	tzcntl	%eax, %eax
# else
	salq	$CHAR_PER_VEC, %rax
	orq	%rdx, %rax
	tzcntq	%rax, %rax
# endif
	addq	%rdi, %rax
	ret


# ifdef USE_AS_STRNLEN

L(last_4x_vec_or_less_load):
	/* Depending on entry adjust rdi / prepare first VEC in YMM1.  */
	VMOVA	(VEC_SIZE * 4)(%rdi), %YMM1
L(last_4x_vec_or_less_cmpeq):
	VPCMP	$0, %YMM1, %YMMZERO, %k0
	addq	$(VEC_SIZE * 3), %rdi
L(last_4x_vec_or_less):
	kmovd	%k0, %eax
	/* If remaining length > VEC_SIZE * 2. This works if esi is off by
	   VEC_SIZE * 4.  */
	testl	$(CHAR_PER_VEC * 2), %esi
	jnz	L(last_4x_vec)

	/* length may have been negative or positive by an offset of
	   CHAR_PER_VEC * 4 depending on where this was called from. This
	   fixes that.  */
	andl	$(CHAR_PER_VEC * 4 - 1), %esi
	testl	%eax, %eax
	jnz	L(last_vec_x1_check)

	/* Check the end of data.  */
	subl	$CHAR_PER_VEC, %esi
	jb	L(max)

	VPCMP	$0, (VEC_SIZE * 2)(%rdi), %YMMZERO, %k0
	kmovd	%k0, %eax
	tzcntl	%eax, %eax
	/* Check the end of data.  */
	cmpl	%eax, %esi
	jb	L(max)

	subq	%rdx, %rdi
#  ifdef USE_AS_WCSLEN
	/* NB: Divide bytes by 4 to get the wchar_t count.  */
	sarq	$2, %rdi
#  endif
	leaq	(CHAR_PER_VEC * 2)(%rdi, %rax), %rax
	ret
L(max):
	movq	%r8, %rax
	ret
# endif

	/* Placed here in strnlen so that the jcc L(last_4x_vec_or_less)
	   in the 4x VEC loop can use 2 byte encoding.  */
	.p2align 4
L(second_vec_return):
	VPCMP	$0, %YMM3, %YMMZERO, %k0
	/* Combine YMM3 matches (k0) with YMM4 matches (k1).  */
# ifdef USE_AS_WCSLEN
	kunpckbw	%k0, %k1, %k0
	kmovd	%k0, %eax
	tzcntl	%eax, %eax
# else
	kunpckdq	%k0, %k1, %k0
	kmovq	%k0, %rax
	tzcntq	%rax, %rax
# endif
	leaq	(CHAR_PER_VEC * 2)(%rdi, %rax), %rax
	ret


# ifdef USE_AS_STRNLEN
L(last_vec_x1_check):
	tzcntl	%eax, %eax
	/* Check the end of data.  */
	cmpl	%eax, %esi
	jb	L(max)
	subq	%rdx, %rdi
#  ifdef USE_AS_WCSLEN
	/* NB: Divide bytes by 4 to get the wchar_t count.  */
	sarq	$2, %rdi
#  endif
	leaq	(CHAR_PER_VEC)(%rdi, %rax), %rax
	ret

	.p2align 4
L(last_4x_vec):
	/* Test first 2x VEC normally.  */
	testl	%eax, %eax
	jnz	L(last_vec_x1)

	VPCMP	$0, (VEC_SIZE * 2)(%rdi), %YMMZERO, %k0
	kmovd	%k0, %eax
	testl	%eax, %eax
	jnz	L(last_vec_x2)

	/* Normalize length.  */
	andl	$(CHAR_PER_VEC * 4 - 1), %esi
	VPCMP	$0, (VEC_SIZE * 3)(%rdi), %YMMZERO, %k0
	kmovd	%k0, %eax
	testl	%eax, %eax
	jnz	L(last_vec_x3)

	/* Check the end of data.  */
	subl	$(CHAR_PER_VEC * 3), %esi
	jb	L(max)

	VPCMP	$0, (VEC_SIZE * 4)(%rdi), %YMMZERO, %k0
	kmovd	%k0, %eax
	tzcntl	%eax, %eax
	/* Check the end of data.  */
	cmpl	%eax, %esi
	jb	L(max_end)

	subq	%rdx, %rdi
#  ifdef USE_AS_WCSLEN
	/* NB: Divide bytes by 4 to get the wchar_t count.  */
	sarq	$2, %rdi
#  endif
	leaq	(CHAR_PER_VEC * 4)(%rdi, %rax), %rax
	ret

	.p2align 4
L(last_vec_x1):
	tzcntl	%eax, %eax
	subq	%rdx, %rdi
#  ifdef USE_AS_WCSLEN
	/* NB: Divide bytes by 4 to get the wchar_t count.  */
	sarq	$2, %rdi
#  endif
	leaq	(CHAR_PER_VEC)(%rdi, %rax), %rax
	ret

	.p2align 4
L(last_vec_x2):
	tzcntl	%eax, %eax
	subq	%rdx, %rdi
#  ifdef USE_AS_WCSLEN
	/* NB: Divide bytes by 4 to get the wchar_t count.  */
	sarq	$2, %rdi
#  endif
	leaq	(CHAR_PER_VEC * 2)(%rdi, %rax), %rax
	ret

	.p2align 4
L(last_vec_x3):
	tzcntl	%eax, %eax
	subl	$(CHAR_PER_VEC * 2), %esi
	/* Check the end of data.  */
	cmpl	%eax, %esi
	jb	L(max_end)
	subq	%rdx, %rdi
#  ifdef USE_AS_WCSLEN
	/* NB: Divide bytes by 4 to get the wchar_t count.  */
	sarq	$2, %rdi
#  endif
	leaq	(CHAR_PER_VEC * 3)(%rdi, %rax), %rax
	ret
L(max_end):
	movq	%r8, %rax
	ret
# endif

	/* Cold case for crossing page with first load.	 */
	.p2align 4
L(cross_page_boundary):
	movq	%rdi, %rdx
	/* Align data to VEC_SIZE.  */
	andq	$-VEC_SIZE, %rdi
	VPCMP	$0, (%rdi), %YMMZERO, %k0
	kmovd	%k0, %eax
	/* Remove the leading bytes.  */
# ifdef USE_AS_WCSLEN
	/* NB: Divide shift count by 4 since each bit in K0 represent 4
	   bytes.  */
	movl	%edx, %ecx
	shrl	$2, %ecx
	andl	$(CHAR_PER_VEC - 1), %ecx
# endif
	/* SHIFT_REG is ecx for USE_AS_WCSLEN and edx otherwise.  */
	sarxl	%SHIFT_REG, %eax, %eax
	testl	%eax, %eax
# ifndef USE_AS_STRNLEN
	jz	L(cross_page_continue)
	tzcntl	%eax, %eax
	ret
# else
	jnz	L(cross_page_less_vec)
#  ifndef USE_AS_WCSLEN
	movl	%edx, %ecx
	andl	$(CHAR_PER_VEC - 1), %ecx
#  endif
	movl	$CHAR_PER_VEC, %eax
	subl	%ecx, %eax
	/* Check the end of data.  */
	cmpq	%rax, %rsi
	ja	L(cross_page_continue)
	movl	%esi, %eax
	ret
L(cross_page_less_vec):
	tzcntl	%eax, %eax
	/* Select min of length and position of first null.  */
	cmpq	%rax, %rsi
	cmovb	%esi, %eax
	ret
# endif

END (STRLEN)
#endif
