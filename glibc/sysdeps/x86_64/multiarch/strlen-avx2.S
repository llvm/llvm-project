/* strlen/strnlen/wcslen/wcsnlen optimized with AVX2.
   Copyright (C) 2017-2021 Free Software Foundation, Inc.
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
#  define STRLEN	__strlen_avx2
# endif

# ifdef USE_AS_WCSLEN
#  define VPCMPEQ	vpcmpeqd
#  define VPMINU	vpminud
#  define CHAR_SIZE	4
# else
#  define VPCMPEQ	vpcmpeqb
#  define VPMINU	vpminub
#  define CHAR_SIZE	1
# endif

# ifndef VZEROUPPER
#  define VZEROUPPER	vzeroupper
# endif

# ifndef SECTION
#  define SECTION(p)	p##.avx
# endif

# define VEC_SIZE 32
# define PAGE_SIZE 4096
# define CHAR_PER_VEC	(VEC_SIZE / CHAR_SIZE)

	.section SECTION(.text),"ax",@progbits
ENTRY (STRLEN)
# ifdef USE_AS_STRNLEN
	/* Check zero length.  */
#  ifdef __ILP32__
	/* Clear upper bits.  */
	and	%RSI_LP, %RSI_LP
#  else
	test	%RSI_LP, %RSI_LP
#  endif
	jz	L(zero)
	/* Store max len in R8_LP before adjusting if using WCSLEN.  */
	mov	%RSI_LP, %R8_LP
# endif
	movl	%edi, %eax
	movq	%rdi, %rdx
	vpxor	%xmm0, %xmm0, %xmm0
	/* Clear high bits from edi. Only keeping bits relevant to page
	   cross check.  */
	andl	$(PAGE_SIZE - 1), %eax
	/* Check if we may cross page boundary with one vector load.  */
	cmpl	$(PAGE_SIZE - VEC_SIZE), %eax
	ja	L(cross_page_boundary)

	/* Check the first VEC_SIZE bytes.  */
	VPCMPEQ	(%rdi), %ymm0, %ymm1
	vpmovmskb %ymm1, %eax
# ifdef USE_AS_STRNLEN
	/* If length < VEC_SIZE handle special.  */
	cmpq	$CHAR_PER_VEC, %rsi
	jbe	L(first_vec_x0)
# endif
	/* If empty continue to aligned_more. Otherwise return bit
	   position of first match.  */
	testl	%eax, %eax
	jz	L(aligned_more)
	tzcntl	%eax, %eax
# ifdef USE_AS_WCSLEN
	/* NB: Divide bytes by 4 to get wchar_t count.  */
	shrl	$2, %eax
# endif
	VZEROUPPER_RETURN

# ifdef USE_AS_STRNLEN
L(zero):
	xorl	%eax, %eax
	ret

	.p2align 4
L(first_vec_x0):
	/* Set bit for max len so that tzcnt will return min of max len
	   and position of first match.  */
#  ifdef USE_AS_WCSLEN
	/* NB: Multiply length by 4 to get byte count.  */
	sall	$2, %esi
#  endif
	btsq	%rsi, %rax
	tzcntl	%eax, %eax
#  ifdef USE_AS_WCSLEN
	/* NB: Divide bytes by 4 to get wchar_t count.  */
	shrl	$2, %eax
#  endif
	VZEROUPPER_RETURN
# endif

	.p2align 4
L(first_vec_x1):
	tzcntl	%eax, %eax
	/* Safe to use 32 bit instructions as these are only called for
	   size = [1, 159].  */
# ifdef USE_AS_STRNLEN
	/* Use ecx which was computed earlier to compute correct value.
	 */
#  ifdef USE_AS_WCSLEN
	leal	-(VEC_SIZE * 4 + 1)(%rax, %rcx, 4), %eax
#  else
	subl	$(VEC_SIZE * 4 + 1), %ecx
	addl	%ecx, %eax
#  endif
# else
	subl	%edx, %edi
	incl	%edi
	addl	%edi, %eax
# endif
# ifdef USE_AS_WCSLEN
	/* NB: Divide bytes by 4 to get wchar_t count.  */
	shrl	$2, %eax
# endif
	VZEROUPPER_RETURN

	.p2align 4
L(first_vec_x2):
	tzcntl	%eax, %eax
	/* Safe to use 32 bit instructions as these are only called for
	   size = [1, 159].  */
# ifdef USE_AS_STRNLEN
	/* Use ecx which was computed earlier to compute correct value.
	 */
#  ifdef USE_AS_WCSLEN
	leal	-(VEC_SIZE * 3 + 1)(%rax, %rcx, 4), %eax
#  else
	subl	$(VEC_SIZE * 3 + 1), %ecx
	addl	%ecx, %eax
#  endif
# else
	subl	%edx, %edi
	addl	$(VEC_SIZE + 1), %edi
	addl	%edi, %eax
# endif
# ifdef USE_AS_WCSLEN
	/* NB: Divide bytes by 4 to get wchar_t count.  */
	shrl	$2, %eax
# endif
	VZEROUPPER_RETURN

	.p2align 4
L(first_vec_x3):
	tzcntl	%eax, %eax
	/* Safe to use 32 bit instructions as these are only called for
	   size = [1, 159].  */
# ifdef USE_AS_STRNLEN
	/* Use ecx which was computed earlier to compute correct value.
	 */
#  ifdef USE_AS_WCSLEN
	leal	-(VEC_SIZE * 2 + 1)(%rax, %rcx, 4), %eax
#  else
	subl	$(VEC_SIZE * 2 + 1), %ecx
	addl	%ecx, %eax
#  endif
# else
	subl	%edx, %edi
	addl	$(VEC_SIZE * 2 + 1), %edi
	addl	%edi, %eax
# endif
# ifdef USE_AS_WCSLEN
	/* NB: Divide bytes by 4 to get wchar_t count.  */
	shrl	$2, %eax
# endif
	VZEROUPPER_RETURN

	.p2align 4
L(first_vec_x4):
	tzcntl	%eax, %eax
	/* Safe to use 32 bit instructions as these are only called for
	   size = [1, 159].  */
# ifdef USE_AS_STRNLEN
	/* Use ecx which was computed earlier to compute correct value.
	 */
#  ifdef USE_AS_WCSLEN
	leal	-(VEC_SIZE * 1 + 1)(%rax, %rcx, 4), %eax
#  else
	subl	$(VEC_SIZE + 1), %ecx
	addl	%ecx, %eax
#  endif
# else
	subl	%edx, %edi
	addl	$(VEC_SIZE * 3 + 1), %edi
	addl	%edi, %eax
# endif
# ifdef USE_AS_WCSLEN
	/* NB: Divide bytes by 4 to get wchar_t count.  */
	shrl	$2, %eax
# endif
	VZEROUPPER_RETURN

	.p2align 5
L(aligned_more):
	/* Align data to VEC_SIZE - 1. This is the same number of
	   instructions as using andq with -VEC_SIZE but saves 4 bytes of
	   code on the x4 check.  */
	orq	$(VEC_SIZE - 1), %rdi
L(cross_page_continue):
	/* Check the first 4 * VEC_SIZE.  Only one VEC_SIZE at a time
	   since data is only aligned to VEC_SIZE.  */
# ifdef USE_AS_STRNLEN
	/* + 1 because rdi is aligned to VEC_SIZE - 1. + CHAR_SIZE
	   because it simplies the logic in last_4x_vec_or_less.  */
	leaq	(VEC_SIZE * 4 + CHAR_SIZE + 1)(%rdi), %rcx
	subq	%rdx, %rcx
#  ifdef USE_AS_WCSLEN
	/* NB: Divide bytes by 4 to get the wchar_t count.  */
	sarl	$2, %ecx
#  endif
# endif
	/* Load first VEC regardless.  */
	VPCMPEQ	1(%rdi), %ymm0, %ymm1
# ifdef USE_AS_STRNLEN
	/* Adjust length. If near end handle specially.  */
	subq	%rcx, %rsi
	jb	L(last_4x_vec_or_less)
# endif
	vpmovmskb %ymm1, %eax
	testl	%eax, %eax
	jnz	L(first_vec_x1)

	VPCMPEQ	(VEC_SIZE + 1)(%rdi), %ymm0, %ymm1
	vpmovmskb %ymm1, %eax
	testl	%eax, %eax
	jnz	L(first_vec_x2)

	VPCMPEQ	(VEC_SIZE * 2 + 1)(%rdi), %ymm0, %ymm1
	vpmovmskb %ymm1, %eax
	testl	%eax, %eax
	jnz	L(first_vec_x3)

	VPCMPEQ	(VEC_SIZE * 3 + 1)(%rdi), %ymm0, %ymm1
	vpmovmskb %ymm1, %eax
	testl	%eax, %eax
	jnz	L(first_vec_x4)

	/* Align data to VEC_SIZE * 4 - 1.  */
# ifdef USE_AS_STRNLEN
	/* Before adjusting length check if at last VEC_SIZE * 4.  */
	cmpq	$(CHAR_PER_VEC * 4 - 1), %rsi
	jbe	L(last_4x_vec_or_less_load)
	incq	%rdi
	movl	%edi, %ecx
	orq	$(VEC_SIZE * 4 - 1), %rdi
	andl	$(VEC_SIZE * 4 - 1), %ecx
#  ifdef USE_AS_WCSLEN
	/* NB: Divide bytes by 4 to get the wchar_t count.  */
	sarl	$2, %ecx
#  endif
	/* Readjust length.  */
	addq	%rcx, %rsi
# else
	incq	%rdi
	orq	$(VEC_SIZE * 4 - 1), %rdi
# endif
	/* Compare 4 * VEC at a time forward.  */
	.p2align 4
L(loop_4x_vec):
# ifdef USE_AS_STRNLEN
	/* Break if at end of length.  */
	subq	$(CHAR_PER_VEC * 4), %rsi
	jb	L(last_4x_vec_or_less_cmpeq)
# endif
	/* Save some code size by microfusing VPMINU with the load.
	   Since the matches in ymm2/ymm4 can only be returned if there
	   where no matches in ymm1/ymm3 respectively there is no issue
	   with overlap.  */
	vmovdqa	1(%rdi), %ymm1
	VPMINU	(VEC_SIZE + 1)(%rdi), %ymm1, %ymm2
	vmovdqa	(VEC_SIZE * 2 + 1)(%rdi), %ymm3
	VPMINU	(VEC_SIZE * 3 + 1)(%rdi), %ymm3, %ymm4

	VPMINU	%ymm2, %ymm4, %ymm5
	VPCMPEQ	%ymm5, %ymm0, %ymm5
	vpmovmskb %ymm5, %ecx

	subq	$-(VEC_SIZE * 4), %rdi
	testl	%ecx, %ecx
	jz	L(loop_4x_vec)


	VPCMPEQ	%ymm1, %ymm0, %ymm1
	vpmovmskb %ymm1, %eax
	subq	%rdx, %rdi
	testl	%eax, %eax
	jnz	L(last_vec_return_x0)

	VPCMPEQ	%ymm2, %ymm0, %ymm2
	vpmovmskb %ymm2, %eax
	testl	%eax, %eax
	jnz	L(last_vec_return_x1)

	/* Combine last 2 VEC.  */
	VPCMPEQ	%ymm3, %ymm0, %ymm3
	vpmovmskb %ymm3, %eax
	/* rcx has combined result from all 4 VEC. It will only be used
	   if the first 3 other VEC all did not contain a match.  */
	salq	$32, %rcx
	orq	%rcx, %rax
	tzcntq	%rax, %rax
	subq	$(VEC_SIZE * 2 - 1), %rdi
	addq	%rdi, %rax
# ifdef USE_AS_WCSLEN
	/* NB: Divide bytes by 4 to get wchar_t count.  */
	shrq	$2, %rax
# endif
	VZEROUPPER_RETURN


# ifdef USE_AS_STRNLEN
	.p2align 4
L(last_4x_vec_or_less_load):
	/* Depending on entry adjust rdi / prepare first VEC in ymm1.
	 */
	subq	$-(VEC_SIZE * 4), %rdi
L(last_4x_vec_or_less_cmpeq):
	VPCMPEQ	1(%rdi), %ymm0, %ymm1
L(last_4x_vec_or_less):
#  ifdef USE_AS_WCSLEN
	/* NB: Multiply length by 4 to get byte count.  */
	sall	$2, %esi
#  endif
	vpmovmskb %ymm1, %eax
	/* If remaining length > VEC_SIZE * 2. This works if esi is off
	   by VEC_SIZE * 4.  */
	testl	$(VEC_SIZE * 2), %esi
	jnz	L(last_4x_vec)

	/* length may have been negative or positive by an offset of
	   VEC_SIZE * 4 depending on where this was called from. This fixes
	   that.  */
	andl	$(VEC_SIZE * 4 - 1), %esi
	testl	%eax, %eax
	jnz	L(last_vec_x1_check)

	subl	$VEC_SIZE, %esi
	jb	L(max)

	VPCMPEQ	(VEC_SIZE + 1)(%rdi), %ymm0, %ymm1
	vpmovmskb %ymm1, %eax
	tzcntl	%eax, %eax
	/* Check the end of data.  */
	cmpl	%eax, %esi
	jb	L(max)
	subq	%rdx, %rdi
	addl	$(VEC_SIZE + 1), %eax
	addq	%rdi, %rax
#  ifdef USE_AS_WCSLEN
	/* NB: Divide bytes by 4 to get wchar_t count.  */
	shrq	$2, %rax
#  endif
	VZEROUPPER_RETURN
# endif

	.p2align 4
L(last_vec_return_x0):
	tzcntl	%eax, %eax
	subq	$(VEC_SIZE * 4 - 1), %rdi
	addq	%rdi, %rax
# ifdef USE_AS_WCSLEN
	/* NB: Divide bytes by 4 to get wchar_t count.  */
	shrq	$2, %rax
# endif
	VZEROUPPER_RETURN

	.p2align 4
L(last_vec_return_x1):
	tzcntl	%eax, %eax
	subq	$(VEC_SIZE * 3 - 1), %rdi
	addq	%rdi, %rax
# ifdef USE_AS_WCSLEN
	/* NB: Divide bytes by 4 to get wchar_t count.  */
	shrq	$2, %rax
# endif
	VZEROUPPER_RETURN

# ifdef USE_AS_STRNLEN
	.p2align 4
L(last_vec_x1_check):

	tzcntl	%eax, %eax
	/* Check the end of data.  */
	cmpl	%eax, %esi
	jb	L(max)
	subq	%rdx, %rdi
	incl	%eax
	addq	%rdi, %rax
#  ifdef USE_AS_WCSLEN
	/* NB: Divide bytes by 4 to get wchar_t count.  */
	shrq	$2, %rax
#  endif
	VZEROUPPER_RETURN

L(max):
	movq	%r8, %rax
	VZEROUPPER_RETURN

	.p2align 4
L(last_4x_vec):
	/* Test first 2x VEC normally.  */
	testl	%eax, %eax
	jnz	L(last_vec_x1)

	VPCMPEQ	(VEC_SIZE + 1)(%rdi), %ymm0, %ymm1
	vpmovmskb %ymm1, %eax
	testl	%eax, %eax
	jnz	L(last_vec_x2)

	/* Normalize length.  */
	andl	$(VEC_SIZE * 4 - 1), %esi
	VPCMPEQ	(VEC_SIZE * 2 + 1)(%rdi), %ymm0, %ymm1
	vpmovmskb %ymm1, %eax
	testl	%eax, %eax
	jnz	L(last_vec_x3)

	subl	$(VEC_SIZE * 3), %esi
	jb	L(max)

	VPCMPEQ	(VEC_SIZE * 3 + 1)(%rdi), %ymm0, %ymm1
	vpmovmskb %ymm1, %eax
	tzcntl	%eax, %eax
	/* Check the end of data.  */
	cmpl	%eax, %esi
	jb	L(max)
	subq	%rdx, %rdi
	addl	$(VEC_SIZE * 3 + 1), %eax
	addq	%rdi, %rax
#  ifdef USE_AS_WCSLEN
	/* NB: Divide bytes by 4 to get wchar_t count.  */
	shrq	$2, %rax
#  endif
	VZEROUPPER_RETURN


	.p2align 4
L(last_vec_x1):
	/* essentially duplicates of first_vec_x1 but use 64 bit
	   instructions.  */
	tzcntl	%eax, %eax
	subq	%rdx, %rdi
	incl	%eax
	addq	%rdi, %rax
#  ifdef USE_AS_WCSLEN
	/* NB: Divide bytes by 4 to get wchar_t count.  */
	shrq	$2, %rax
#  endif
	VZEROUPPER_RETURN

	.p2align 4
L(last_vec_x2):
	/* essentially duplicates of first_vec_x1 but use 64 bit
	   instructions.  */
	tzcntl	%eax, %eax
	subq	%rdx, %rdi
	addl	$(VEC_SIZE + 1), %eax
	addq	%rdi, %rax
#  ifdef USE_AS_WCSLEN
	/* NB: Divide bytes by 4 to get wchar_t count.  */
	shrq	$2, %rax
#  endif
	VZEROUPPER_RETURN

	.p2align 4
L(last_vec_x3):
	tzcntl	%eax, %eax
	subl	$(VEC_SIZE * 2), %esi
	/* Check the end of data.  */
	cmpl	%eax, %esi
	jb	L(max_end)
	subq	%rdx, %rdi
	addl	$(VEC_SIZE * 2 + 1), %eax
	addq	%rdi, %rax
#  ifdef USE_AS_WCSLEN
	/* NB: Divide bytes by 4 to get wchar_t count.  */
	shrq	$2, %rax
#  endif
	VZEROUPPER_RETURN
L(max_end):
	movq	%r8, %rax
	VZEROUPPER_RETURN
# endif

	/* Cold case for crossing page with first load.  */
	.p2align 4
L(cross_page_boundary):
	/* Align data to VEC_SIZE - 1.  */
	orq	$(VEC_SIZE - 1), %rdi
	VPCMPEQ	-(VEC_SIZE - 1)(%rdi), %ymm0, %ymm1
	vpmovmskb %ymm1, %eax
	/* Remove the leading bytes. sarxl only uses bits [5:0] of COUNT
	   so no need to manually mod rdx.  */
	sarxl	%edx, %eax, %eax
# ifdef USE_AS_STRNLEN
	testl	%eax, %eax
	jnz	L(cross_page_less_vec)
	leaq	1(%rdi), %rcx
	subq	%rdx, %rcx
#  ifdef USE_AS_WCSLEN
	/* NB: Divide bytes by 4 to get wchar_t count.  */
	shrl	$2, %ecx
#  endif
	/* Check length.  */
	cmpq	%rsi, %rcx
	jb	L(cross_page_continue)
	movq	%r8, %rax
# else
	testl	%eax, %eax
	jz	L(cross_page_continue)
	tzcntl	%eax, %eax
#  ifdef USE_AS_WCSLEN
	/* NB: Divide length by 4 to get wchar_t count.  */
	shrl	$2, %eax
#  endif
# endif
L(return_vzeroupper):
	ZERO_UPPER_VEC_REGISTERS_RETURN

# ifdef USE_AS_STRNLEN
	.p2align 4
L(cross_page_less_vec):
	tzcntl	%eax, %eax
#  ifdef USE_AS_WCSLEN
	/* NB: Multiply length by 4 to get byte count.  */
	sall	$2, %esi
#  endif
	cmpq	%rax, %rsi
	cmovb	%esi, %eax
#  ifdef USE_AS_WCSLEN
	shrl	$2, %eax
#  endif
	VZEROUPPER_RETURN
# endif

END (STRLEN)
#endif
