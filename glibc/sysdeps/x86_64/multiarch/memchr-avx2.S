/* memchr/wmemchr optimized with AVX2.
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

# ifndef MEMCHR
#  define MEMCHR	__memchr_avx2
# endif

# ifdef USE_AS_WMEMCHR
#  define VPCMPEQ	vpcmpeqd
#  define VPBROADCAST	vpbroadcastd
#  define CHAR_SIZE	4
# else
#  define VPCMPEQ	vpcmpeqb
#  define VPBROADCAST	vpbroadcastb
#  define CHAR_SIZE	1
# endif

# ifdef USE_AS_RAWMEMCHR
#  define ERAW_PTR_REG	ecx
#  define RRAW_PTR_REG	rcx
#  define ALGN_PTR_REG	rdi
# else
#  define ERAW_PTR_REG	edi
#  define RRAW_PTR_REG	rdi
#  define ALGN_PTR_REG	rcx
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
ENTRY (MEMCHR)
# ifndef USE_AS_RAWMEMCHR
	/* Check for zero length.  */
#  ifdef __ILP32__
	/* Clear upper bits.  */
	and	%RDX_LP, %RDX_LP
#  else
	test	%RDX_LP, %RDX_LP
#  endif
	jz	L(null)
# endif
	/* Broadcast CHAR to YMMMATCH.  */
	vmovd	%esi, %xmm0
	VPBROADCAST %xmm0, %ymm0
	/* Check if we may cross page boundary with one vector load.  */
	movl	%edi, %eax
	andl	$(PAGE_SIZE - 1), %eax
	cmpl	$(PAGE_SIZE - VEC_SIZE), %eax
	ja	L(cross_page_boundary)

	/* Check the first VEC_SIZE bytes.  */
	VPCMPEQ	(%rdi), %ymm0, %ymm1
	vpmovmskb %ymm1, %eax
# ifndef USE_AS_RAWMEMCHR
	/* If length < CHAR_PER_VEC handle special.  */
	cmpq	$CHAR_PER_VEC, %rdx
	jbe	L(first_vec_x0)
# endif
	testl	%eax, %eax
	jz	L(aligned_more)
	tzcntl	%eax, %eax
	addq	%rdi, %rax
	VZEROUPPER_RETURN

# ifndef USE_AS_RAWMEMCHR
	.p2align 5
L(first_vec_x0):
	/* Check if first match was before length.  */
	tzcntl	%eax, %eax
#  ifdef USE_AS_WMEMCHR
	/* NB: Multiply length by 4 to get byte count.  */
	sall	$2, %edx
#  endif
	xorl	%ecx, %ecx
	cmpl	%eax, %edx
	leaq	(%rdi, %rax), %rax
	cmovle	%rcx, %rax
	VZEROUPPER_RETURN

L(null):
	xorl	%eax, %eax
	ret
# endif
	.p2align 4
L(cross_page_boundary):
	/* Save pointer before aligning as its original value is
	   necessary for computer return address if byte is found or
	   adjusting length if it is not and this is memchr.  */
	movq	%rdi, %rcx
	/* Align data to VEC_SIZE - 1. ALGN_PTR_REG is rcx for memchr
	   and rdi for rawmemchr.  */
	orq	$(VEC_SIZE - 1), %ALGN_PTR_REG
	VPCMPEQ	-(VEC_SIZE - 1)(%ALGN_PTR_REG), %ymm0, %ymm1
	vpmovmskb %ymm1, %eax
# ifndef USE_AS_RAWMEMCHR
	/* Calculate length until end of page (length checked for a
	   match).  */
	leaq	1(%ALGN_PTR_REG), %rsi
	subq	%RRAW_PTR_REG, %rsi
#  ifdef USE_AS_WMEMCHR
	/* NB: Divide bytes by 4 to get wchar_t count.  */
	shrl	$2, %esi
#  endif
# endif
	/* Remove the leading bytes.  */
	sarxl	%ERAW_PTR_REG, %eax, %eax
# ifndef USE_AS_RAWMEMCHR
	/* Check the end of data.  */
	cmpq	%rsi, %rdx
	jbe	L(first_vec_x0)
# endif
	testl	%eax, %eax
	jz	L(cross_page_continue)
	tzcntl	%eax, %eax
	addq	%RRAW_PTR_REG, %rax
L(return_vzeroupper):
	ZERO_UPPER_VEC_REGISTERS_RETURN

	.p2align 4
L(first_vec_x1):
	tzcntl	%eax, %eax
	incq	%rdi
	addq	%rdi, %rax
	VZEROUPPER_RETURN

	.p2align 4
L(first_vec_x2):
	tzcntl	%eax, %eax
	addq	$(VEC_SIZE + 1), %rdi
	addq	%rdi, %rax
	VZEROUPPER_RETURN

	.p2align 4
L(first_vec_x3):
	tzcntl	%eax, %eax
	addq	$(VEC_SIZE * 2 + 1), %rdi
	addq	%rdi, %rax
	VZEROUPPER_RETURN


	.p2align 4
L(first_vec_x4):
	tzcntl	%eax, %eax
	addq	$(VEC_SIZE * 3 + 1), %rdi
	addq	%rdi, %rax
	VZEROUPPER_RETURN

	.p2align 4
L(aligned_more):
	/* Check the first 4 * VEC_SIZE.  Only one VEC_SIZE at a time
	   since data is only aligned to VEC_SIZE.  */

# ifndef USE_AS_RAWMEMCHR
L(cross_page_continue):
	/* Align data to VEC_SIZE - 1.  */
	xorl	%ecx, %ecx
	subl	%edi, %ecx
	orq	$(VEC_SIZE - 1), %rdi
	/* esi is for adjusting length to see if near the end.  */
	leal	(VEC_SIZE * 4 + 1)(%rdi, %rcx), %esi
#  ifdef USE_AS_WMEMCHR
	/* NB: Divide bytes by 4 to get the wchar_t count.  */
	sarl	$2, %esi
#  endif
# else
	orq	$(VEC_SIZE - 1), %rdi
L(cross_page_continue):
# endif
	/* Load first VEC regardless.  */
	VPCMPEQ	1(%rdi), %ymm0, %ymm1
	vpmovmskb %ymm1, %eax
# ifndef USE_AS_RAWMEMCHR
	/* Adjust length. If near end handle specially.  */
	subq	%rsi, %rdx
	jbe	L(last_4x_vec_or_less)
# endif
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

# ifndef USE_AS_RAWMEMCHR
	/* Check if at last VEC_SIZE * 4 length.  */
	subq	$(CHAR_PER_VEC * 4), %rdx
	jbe	L(last_4x_vec_or_less_cmpeq)
	/* Align data to VEC_SIZE * 4 - 1 for the loop and readjust
	   length.  */
	incq	%rdi
	movl	%edi, %ecx
	orq	$(VEC_SIZE * 4 - 1), %rdi
	andl	$(VEC_SIZE * 4 - 1), %ecx
#  ifdef USE_AS_WMEMCHR
	/* NB: Divide bytes by 4 to get the wchar_t count.  */
	sarl	$2, %ecx
#  endif
	addq	%rcx, %rdx
# else
	/* Align data to VEC_SIZE * 4 - 1 for loop.  */
	incq	%rdi
	orq	$(VEC_SIZE * 4 - 1), %rdi
# endif

	/* Compare 4 * VEC at a time forward.  */
	.p2align 4
L(loop_4x_vec):
	VPCMPEQ	1(%rdi), %ymm0, %ymm1
	VPCMPEQ	(VEC_SIZE + 1)(%rdi), %ymm0, %ymm2
	VPCMPEQ	(VEC_SIZE * 2 + 1)(%rdi), %ymm0, %ymm3
	VPCMPEQ	(VEC_SIZE * 3 + 1)(%rdi), %ymm0, %ymm4
	vpor	%ymm1, %ymm2, %ymm5
	vpor	%ymm3, %ymm4, %ymm6
	vpor	%ymm5, %ymm6, %ymm5

	vpmovmskb %ymm5, %ecx
# ifdef USE_AS_RAWMEMCHR
	subq	$-(VEC_SIZE * 4), %rdi
	testl	%ecx, %ecx
	jz	L(loop_4x_vec)
# else
	testl	%ecx, %ecx
	jnz	L(loop_4x_vec_end)

	subq	$-(VEC_SIZE * 4), %rdi

	subq	$(CHAR_PER_VEC * 4), %rdx
	ja	L(loop_4x_vec)

	/* Fall through into less than 4 remaining vectors of length
	   case.  */
	VPCMPEQ	(VEC_SIZE * 0 + 1)(%rdi), %ymm0, %ymm1
	vpmovmskb %ymm1, %eax
	.p2align 4
L(last_4x_vec_or_less):
#  ifdef USE_AS_WMEMCHR
	/* NB: Multiply length by 4 to get byte count.  */
	sall	$2, %edx
#  endif
	/* Check if first VEC contained match.  */
	testl	%eax, %eax
	jnz	L(first_vec_x1_check)

	/* If remaining length > VEC_SIZE * 2.  */
	addl	$(VEC_SIZE * 2), %edx
	jg	L(last_4x_vec)

L(last_2x_vec):
	/* If remaining length < VEC_SIZE.  */
	addl	$VEC_SIZE, %edx
	jle	L(zero_end)

	/* Check VEC2 and compare any match with remaining length.  */
	VPCMPEQ	(VEC_SIZE + 1)(%rdi), %ymm0, %ymm1
	vpmovmskb %ymm1, %eax
	tzcntl	%eax, %eax
	cmpl	%eax, %edx
	jbe	L(set_zero_end)
	addq	$(VEC_SIZE + 1), %rdi
	addq	%rdi, %rax
L(zero_end):
	VZEROUPPER_RETURN

	.p2align 4
L(loop_4x_vec_end):
# endif
	/* rawmemchr will fall through into this if match was found in
	   loop.  */

	vpmovmskb %ymm1, %eax
	testl	%eax, %eax
	jnz	L(last_vec_x1_return)

	vpmovmskb %ymm2, %eax
	testl	%eax, %eax
	jnz	L(last_vec_x2_return)

	vpmovmskb %ymm3, %eax
	/* Combine VEC3 matches (eax) with VEC4 matches (ecx).  */
	salq	$32, %rcx
	orq	%rcx, %rax
	tzcntq	%rax, %rax
# ifdef USE_AS_RAWMEMCHR
	subq	$(VEC_SIZE * 2 - 1), %rdi
# else
	subq	$-(VEC_SIZE * 2 + 1), %rdi
# endif
	addq	%rdi, %rax
	VZEROUPPER_RETURN
# ifndef USE_AS_RAWMEMCHR

	.p2align 4
L(first_vec_x1_check):
	tzcntl	%eax, %eax
	/* Adjust length.  */
	subl	$-(VEC_SIZE * 4), %edx
	/* Check if match within remaining length.  */
	cmpl	%eax, %edx
	jbe	L(set_zero_end)
	incq	%rdi
	addq	%rdi, %rax
	VZEROUPPER_RETURN
	.p2align 4
L(set_zero_end):
	xorl	%eax, %eax
	VZEROUPPER_RETURN
# endif

	.p2align 4
L(last_vec_x1_return):
	tzcntl	%eax, %eax
# ifdef USE_AS_RAWMEMCHR
	subq	$(VEC_SIZE * 4 - 1), %rdi
# else
	incq	%rdi
# endif
	addq	%rdi, %rax
	VZEROUPPER_RETURN

	.p2align 4
L(last_vec_x2_return):
	tzcntl	%eax, %eax
# ifdef USE_AS_RAWMEMCHR
	subq	$(VEC_SIZE * 3 - 1), %rdi
# else
	subq	$-(VEC_SIZE + 1), %rdi
# endif
	addq	%rdi, %rax
	VZEROUPPER_RETURN

# ifndef USE_AS_RAWMEMCHR
	.p2align 4
L(last_4x_vec_or_less_cmpeq):
	VPCMPEQ	(VEC_SIZE * 4 + 1)(%rdi), %ymm0, %ymm1
	vpmovmskb %ymm1, %eax
#  ifdef USE_AS_WMEMCHR
	/* NB: Multiply length by 4 to get byte count.  */
	sall	$2, %edx
#  endif
	subq	$-(VEC_SIZE * 4), %rdi
	/* Check first VEC regardless.  */
	testl	%eax, %eax
	jnz	L(first_vec_x1_check)

	/* If remaining length <= CHAR_PER_VEC * 2.  */
	addl	$(VEC_SIZE * 2), %edx
	jle	L(last_2x_vec)
	.p2align 4
L(last_4x_vec):
	VPCMPEQ	(VEC_SIZE + 1)(%rdi), %ymm0, %ymm1
	vpmovmskb %ymm1, %eax
	testl	%eax, %eax
	jnz	L(last_vec_x2_return)

	VPCMPEQ	(VEC_SIZE * 2 + 1)(%rdi), %ymm0, %ymm1
	vpmovmskb %ymm1, %eax

	/* Create mask for possible matches within remaining length.  */
	movq	$-1, %rcx
	bzhiq	%rdx, %rcx, %rcx

	/* Test matches in data against length match.  */
	andl	%ecx, %eax
	jnz	L(last_vec_x3)

	/* if remaining length <= VEC_SIZE * 3 (Note this is after
	   remaining length was found to be > VEC_SIZE * 2.  */
	subl	$VEC_SIZE, %edx
	jbe	L(zero_end2)

	VPCMPEQ	(VEC_SIZE * 3 + 1)(%rdi), %ymm0, %ymm1
	vpmovmskb %ymm1, %eax
	/* Shift remaining length mask for last VEC.  */
	shrq	$32, %rcx
	andl	%ecx, %eax
	jz	L(zero_end2)
	tzcntl	%eax, %eax
	addq	$(VEC_SIZE * 3 + 1), %rdi
	addq	%rdi, %rax
L(zero_end2):
	VZEROUPPER_RETURN

	.p2align 4
L(last_vec_x3):
	tzcntl	%eax, %eax
	subq	$-(VEC_SIZE * 2 + 1), %rdi
	addq	%rdi, %rax
	VZEROUPPER_RETURN
# endif

END (MEMCHR)
#endif
