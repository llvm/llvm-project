/* memchr/wmemchr optimized with 256-bit EVEX instructions.
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

# ifndef MEMCHR
#  define MEMCHR	__memchr_evex
# endif

# ifdef USE_AS_WMEMCHR
#  define VPBROADCAST	vpbroadcastd
#  define VPMINU	vpminud
#  define VPCMP	vpcmpd
#  define VPCMPEQ	vpcmpeqd
#  define CHAR_SIZE	4
# else
#  define VPBROADCAST	vpbroadcastb
#  define VPMINU	vpminub
#  define VPCMP	vpcmpb
#  define VPCMPEQ	vpcmpeqb
#  define CHAR_SIZE	1
# endif

	/* In the 4x loop the RTM and non-RTM versions have data pointer
	   off by VEC_SIZE * 4 with RTM version being VEC_SIZE * 4 greater.
	   This is represented by BASE_OFFSET. As well because the RTM
	   version uses vpcmp which stores a bit per element compared where
	   the non-RTM version uses vpcmpeq which stores a bit per byte
	   compared RET_SCALE of CHAR_SIZE is only relevant for the RTM
	   version.  */
# ifdef USE_IN_RTM
#  define VZEROUPPER
#  define BASE_OFFSET	(VEC_SIZE * 4)
#  define RET_SCALE	CHAR_SIZE
# else
#  define VZEROUPPER	vzeroupper
#  define BASE_OFFSET	0
#  define RET_SCALE	1
# endif

	/* In the return from 4x loop memchr and rawmemchr versions have
	   data pointers off by VEC_SIZE * 4 with memchr version being
	   VEC_SIZE * 4 greater.  */
# ifdef USE_AS_RAWMEMCHR
#  define RET_OFFSET	(BASE_OFFSET - (VEC_SIZE * 4))
#  define RAW_PTR_REG	rcx
#  define ALGN_PTR_REG	rdi
# else
#  define RET_OFFSET	BASE_OFFSET
#  define RAW_PTR_REG	rdi
#  define ALGN_PTR_REG	rcx
# endif

# define XMMZERO	xmm23
# define YMMZERO	ymm23
# define XMMMATCH	xmm16
# define YMMMATCH	ymm16
# define YMM1		ymm17
# define YMM2		ymm18
# define YMM3		ymm19
# define YMM4		ymm20
# define YMM5		ymm21
# define YMM6		ymm22

# ifndef SECTION
#  define SECTION(p)	p##.evex
# endif

# define VEC_SIZE 32
# define CHAR_PER_VEC (VEC_SIZE / CHAR_SIZE)
# define PAGE_SIZE 4096

	.section SECTION(.text),"ax",@progbits
ENTRY (MEMCHR)
# ifndef USE_AS_RAWMEMCHR
	/* Check for zero length.  */
	test	%RDX_LP, %RDX_LP
	jz	L(zero)

#  ifdef __ILP32__
	/* Clear the upper 32 bits.  */
	movl	%edx, %edx
#  endif
# endif
	/* Broadcast CHAR to YMMMATCH.  */
	VPBROADCAST %esi, %YMMMATCH
	/* Check if we may cross page boundary with one vector load.  */
	movl	%edi, %eax
	andl	$(PAGE_SIZE - 1), %eax
	cmpl	$(PAGE_SIZE - VEC_SIZE), %eax
	ja	L(cross_page_boundary)

	/* Check the first VEC_SIZE bytes.  */
	VPCMP	$0, (%rdi), %YMMMATCH, %k0
	kmovd	%k0, %eax
# ifndef USE_AS_RAWMEMCHR
	/* If length < CHAR_PER_VEC handle special.  */
	cmpq	$CHAR_PER_VEC, %rdx
	jbe	L(first_vec_x0)
# endif
	testl	%eax, %eax
	jz	L(aligned_more)
	tzcntl	%eax, %eax
# ifdef USE_AS_WMEMCHR
	/* NB: Multiply bytes by CHAR_SIZE to get the wchar_t count.  */
	leaq	(%rdi, %rax, CHAR_SIZE), %rax
# else
	addq	%rdi, %rax
# endif
	ret

# ifndef USE_AS_RAWMEMCHR
L(zero):
	xorl	%eax, %eax
	ret

	.p2align 5
L(first_vec_x0):
	/* Check if first match was before length.  */
	tzcntl	%eax, %eax
	xorl	%ecx, %ecx
	cmpl	%eax, %edx
	leaq	(%rdi, %rax, CHAR_SIZE), %rax
	cmovle	%rcx, %rax
	ret
# else
	/* NB: first_vec_x0 is 17 bytes which will leave
	   cross_page_boundary (which is relatively cold) close enough
	   to ideal alignment. So only realign L(cross_page_boundary) if
	   rawmemchr.  */
	.p2align 4
# endif
L(cross_page_boundary):
	/* Save pointer before aligning as its original value is
	   necessary for computer return address if byte is found or
	   adjusting length if it is not and this is memchr.  */
	movq	%rdi, %rcx
	/* Align data to VEC_SIZE. ALGN_PTR_REG is rcx for memchr and rdi
	   for rawmemchr.  */
	andq	$-VEC_SIZE, %ALGN_PTR_REG
	VPCMP	$0, (%ALGN_PTR_REG), %YMMMATCH, %k0
	kmovd	%k0, %r8d
# ifdef USE_AS_WMEMCHR
	/* NB: Divide shift count by 4 since each bit in K0 represent 4
	   bytes.  */
	sarl	$2, %eax
# endif
# ifndef USE_AS_RAWMEMCHR
	movl	$(PAGE_SIZE / CHAR_SIZE), %esi
	subl	%eax, %esi
# endif
# ifdef USE_AS_WMEMCHR
	andl	$(CHAR_PER_VEC - 1), %eax
# endif
	/* Remove the leading bytes.  */
	sarxl	%eax, %r8d, %eax
# ifndef USE_AS_RAWMEMCHR
	/* Check the end of data.  */
	cmpq	%rsi, %rdx
	jbe	L(first_vec_x0)
# endif
	testl	%eax, %eax
	jz	L(cross_page_continue)
	tzcntl	%eax, %eax
# ifdef USE_AS_WMEMCHR
	/* NB: Multiply bytes by CHAR_SIZE to get the wchar_t count.  */
	leaq	(%RAW_PTR_REG, %rax, CHAR_SIZE), %rax
# else
	addq	%RAW_PTR_REG, %rax
# endif
	ret

	.p2align 4
L(first_vec_x1):
	tzcntl	%eax, %eax
	leaq	VEC_SIZE(%rdi, %rax, CHAR_SIZE), %rax
	ret

	.p2align 4
L(first_vec_x2):
	tzcntl	%eax, %eax
	leaq	(VEC_SIZE * 2)(%rdi, %rax, CHAR_SIZE), %rax
	ret

	.p2align 4
L(first_vec_x3):
	tzcntl	%eax, %eax
	leaq	(VEC_SIZE * 3)(%rdi, %rax, CHAR_SIZE), %rax
	ret

	.p2align 4
L(first_vec_x4):
	tzcntl	%eax, %eax
	leaq	(VEC_SIZE * 4)(%rdi, %rax, CHAR_SIZE), %rax
	ret

	.p2align 5
L(aligned_more):
	/* Check the first 4 * VEC_SIZE.  Only one VEC_SIZE at a time
	   since data is only aligned to VEC_SIZE.  */

# ifndef USE_AS_RAWMEMCHR
	/* Align data to VEC_SIZE.  */
L(cross_page_continue):
	xorl	%ecx, %ecx
	subl	%edi, %ecx
	andq	$-VEC_SIZE, %rdi
	/* esi is for adjusting length to see if near the end.  */
	leal	(VEC_SIZE * 5)(%rdi, %rcx), %esi
#  ifdef USE_AS_WMEMCHR
	/* NB: Divide bytes by 4 to get the wchar_t count.  */
	sarl	$2, %esi
#  endif
# else
	andq	$-VEC_SIZE, %rdi
L(cross_page_continue):
# endif
	/* Load first VEC regardless.  */
	VPCMP	$0, (VEC_SIZE)(%rdi), %YMMMATCH, %k0
	kmovd	%k0, %eax
# ifndef USE_AS_RAWMEMCHR
	/* Adjust length. If near end handle specially.  */
	subq	%rsi, %rdx
	jbe	L(last_4x_vec_or_less)
# endif
	testl	%eax, %eax
	jnz	L(first_vec_x1)

	VPCMP	$0, (VEC_SIZE * 2)(%rdi), %YMMMATCH, %k0
	kmovd	%k0, %eax
	testl	%eax, %eax
	jnz	L(first_vec_x2)

	VPCMP	$0, (VEC_SIZE * 3)(%rdi), %YMMMATCH, %k0
	kmovd	%k0, %eax
	testl	%eax, %eax
	jnz	L(first_vec_x3)

	VPCMP	$0, (VEC_SIZE * 4)(%rdi), %YMMMATCH, %k0
	kmovd	%k0, %eax
	testl	%eax, %eax
	jnz	L(first_vec_x4)


# ifndef USE_AS_RAWMEMCHR
	/* Check if at last CHAR_PER_VEC * 4 length.  */
	subq	$(CHAR_PER_VEC * 4), %rdx
	jbe	L(last_4x_vec_or_less_cmpeq)
	/* +VEC_SIZE if USE_IN_RTM otherwise +VEC_SIZE * 5.  */
	addq	$(VEC_SIZE + (VEC_SIZE * 4 - BASE_OFFSET)), %rdi

	/* Align data to VEC_SIZE * 4 for the loop and readjust length.
	 */
#  ifdef USE_AS_WMEMCHR
	movl	%edi, %ecx
	andq	$-(4 * VEC_SIZE), %rdi
	subl	%edi, %ecx
	/* NB: Divide bytes by 4 to get the wchar_t count.  */
	sarl	$2, %ecx
	addq	%rcx, %rdx
#  else
	addq	%rdi, %rdx
	andq	$-(4 * VEC_SIZE), %rdi
	subq	%rdi, %rdx
#  endif
# else
	addq	$(VEC_SIZE + (VEC_SIZE * 4 - BASE_OFFSET)), %rdi
	andq	$-(4 * VEC_SIZE), %rdi
# endif
# ifdef USE_IN_RTM
	vpxorq	%XMMZERO, %XMMZERO, %XMMZERO
# else
	/* copy ymmmatch to ymm0 so we can use vpcmpeq which is not
	   encodable with EVEX registers (ymm16-ymm31).  */
	vmovdqa64 %YMMMATCH, %ymm0
# endif

	/* Compare 4 * VEC at a time forward.  */
	.p2align 4
L(loop_4x_vec):
	/* Two versions of the loop. One that does not require
	   vzeroupper by not using ymm0-ymm15 and another does that require
	   vzeroupper because it uses ymm0-ymm15. The reason why ymm0-ymm15
	   is used at all is because there is no EVEX encoding vpcmpeq and
	   with vpcmpeq this loop can be performed more efficiently. The
	   non-vzeroupper version is safe for RTM while the vzeroupper
	   version should be prefered if RTM are not supported.  */
# ifdef USE_IN_RTM
	/* It would be possible to save some instructions using 4x VPCMP
	   but bottleneck on port 5 makes it not woth it.  */
	VPCMP	$4, (VEC_SIZE * 4)(%rdi), %YMMMATCH, %k1
	/* xor will set bytes match esi to zero.  */
	vpxorq	(VEC_SIZE * 5)(%rdi), %YMMMATCH, %YMM2
	vpxorq	(VEC_SIZE * 6)(%rdi), %YMMMATCH, %YMM3
	VPCMP	$0, (VEC_SIZE * 7)(%rdi), %YMMMATCH, %k3
	/* Reduce VEC2 / VEC3 with min and VEC1 with zero mask.  */
	VPMINU	%YMM2, %YMM3, %YMM3{%k1}{z}
	VPCMP	$0, %YMM3, %YMMZERO, %k2
# else
	/* Since vptern can only take 3x vectors fastest to do 1 vec
	   seperately with EVEX vpcmp.  */
#  ifdef USE_AS_WMEMCHR
	/* vptern can only accept masks for epi32/epi64 so can only save
	   instruction using not equals mask on vptern with wmemchr.  */
	VPCMP	$4, (%rdi), %YMMMATCH, %k1
#  else
	VPCMP	$0, (%rdi), %YMMMATCH, %k1
#  endif
	/* Compare 3x with vpcmpeq and or them all together with vptern.
	 */
	VPCMPEQ	VEC_SIZE(%rdi), %ymm0, %ymm2
	VPCMPEQ	(VEC_SIZE * 2)(%rdi), %ymm0, %ymm3
	VPCMPEQ	(VEC_SIZE * 3)(%rdi), %ymm0, %ymm4
#  ifdef USE_AS_WMEMCHR
	/* This takes the not of or between ymm2, ymm3, ymm4 as well as
	   combines result from VEC0 with zero mask.  */
	vpternlogd $1, %ymm2, %ymm3, %ymm4{%k1}{z}
	vpmovmskb %ymm4, %ecx
#  else
	/* 254 is mask for oring ymm2, ymm3, ymm4 into ymm4.  */
	vpternlogd $254, %ymm2, %ymm3, %ymm4
	vpmovmskb %ymm4, %ecx
	kmovd	%k1, %eax
#  endif
# endif

# ifdef USE_AS_RAWMEMCHR
	subq	$-(VEC_SIZE * 4), %rdi
# endif
# ifdef USE_IN_RTM
	kortestd %k2, %k3
# else
#  ifdef USE_AS_WMEMCHR
	/* ecx contains not of matches. All 1s means no matches. incl will
	   overflow and set zeroflag if that is the case.  */
	incl	%ecx
#  else
	/* If either VEC1 (eax) or VEC2-VEC4 (ecx) are not zero. Adding
	   to ecx is not an issue because if eax is non-zero it will be
	   used for returning the match. If it is zero the add does
	   nothing.  */
	addq	%rax, %rcx
#  endif
# endif
# ifdef USE_AS_RAWMEMCHR
	jz	L(loop_4x_vec)
# else
	jnz	L(loop_4x_vec_end)

	subq	$-(VEC_SIZE * 4), %rdi

	subq	$(CHAR_PER_VEC * 4), %rdx
	ja	L(loop_4x_vec)

	/* Fall through into less than 4 remaining vectors of length case.
	 */
	VPCMP	$0, BASE_OFFSET(%rdi), %YMMMATCH, %k0
	addq	$(BASE_OFFSET - VEC_SIZE), %rdi
	kmovd	%k0, %eax
	VZEROUPPER

L(last_4x_vec_or_less):
	/* Check if first VEC contained match.  */
	testl	%eax, %eax
	jnz	L(first_vec_x1_check)

	/* If remaining length > CHAR_PER_VEC * 2.  */
	addl	$(CHAR_PER_VEC * 2), %edx
	jg	L(last_4x_vec)

L(last_2x_vec):
	/* If remaining length < CHAR_PER_VEC.  */
	addl	$CHAR_PER_VEC, %edx
	jle	L(zero_end)

	/* Check VEC2 and compare any match with remaining length.  */
	VPCMP	$0, (VEC_SIZE * 2)(%rdi), %YMMMATCH, %k0
	kmovd	%k0, %eax
	tzcntl	%eax, %eax
	cmpl	%eax, %edx
	jbe	L(set_zero_end)
	leaq	(VEC_SIZE * 2)(%rdi, %rax, CHAR_SIZE), %rax
L(zero_end):
	ret


	.p2align 4
L(first_vec_x1_check):
	tzcntl	%eax, %eax
	/* Adjust length.  */
	subl	$-(CHAR_PER_VEC * 4), %edx
	/* Check if match within remaining length.  */
	cmpl	%eax, %edx
	jbe	L(set_zero_end)
	/* NB: Multiply bytes by CHAR_SIZE to get the wchar_t count.  */
	leaq	VEC_SIZE(%rdi, %rax, CHAR_SIZE), %rax
	ret
L(set_zero_end):
	xorl	%eax, %eax
	ret

	.p2align 4
L(loop_4x_vec_end):
# endif
	/* rawmemchr will fall through into this if match was found in
	   loop.  */

# if defined USE_IN_RTM || defined USE_AS_WMEMCHR
	/* k1 has not of matches with VEC1.  */
	kmovd	%k1, %eax
#  ifdef USE_AS_WMEMCHR
	subl	$((1 << CHAR_PER_VEC) - 1), %eax
#  else
	incl	%eax
#  endif
# else
	/* eax already has matches for VEC1.  */
	testl	%eax, %eax
# endif
	jnz	L(last_vec_x1_return)

# ifdef USE_IN_RTM
	VPCMP	$0, %YMM2, %YMMZERO, %k0
	kmovd	%k0, %eax
# else
	vpmovmskb %ymm2, %eax
# endif
	testl	%eax, %eax
	jnz	L(last_vec_x2_return)

# ifdef USE_IN_RTM
	kmovd	%k2, %eax
	testl	%eax, %eax
	jnz	L(last_vec_x3_return)

	kmovd	%k3, %eax
	tzcntl	%eax, %eax
	leaq	(VEC_SIZE * 3 + RET_OFFSET)(%rdi, %rax, CHAR_SIZE), %rax
# else
	vpmovmskb %ymm3, %eax
	/* Combine matches in VEC3 (eax) with matches in VEC4 (ecx).  */
	salq	$VEC_SIZE, %rcx
	orq	%rcx, %rax
	tzcntq	%rax, %rax
	leaq	(VEC_SIZE * 2 + RET_OFFSET)(%rdi, %rax), %rax
	VZEROUPPER
# endif
	ret

	.p2align 4
L(last_vec_x1_return):
	tzcntl	%eax, %eax
# if defined USE_AS_WMEMCHR || RET_OFFSET != 0
	/* NB: Multiply bytes by CHAR_SIZE to get the wchar_t count.  */
	leaq	RET_OFFSET(%rdi, %rax, CHAR_SIZE), %rax
# else
	addq	%rdi, %rax
# endif
	VZEROUPPER
	ret

	.p2align 4
L(last_vec_x2_return):
	tzcntl	%eax, %eax
	/* NB: Multiply bytes by RET_SCALE to get the wchar_t count
	   if relevant (RET_SCALE = CHAR_SIZE if USE_AS_WMEMCHAR and
	   USE_IN_RTM are both defined. Otherwise RET_SCALE = 1.  */
	leaq	(VEC_SIZE + RET_OFFSET)(%rdi, %rax, RET_SCALE), %rax
	VZEROUPPER
	ret

# ifdef USE_IN_RTM
	.p2align 4
L(last_vec_x3_return):
	tzcntl	%eax, %eax
	/* NB: Multiply bytes by CHAR_SIZE to get the wchar_t count.  */
	leaq	(VEC_SIZE * 2 + RET_OFFSET)(%rdi, %rax, CHAR_SIZE), %rax
	ret
# endif

# ifndef USE_AS_RAWMEMCHR
L(last_4x_vec_or_less_cmpeq):
	VPCMP	$0, (VEC_SIZE * 5)(%rdi), %YMMMATCH, %k0
	kmovd	%k0, %eax
	subq	$-(VEC_SIZE * 4), %rdi
	/* Check first VEC regardless.  */
	testl	%eax, %eax
	jnz	L(first_vec_x1_check)

	/* If remaining length <= CHAR_PER_VEC * 2.  */
	addl	$(CHAR_PER_VEC * 2), %edx
	jle	L(last_2x_vec)

	.p2align 4
L(last_4x_vec):
	VPCMP	$0, (VEC_SIZE * 2)(%rdi), %YMMMATCH, %k0
	kmovd	%k0, %eax
	testl	%eax, %eax
	jnz	L(last_vec_x2)


	VPCMP	$0, (VEC_SIZE * 3)(%rdi), %YMMMATCH, %k0
	kmovd	%k0, %eax
	/* Create mask for possible matches within remaining length.  */
#  ifdef USE_AS_WMEMCHR
	movl	$((1 << (CHAR_PER_VEC * 2)) - 1), %ecx
	bzhil	%edx, %ecx, %ecx
#  else
	movq	$-1, %rcx
	bzhiq	%rdx, %rcx, %rcx
#  endif
	/* Test matches in data against length match.  */
	andl	%ecx, %eax
	jnz	L(last_vec_x3)

	/* if remaining length <= CHAR_PER_VEC * 3 (Note this is after
	   remaining length was found to be > CHAR_PER_VEC * 2.  */
	subl	$CHAR_PER_VEC, %edx
	jbe	L(zero_end2)


	VPCMP	$0, (VEC_SIZE * 4)(%rdi), %YMMMATCH, %k0
	kmovd	%k0, %eax
	/* Shift remaining length mask for last VEC.  */
#  ifdef USE_AS_WMEMCHR
	shrl	$CHAR_PER_VEC, %ecx
#  else
	shrq	$CHAR_PER_VEC, %rcx
#  endif
	andl	%ecx, %eax
	jz	L(zero_end2)
	tzcntl	%eax, %eax
	leaq	(VEC_SIZE * 4)(%rdi, %rax, CHAR_SIZE), %rax
L(zero_end2):
	ret

L(last_vec_x2):
	tzcntl	%eax, %eax
	leaq	(VEC_SIZE * 2)(%rdi, %rax, CHAR_SIZE), %rax
	ret

	.p2align 4
L(last_vec_x3):
	tzcntl	%eax, %eax
	leaq	(VEC_SIZE * 3)(%rdi, %rax, CHAR_SIZE), %rax
	ret
# endif

END (MEMCHR)
#endif
