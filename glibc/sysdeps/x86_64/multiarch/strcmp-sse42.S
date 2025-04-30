/* strcmp with SSE4.2
   Copyright (C) 2009-2021 Free Software Foundation, Inc.
   Contributed by Intel Corporation.
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

#include <sysdep.h>

#ifndef STRCMP_SSE42
# define STRCMP_SSE42	__strcmp_sse42
#endif

#if defined USE_AS_STRCASECMP_L || defined USE_AS_STRNCASECMP_L
# include "locale-defines.h"
#endif

#if defined USE_AS_STRNCMP || defined USE_AS_STRNCASECMP_L
/* Since the counter, %r11, is unsigned, we branch to strcmp_exitz
   if the new counter > the old one or is 0.  */
# define UPDATE_STRNCMP_COUNTER				\
	/* calculate left number to compare */		\
	lea	-16(%rcx, %r11), %r9;			\
	cmp	%r9, %r11;				\
	jb	LABEL(strcmp_exitz);			\
	test	%r9, %r9;				\
	je	LABEL(strcmp_exitz);			\
	mov	%r9, %r11
#else
# define UPDATE_STRNCMP_COUNTER
#endif

#ifdef USE_AVX
# define SECTION	avx
# define GLABEL(l)	l##_avx
#else
# define SECTION	sse4.2
# define GLABEL(l)	l##_sse42
#endif

#define LABEL(l)	.L##l

/* We use 0x1a:
	_SIDD_SBYTE_OPS
	| _SIDD_CMP_EQUAL_EACH
	| _SIDD_NEGATIVE_POLARITY
	| _SIDD_LEAST_SIGNIFICANT
   on pcmpistri to find out if two 16byte data elements are the same
   and the offset of the first different byte.  There are 4 cases:

   1. Both 16byte data elements are valid and identical.
   2. Both 16byte data elements have EOS and identical.
   3. Both 16byte data elements are valid and they differ at offset X.
   4. At least one 16byte data element has EOS at offset X.  Two 16byte
      data elements must differ at or before offset X.

   Here is the table of ECX, CFlag, ZFlag and SFlag for 4 cases:

   case		ECX	CFlag	ZFlag	SFlag
    1		16	  0	  0	  0
    2		16	  0	  1	  1
    3		 X	  1	  0	  0
    4	       0 <= X	  1	 0/1	 0/1

   We exit from the loop for cases 2, 3 and 4 with jbe which branches
   when either CFlag or ZFlag is 1.  If CFlag == 0, we return 0 for
   case 2.  */

	/* Put all SSE 4.2 functions together.  */
	.section .text.SECTION,"ax",@progbits
	.align	16
	.type	STRCMP_SSE42, @function
	.globl	STRCMP_SSE42
	.hidden	STRCMP_SSE42
#ifdef USE_AS_STRCASECMP_L
ENTRY (GLABEL(__strcasecmp))
	movq	__libc_tsd_LOCALE@gottpoff(%rip),%rax
	mov	%fs:(%rax),%RDX_LP

	// XXX 5 byte should be before the function
	/* 5-byte NOP.  */
	.byte	0x0f,0x1f,0x44,0x00,0x00
END (GLABEL(__strcasecmp))
	/* FALLTHROUGH to strcasecmp_l.  */
#endif
#ifdef USE_AS_STRNCASECMP_L
ENTRY (GLABEL(__strncasecmp))
	movq	__libc_tsd_LOCALE@gottpoff(%rip),%rax
	mov	%fs:(%rax),%RCX_LP

	// XXX 5 byte should be before the function
	/* 5-byte NOP.  */
	.byte	0x0f,0x1f,0x44,0x00,0x00
END (GLABEL(__strncasecmp))
	/* FALLTHROUGH to strncasecmp_l.  */
#endif


#ifdef USE_AVX
# define movdqa vmovdqa
# define movdqu vmovdqu
# define pmovmskb vpmovmskb
# define pcmpistri vpcmpistri
# define psubb vpsubb
# define pcmpeqb vpcmpeqb
# define psrldq vpsrldq
# define pslldq vpslldq
# define palignr vpalignr
# define pxor vpxor
# define D(arg) arg, arg
#else
# define D(arg) arg
#endif

STRCMP_SSE42:
	cfi_startproc
	_CET_ENDBR
	CALL_MCOUNT

/*
 * This implementation uses SSE to compare up to 16 bytes at a time.
 */
#ifdef USE_AS_STRCASECMP_L
	/* We have to fall back on the C implementation for locales
	   with encodings not matching ASCII for single bytes.  */
# if LOCALE_T___LOCALES != 0 || LC_CTYPE != 0
	mov	LOCALE_T___LOCALES+LC_CTYPE*LP_SIZE(%rdx), %RAX_LP
# else
	mov	(%rdx), %RAX_LP
# endif
	testl	$1, LOCALE_DATA_VALUES+_NL_CTYPE_NONASCII_CASE*SIZEOF_VALUES(%rax)
	jne	__strcasecmp_l_nonascii
#endif
#ifdef USE_AS_STRNCASECMP_L
	/* We have to fall back on the C implementation for locales
	   with encodings not matching ASCII for single bytes.  */
# if LOCALE_T___LOCALES != 0 || LC_CTYPE != 0
	mov	LOCALE_T___LOCALES+LC_CTYPE*LP_SIZE(%rcx), %RAX_LP
# else
	mov	(%rcx), %RAX_LP
# endif
	testl	$1, LOCALE_DATA_VALUES+_NL_CTYPE_NONASCII_CASE*SIZEOF_VALUES(%rax)
	jne	__strncasecmp_l_nonascii
#endif

#if defined USE_AS_STRNCMP || defined USE_AS_STRNCASECMP_L
	test	%RDX_LP, %RDX_LP
	je	LABEL(strcmp_exitz)
	cmp	$1, %RDX_LP
	je	LABEL(Byte0)
	mov	%RDX_LP, %R11_LP
#endif
	mov	%esi, %ecx
	mov	%edi, %eax
/* Use 64bit AND here to avoid long NOP padding.  */
	and	$0x3f, %rcx		/* rsi alignment in cache line */
	and	$0x3f, %rax		/* rdi alignment in cache line */
#if defined USE_AS_STRCASECMP_L || defined USE_AS_STRNCASECMP_L
	.section .rodata.cst16,"aM",@progbits,16
	.align 16
LABEL(belowupper):
	.quad	0x4040404040404040
	.quad	0x4040404040404040
LABEL(topupper):
# ifdef USE_AVX
	.quad	0x5a5a5a5a5a5a5a5a
	.quad	0x5a5a5a5a5a5a5a5a
# else
	.quad	0x5b5b5b5b5b5b5b5b
	.quad	0x5b5b5b5b5b5b5b5b
# endif
LABEL(touppermask):
	.quad	0x2020202020202020
	.quad	0x2020202020202020
	.previous
	movdqa	LABEL(belowupper)(%rip), %xmm4
# define UCLOW_reg %xmm4
	movdqa	LABEL(topupper)(%rip), %xmm5
# define UCHIGH_reg %xmm5
	movdqa	LABEL(touppermask)(%rip), %xmm6
# define LCQWORD_reg %xmm6
#endif
	cmp	$0x30, %ecx
	ja	LABEL(crosscache)/* rsi: 16-byte load will cross cache line */
	cmp	$0x30, %eax
	ja	LABEL(crosscache)/* rdi: 16-byte load will cross cache line */
	movdqu	(%rdi), %xmm1
	movdqu	(%rsi), %xmm2
#if defined USE_AS_STRCASECMP_L || defined USE_AS_STRNCASECMP_L
# ifdef USE_AVX
#  define TOLOWER(reg1, reg2) \
	vpcmpgtb UCLOW_reg, reg1, %xmm7;			\
	vpcmpgtb UCHIGH_reg, reg1, %xmm8;			\
	vpcmpgtb UCLOW_reg, reg2, %xmm9;			\
	vpcmpgtb UCHIGH_reg, reg2, %xmm10;			\
	vpandn	%xmm7, %xmm8, %xmm8;					\
	vpandn	%xmm9, %xmm10, %xmm10;					\
	vpand	LCQWORD_reg, %xmm8, %xmm8;				\
	vpand	LCQWORD_reg, %xmm10, %xmm10;				\
	vpor	reg1, %xmm8, reg1;					\
	vpor	reg2, %xmm10, reg2
# else
#  define TOLOWER(reg1, reg2) \
	movdqa	reg1, %xmm7;					\
	movdqa	UCHIGH_reg, %xmm8;				\
	movdqa	reg2, %xmm9;					\
	movdqa	UCHIGH_reg, %xmm10;				\
	pcmpgtb	UCLOW_reg, %xmm7;				\
	pcmpgtb	reg1, %xmm8;					\
	pcmpgtb	UCLOW_reg, %xmm9;				\
	pcmpgtb	reg2, %xmm10;					\
	pand	%xmm8, %xmm7;					\
	pand	%xmm10, %xmm9;					\
	pand	LCQWORD_reg, %xmm7;				\
	pand	LCQWORD_reg, %xmm9;				\
	por	%xmm7, reg1;					\
	por	%xmm9, reg2
# endif
	TOLOWER (%xmm1, %xmm2)
#else
# define TOLOWER(reg1, reg2)
#endif
	pxor	%xmm0, D(%xmm0)		/* clear %xmm0 for null char checks */
	pcmpeqb	%xmm1, D(%xmm0)		/* Any null chars? */
	pcmpeqb	%xmm2, D(%xmm1)		/* compare first 16 bytes for equality */
	psubb	%xmm0, D(%xmm1)		/* packed sub of comparison results*/
	pmovmskb %xmm1, %edx
	sub	$0xffff, %edx		/* if first 16 bytes are same, edx == 0xffff */
	jnz	LABEL(less16bytes)/* If not, find different value or null char */
#if defined USE_AS_STRNCMP || defined USE_AS_STRNCASECMP_L
	sub	$16, %r11
	jbe	LABEL(strcmp_exitz)/* finish comparison */
#endif
	add	$16, %rsi		/* prepare to search next 16 bytes */
	add	$16, %rdi		/* prepare to search next 16 bytes */

	/*
	 * Determine source and destination string offsets from 16-byte
	 * alignment.  Use relative offset difference between the two to
	 * determine which case below to use.
	 */
	.p2align 4
LABEL(crosscache):
	and	$0xfffffffffffffff0, %rsi /* force %rsi is 16 byte aligned */
	and	$0xfffffffffffffff0, %rdi /* force %rdi is 16 byte aligned */
	mov	$0xffff, %edx		/* for equivalent offset */
	xor	%r8d, %r8d
	and	$0xf, %ecx		/* offset of rsi */
	and	$0xf, %eax		/* offset of rdi */
	pxor	%xmm0, D(%xmm0)		/* clear %xmm0 for null char check */
	cmp	%eax, %ecx
	je	LABEL(ashr_0)		/* rsi and rdi relative offset same */
	ja	LABEL(bigger)
	mov	%edx, %r8d		/* r8d is offset flag for exit tail */
	xchg	%ecx, %eax
	xchg	%rsi, %rdi
LABEL(bigger):
	movdqa	(%rdi), %xmm2
	movdqa	(%rsi), %xmm1
	lea	15(%rax), %r9
	sub	%rcx, %r9
	lea	LABEL(unaligned_table)(%rip), %r10
	movslq	(%r10, %r9,4), %r9
	pcmpeqb	%xmm1, D(%xmm0)		/* Any null chars? */
	lea	(%r10, %r9), %r10
	_CET_NOTRACK jmp *%r10		/* jump to corresponding case */

/*
 * The following cases will be handled by ashr_0
 *  rcx(offset of rsi)  rax(offset of rdi)  relative offset  corresponding case
 *        n(0~15)            n(0~15)           15(15+ n-n)         ashr_0
 */
	.p2align 4
LABEL(ashr_0):

	movdqa	(%rsi), %xmm1
	pcmpeqb	%xmm1, D(%xmm0)		/* Any null chars? */
#if !defined USE_AS_STRCASECMP_L && !defined USE_AS_STRNCASECMP_L
	pcmpeqb	(%rdi), D(%xmm1)	/* compare 16 bytes for equality */
#else
	movdqa	(%rdi), %xmm2
	TOLOWER (%xmm1, %xmm2)
	pcmpeqb	%xmm2, D(%xmm1)		/* compare 16 bytes for equality */
#endif
	psubb	%xmm0, D(%xmm1)		/* packed sub of comparison results*/
	pmovmskb %xmm1, %r9d
	shr	%cl, %edx		/* adjust 0xffff for offset */
	shr	%cl, %r9d		/* adjust for 16-byte offset */
	sub	%r9d, %edx
	/*
	 * edx must be the same with r9d if in left byte (16-rcx) is equal to
	 * the start from (16-rax) and no null char was seen.
	 */
	jne	LABEL(less32bytes)	/* mismatch or null char */
	UPDATE_STRNCMP_COUNTER
	mov	$16, %rcx
	mov	$16, %r9

	/*
	 * Now both strings are aligned at 16-byte boundary. Loop over strings
	 * checking 32-bytes per iteration.
	 */
	mov	%rcx, %rdx	/* only for offset of sse4 instruction loop*/
	.p2align 4
LABEL(ashr_0_use):
	movdqa	(%rdi,%rdx), %xmm0
#if !defined USE_AS_STRCASECMP_L && !defined USE_AS_STRNCASECMP_L
	pcmpistri      $0x1a,(%rsi,%rdx), %xmm0
#else
	movdqa	(%rsi,%rdx), %xmm1
	TOLOWER (%xmm0, %xmm1)
	pcmpistri $0x1a, %xmm1, %xmm0
#endif
	lea	16(%rdx), %rdx
	jbe	LABEL(ashr_0_exit_use)
#if defined USE_AS_STRNCMP || defined USE_AS_STRNCASECMP_L
	sub	$16, %r11
	jbe	LABEL(strcmp_exitz)
#endif

	movdqa	(%rdi,%rdx), %xmm0
#if !defined USE_AS_STRCASECMP_L && !defined USE_AS_STRNCASECMP_L
	pcmpistri      $0x1a,(%rsi,%rdx), %xmm0
#else
	movdqa	(%rsi,%rdx), %xmm1
	TOLOWER (%xmm0, %xmm1)
	pcmpistri $0x1a, %xmm1, %xmm0
#endif
	lea	16(%rdx), %rdx
	jbe	LABEL(ashr_0_exit_use)
#if defined USE_AS_STRNCMP || defined USE_AS_STRNCASECMP_L
	sub	$16, %r11
	jbe	LABEL(strcmp_exitz)
#endif
	jmp	LABEL(ashr_0_use)


	.p2align 4
LABEL(ashr_0_exit_use):
	jnc	LABEL(strcmp_exitz)
#if defined USE_AS_STRNCMP || defined USE_AS_STRNCASECMP_L
	sub	%rcx, %r11
	jbe	LABEL(strcmp_exitz)
#endif
	lea	-16(%rdx, %rcx), %rcx
	movzbl	(%rdi, %rcx), %eax
	movzbl	(%rsi, %rcx), %edx
#if defined USE_AS_STRCASECMP_L || defined USE_AS_STRNCASECMP_L
	leaq	_nl_C_LC_CTYPE_tolower+128*4(%rip), %rcx
	movl	(%rcx,%rax,4), %eax
	movl	(%rcx,%rdx,4), %edx
#endif
	sub	%edx, %eax
	ret



/*
 * The following cases will be handled by ashr_1
 * rcx(offset of rsi)  rax(offset of rdi)   relative offset	corresponding case
 *        n(15)            n -15            0(15 +(n-15) - n)         ashr_1
 */
	.p2align 4
LABEL(ashr_1):
	pslldq	$15, D(%xmm2)		/* shift first string to align with second */
	TOLOWER (%xmm1, %xmm2)
	pcmpeqb	%xmm1, D(%xmm2)		/* compare 16 bytes for equality */
	psubb	%xmm0, D(%xmm2)		/* packed sub of comparison results*/
	pmovmskb %xmm2, %r9d
	shr	%cl, %edx		/* adjust 0xffff for offset */
	shr	%cl, %r9d		/* adjust for 16-byte offset */
	sub	%r9d, %edx
	jnz	LABEL(less32bytes)	/* mismatch or null char seen */
	movdqa	(%rdi), %xmm3
	UPDATE_STRNCMP_COUNTER

	mov	$16, %rcx		/* index for loads*/
	mov	$1, %r9d		/* byte position left over from less32bytes case */
	/*
	 * Setup %r10 value allows us to detect crossing a page boundary.
	 * When %r10 goes positive we have crossed a page boundary and
	 * need to do a nibble.
	 */
	lea	1(%rdi), %r10
	and	$0xfff, %r10		/* offset into 4K page */
	sub	$0x1000, %r10		/* subtract 4K pagesize */
	mov	%rcx, %rdx	/* only for offset of sse4 instruction loop*/

	.p2align 4
LABEL(loop_ashr_1_use):
	add	$16, %r10
	jg	LABEL(nibble_ashr_1_use)

LABEL(nibble_ashr_1_restart_use):
	movdqa	(%rdi, %rdx), %xmm0
	palignr $1, -16(%rdi, %rdx), D(%xmm0)
#if !defined USE_AS_STRCASECMP_L && !defined USE_AS_STRNCASECMP_L
	pcmpistri	$0x1a,(%rsi,%rdx), %xmm0
#else
	movdqa	(%rsi,%rdx), %xmm1
	TOLOWER (%xmm0, %xmm1)
	pcmpistri $0x1a, %xmm1, %xmm0
#endif
	jbe	LABEL(exit_use)
#if defined USE_AS_STRNCMP || defined USE_AS_STRNCASECMP_L
	sub	$16, %r11
	jbe	LABEL(strcmp_exitz)
#endif

	add	$16, %rdx
	add	$16, %r10
	jg	LABEL(nibble_ashr_1_use)

	movdqa	(%rdi, %rdx), %xmm0
	palignr $1, -16(%rdi, %rdx), D(%xmm0)
#if !defined USE_AS_STRCASECMP_L && !defined USE_AS_STRNCASECMP_L
	pcmpistri	$0x1a,(%rsi,%rdx), %xmm0
#else
	movdqa	(%rsi,%rdx), %xmm1
	TOLOWER (%xmm0, %xmm1)
	pcmpistri $0x1a, %xmm1, %xmm0
#endif
	jbe	LABEL(exit_use)
#if defined USE_AS_STRNCMP || defined USE_AS_STRNCASECMP_L
	sub	$16, %r11
	jbe	LABEL(strcmp_exitz)
#endif
	add	$16, %rdx
	jmp	LABEL(loop_ashr_1_use)

	.p2align 4
LABEL(nibble_ashr_1_use):
	sub	$0x1000, %r10
	movdqa	-16(%rdi, %rdx), %xmm0
	psrldq	$1, D(%xmm0)
	pcmpistri      $0x3a,%xmm0, %xmm0
#if defined USE_AS_STRNCMP || defined USE_AS_STRNCASECMP_L
	cmp	%r11, %rcx
	jae	LABEL(nibble_ashr_exit_use)
#endif
	cmp	$14, %ecx
	ja	LABEL(nibble_ashr_1_restart_use)

	jmp	LABEL(nibble_ashr_exit_use)

/*
 * The following cases will be handled by ashr_2
 * rcx(offset of rsi)  rax(offset of rdi)   relative offset	corresponding case
 *        n(14~15)            n -14         1(15 +(n-14) - n)         ashr_2
 */
	.p2align 4
LABEL(ashr_2):
	pslldq	$14, D(%xmm2)
	TOLOWER (%xmm1, %xmm2)
	pcmpeqb	%xmm1, D(%xmm2)
	psubb	%xmm0, D(%xmm2)
	pmovmskb %xmm2, %r9d
	shr	%cl, %edx
	shr	%cl, %r9d
	sub	%r9d, %edx
	jnz	LABEL(less32bytes)
	movdqa	(%rdi), %xmm3
	UPDATE_STRNCMP_COUNTER

	mov	$16, %rcx	/* index for loads */
	mov	$2, %r9d	/* byte position left over from less32bytes case */
	/*
	 * Setup %r10 value allows us to detect crossing a page boundary.
	 * When %r10 goes positive we have crossed a page boundary and
	 * need to do a nibble.
	 */
	lea	2(%rdi), %r10
	and	$0xfff, %r10	/* offset into 4K page */
	sub	$0x1000, %r10	/* subtract 4K pagesize */
	mov	%rcx, %rdx	/* only for offset of sse4 instruction loop*/

	.p2align 4
LABEL(loop_ashr_2_use):
	add	$16, %r10
	jg	LABEL(nibble_ashr_2_use)

LABEL(nibble_ashr_2_restart_use):
	movdqa	(%rdi, %rdx), %xmm0
	palignr $2, -16(%rdi, %rdx), D(%xmm0)
#if !defined USE_AS_STRCASECMP_L && !defined USE_AS_STRNCASECMP_L
	pcmpistri	$0x1a,(%rsi,%rdx), %xmm0
#else
	movdqa	(%rsi,%rdx), %xmm1
	TOLOWER (%xmm0, %xmm1)
	pcmpistri $0x1a, %xmm1, %xmm0
#endif
	jbe	LABEL(exit_use)
#if defined USE_AS_STRNCMP || defined USE_AS_STRNCASECMP_L
	sub	$16, %r11
	jbe	LABEL(strcmp_exitz)
#endif

	add	$16, %rdx
	add	$16, %r10
	jg	LABEL(nibble_ashr_2_use)

	movdqa	(%rdi, %rdx), %xmm0
	palignr $2, -16(%rdi, %rdx), D(%xmm0)
#if !defined USE_AS_STRCASECMP_L && !defined USE_AS_STRNCASECMP_L
	pcmpistri	$0x1a,(%rsi,%rdx), %xmm0
#else
	movdqa	(%rsi,%rdx), %xmm1
	TOLOWER (%xmm0, %xmm1)
	pcmpistri $0x1a, %xmm1, %xmm0
#endif
	jbe	LABEL(exit_use)
#if defined USE_AS_STRNCMP || defined USE_AS_STRNCASECMP_L
	sub	$16, %r11
	jbe	LABEL(strcmp_exitz)
#endif
	add	$16, %rdx
	jmp	LABEL(loop_ashr_2_use)

	.p2align 4
LABEL(nibble_ashr_2_use):
	sub	$0x1000, %r10
	movdqa	-16(%rdi, %rdx), %xmm0
	psrldq	$2, D(%xmm0)
	pcmpistri      $0x3a,%xmm0, %xmm0
#if defined USE_AS_STRNCMP || defined USE_AS_STRNCASECMP_L
	cmp	%r11, %rcx
	jae	LABEL(nibble_ashr_exit_use)
#endif
	cmp	$13, %ecx
	ja	LABEL(nibble_ashr_2_restart_use)

	jmp	LABEL(nibble_ashr_exit_use)

/*
 * The following cases will be handled by ashr_3
 *  rcx(offset of rsi)  rax(offset of rdi)  relative offset	 corresponding case
 *        n(13~15)            n -13         2(15 +(n-13) - n)         ashr_3
 */
	.p2align 4
LABEL(ashr_3):
	pslldq	$13, D(%xmm2)
	TOLOWER (%xmm1, %xmm2)
	pcmpeqb	%xmm1, D(%xmm2)
	psubb	%xmm0, D(%xmm2)
	pmovmskb %xmm2, %r9d
	shr	%cl, %edx
	shr	%cl, %r9d
	sub	%r9d, %edx
	jnz	LABEL(less32bytes)
	movdqa	(%rdi), %xmm3

	UPDATE_STRNCMP_COUNTER

	mov	$16, %rcx	/* index for loads */
	mov	$3, %r9d	/* byte position left over from less32bytes case */
	/*
	 * Setup %r10 value allows us to detect crossing a page boundary.
	 * When %r10 goes positive we have crossed a page boundary and
	 * need to do a nibble.
	 */
	lea	3(%rdi), %r10
	and	$0xfff, %r10	/* offset into 4K page */
	sub	$0x1000, %r10	/* subtract 4K pagesize */
	mov	%rcx, %rdx	/* only for offset of sse4 instruction loop*/

LABEL(loop_ashr_3_use):
	add	$16, %r10
	jg	LABEL(nibble_ashr_3_use)

LABEL(nibble_ashr_3_restart_use):
	movdqa	(%rdi, %rdx), %xmm0
	palignr $3, -16(%rdi, %rdx), D(%xmm0)
#if !defined USE_AS_STRCASECMP_L && !defined USE_AS_STRNCASECMP_L
	pcmpistri	$0x1a,(%rsi,%rdx), %xmm0
#else
	movdqa	(%rsi,%rdx), %xmm1
	TOLOWER (%xmm0, %xmm1)
	pcmpistri $0x1a, %xmm1, %xmm0
#endif
	jbe	LABEL(exit_use)
#if defined USE_AS_STRNCMP || defined USE_AS_STRNCASECMP_L
	sub	$16, %r11
	jbe	LABEL(strcmp_exitz)
#endif

	add	$16, %rdx
	add	$16, %r10
	jg	LABEL(nibble_ashr_3_use)

	movdqa	(%rdi, %rdx), %xmm0
	palignr $3, -16(%rdi, %rdx), D(%xmm0)
#if !defined USE_AS_STRCASECMP_L && !defined USE_AS_STRNCASECMP_L
	pcmpistri	$0x1a,(%rsi,%rdx), %xmm0
#else
	movdqa	(%rsi,%rdx), %xmm1
	TOLOWER (%xmm0, %xmm1)
	pcmpistri $0x1a, %xmm1, %xmm0
#endif
	jbe	LABEL(exit_use)
#if defined USE_AS_STRNCMP || defined USE_AS_STRNCASECMP_L
	sub	$16, %r11
	jbe	LABEL(strcmp_exitz)
#endif
	add	$16, %rdx
	jmp	LABEL(loop_ashr_3_use)

	.p2align 4
LABEL(nibble_ashr_3_use):
	sub	$0x1000, %r10
	movdqa	-16(%rdi, %rdx), %xmm0
	psrldq	$3, D(%xmm0)
	pcmpistri      $0x3a,%xmm0, %xmm0
#if defined USE_AS_STRNCMP || defined USE_AS_STRNCASECMP_L
	cmp	%r11, %rcx
	jae	LABEL(nibble_ashr_exit_use)
#endif
	cmp	$12, %ecx
	ja	LABEL(nibble_ashr_3_restart_use)

	jmp	LABEL(nibble_ashr_exit_use)

/*
 * The following cases will be handled by ashr_4
 *  rcx(offset of rsi)  rax(offset of rdi)  relative offset	 corresponding case
 *        n(12~15)            n -12         3(15 +(n-12) - n)         ashr_4
 */
	.p2align 4
LABEL(ashr_4):
	pslldq	$12, D(%xmm2)
	TOLOWER (%xmm1, %xmm2)
	pcmpeqb	%xmm1, D(%xmm2)
	psubb	%xmm0, D(%xmm2)
	pmovmskb %xmm2, %r9d
	shr	%cl, %edx
	shr	%cl, %r9d
	sub	%r9d, %edx
	jnz	LABEL(less32bytes)
	movdqa	(%rdi), %xmm3

	UPDATE_STRNCMP_COUNTER

	mov	$16, %rcx	/* index for loads */
	mov	$4, %r9d	/* byte position left over from less32bytes case */
	/*
	 * Setup %r10 value allows us to detect crossing a page boundary.
	 * When %r10 goes positive we have crossed a page boundary and
	 * need to do a nibble.
	 */
	lea	4(%rdi), %r10
	and	$0xfff, %r10	/* offset into 4K page */
	sub	$0x1000, %r10	/* subtract 4K pagesize */
	mov	%rcx, %rdx	/* only for offset of sse4 instruction loop*/

	.p2align 4
LABEL(loop_ashr_4_use):
	add	$16, %r10
	jg	LABEL(nibble_ashr_4_use)

LABEL(nibble_ashr_4_restart_use):
	movdqa	(%rdi, %rdx), %xmm0
	palignr $4, -16(%rdi, %rdx), D(%xmm0)
#if !defined USE_AS_STRCASECMP_L && !defined USE_AS_STRNCASECMP_L
	pcmpistri	$0x1a,(%rsi,%rdx), %xmm0
#else
	movdqa	(%rsi,%rdx), %xmm1
	TOLOWER (%xmm0, %xmm1)
	pcmpistri $0x1a, %xmm1, %xmm0
#endif
	jbe	LABEL(exit_use)
#if defined USE_AS_STRNCMP || defined USE_AS_STRNCASECMP_L
	sub	$16, %r11
	jbe	LABEL(strcmp_exitz)
#endif

	add	$16, %rdx
	add	$16, %r10
	jg	LABEL(nibble_ashr_4_use)

	movdqa	(%rdi, %rdx), %xmm0
	palignr $4, -16(%rdi, %rdx), D(%xmm0)
#if !defined USE_AS_STRCASECMP_L && !defined USE_AS_STRNCASECMP_L
	pcmpistri	$0x1a,(%rsi,%rdx), %xmm0
#else
	movdqa	(%rsi,%rdx), %xmm1
	TOLOWER (%xmm0, %xmm1)
	pcmpistri $0x1a, %xmm1, %xmm0
#endif
	jbe	LABEL(exit_use)
#if defined USE_AS_STRNCMP || defined USE_AS_STRNCASECMP_L
	sub	$16, %r11
	jbe	LABEL(strcmp_exitz)
#endif
	add	$16, %rdx
	jmp	LABEL(loop_ashr_4_use)

	.p2align 4
LABEL(nibble_ashr_4_use):
	sub	$0x1000, %r10
	movdqa	-16(%rdi, %rdx), %xmm0
	psrldq	$4, D(%xmm0)
	pcmpistri      $0x3a,%xmm0, %xmm0
#if defined USE_AS_STRNCMP || defined USE_AS_STRNCASECMP_L
	cmp	%r11, %rcx
	jae	LABEL(nibble_ashr_exit_use)
#endif
	cmp	$11, %ecx
	ja	LABEL(nibble_ashr_4_restart_use)

	jmp	LABEL(nibble_ashr_exit_use)

/*
 * The following cases will be handled by ashr_5
 *  rcx(offset of rsi)  rax(offset of rdi)        relative offset      corresponding case
 *        n(11~15)          n - 11		  4(15 +(n-11) - n)         ashr_5
 */
	.p2align 4
LABEL(ashr_5):
	pslldq	$11, D(%xmm2)
	TOLOWER (%xmm1, %xmm2)
	pcmpeqb	%xmm1, D(%xmm2)
	psubb	%xmm0, D(%xmm2)
	pmovmskb %xmm2, %r9d
	shr	%cl, %edx
	shr	%cl, %r9d
	sub	%r9d, %edx
	jnz	LABEL(less32bytes)
	movdqa	(%rdi), %xmm3

	UPDATE_STRNCMP_COUNTER

	mov	$16, %rcx	/* index for loads */
	mov	$5, %r9d	/* byte position left over from less32bytes case */
	/*
	 * Setup %r10 value allows us to detect crossing a page boundary.
	 * When %r10 goes positive we have crossed a page boundary and
	 * need to do a nibble.
	 */
	lea	5(%rdi), %r10
	and	$0xfff, %r10	/* offset into 4K page */
	sub	$0x1000, %r10	/* subtract 4K pagesize */
	mov	%rcx, %rdx	/* only for offset of sse4 instruction loop*/

	.p2align 4
LABEL(loop_ashr_5_use):
	add	$16, %r10
	jg	LABEL(nibble_ashr_5_use)

LABEL(nibble_ashr_5_restart_use):
	movdqa	(%rdi, %rdx), %xmm0
	palignr $5, -16(%rdi, %rdx), D(%xmm0)
#if !defined USE_AS_STRCASECMP_L && !defined USE_AS_STRNCASECMP_L
	pcmpistri	$0x1a,(%rsi,%rdx), %xmm0
#else
	movdqa	(%rsi,%rdx), %xmm1
	TOLOWER (%xmm0, %xmm1)
	pcmpistri $0x1a, %xmm1, %xmm0
#endif
	jbe	LABEL(exit_use)
#if defined USE_AS_STRNCMP || defined USE_AS_STRNCASECMP_L
	sub	$16, %r11
	jbe	LABEL(strcmp_exitz)
#endif

	add	$16, %rdx
	add	$16, %r10
	jg	LABEL(nibble_ashr_5_use)

	movdqa	(%rdi, %rdx), %xmm0

	palignr $5, -16(%rdi, %rdx), D(%xmm0)
#if !defined USE_AS_STRCASECMP_L && !defined USE_AS_STRNCASECMP_L
	pcmpistri	$0x1a,(%rsi,%rdx), %xmm0
#else
	movdqa	(%rsi,%rdx), %xmm1
	TOLOWER (%xmm0, %xmm1)
	pcmpistri $0x1a, %xmm1, %xmm0
#endif
	jbe	LABEL(exit_use)
#if defined USE_AS_STRNCMP || defined USE_AS_STRNCASECMP_L
	sub	$16, %r11
	jbe	LABEL(strcmp_exitz)
#endif
	add	$16, %rdx
	jmp	LABEL(loop_ashr_5_use)

	.p2align 4
LABEL(nibble_ashr_5_use):
	sub	$0x1000, %r10
	movdqa	-16(%rdi, %rdx), %xmm0
	psrldq	$5, D(%xmm0)
	pcmpistri      $0x3a,%xmm0, %xmm0
#if defined USE_AS_STRNCMP || defined USE_AS_STRNCASECMP_L
	cmp	%r11, %rcx
	jae	LABEL(nibble_ashr_exit_use)
#endif
	cmp	$10, %ecx
	ja	LABEL(nibble_ashr_5_restart_use)

	jmp	LABEL(nibble_ashr_exit_use)

/*
 * The following cases will be handled by ashr_6
 *  rcx(offset of rsi)  rax(offset of rdi)        relative offset      corresponding case
 *        n(10~15)          n - 10		  5(15 +(n-10) - n)         ashr_6
 */
	.p2align 4
LABEL(ashr_6):
	pslldq	$10, D(%xmm2)
	TOLOWER (%xmm1, %xmm2)
	pcmpeqb	%xmm1, D(%xmm2)
	psubb	%xmm0, D(%xmm2)
	pmovmskb %xmm2, %r9d
	shr	%cl, %edx
	shr	%cl, %r9d
	sub	%r9d, %edx
	jnz	LABEL(less32bytes)
	movdqa	(%rdi), %xmm3

	UPDATE_STRNCMP_COUNTER

	mov	$16, %rcx	/* index for loads */
	mov	$6, %r9d	/* byte position left over from less32bytes case */
	/*
	 * Setup %r10 value allows us to detect crossing a page boundary.
	 * When %r10 goes positive we have crossed a page boundary and
	 * need to do a nibble.
	 */
	lea	6(%rdi), %r10
	and	$0xfff, %r10	/* offset into 4K page */
	sub	$0x1000, %r10	/* subtract 4K pagesize */
	mov	%rcx, %rdx	/* only for offset of sse4 instruction loop*/

	.p2align 4
LABEL(loop_ashr_6_use):
	add	$16, %r10
	jg	LABEL(nibble_ashr_6_use)

LABEL(nibble_ashr_6_restart_use):
	movdqa	(%rdi, %rdx), %xmm0
	palignr $6, -16(%rdi, %rdx), D(%xmm0)
#if !defined USE_AS_STRCASECMP_L && !defined USE_AS_STRNCASECMP_L
	pcmpistri $0x1a,(%rsi,%rdx), %xmm0
#else
	movdqa	(%rsi,%rdx), %xmm1
	TOLOWER (%xmm0, %xmm1)
	pcmpistri $0x1a, %xmm1, %xmm0
#endif
	jbe	LABEL(exit_use)
#if defined USE_AS_STRNCMP || defined USE_AS_STRNCASECMP_L
	sub	$16, %r11
	jbe	LABEL(strcmp_exitz)
#endif

	add	$16, %rdx
	add	$16, %r10
	jg	LABEL(nibble_ashr_6_use)

	movdqa	(%rdi, %rdx), %xmm0
	palignr $6, -16(%rdi, %rdx), D(%xmm0)
#if !defined USE_AS_STRCASECMP_L && !defined USE_AS_STRNCASECMP_L
	pcmpistri $0x1a,(%rsi,%rdx), %xmm0
#else
	movdqa	(%rsi,%rdx), %xmm1
	TOLOWER (%xmm0, %xmm1)
	pcmpistri $0x1a, %xmm1, %xmm0
#endif
	jbe	LABEL(exit_use)
#if defined USE_AS_STRNCMP || defined USE_AS_STRNCASECMP_L
	sub	$16, %r11
	jbe	LABEL(strcmp_exitz)
#endif
	add	$16, %rdx
	jmp	LABEL(loop_ashr_6_use)

	.p2align 4
LABEL(nibble_ashr_6_use):
	sub	$0x1000, %r10
	movdqa	-16(%rdi, %rdx), %xmm0
	psrldq	$6, D(%xmm0)
	pcmpistri      $0x3a,%xmm0, %xmm0
#if defined USE_AS_STRNCMP || defined USE_AS_STRNCASECMP_L
	cmp	%r11, %rcx
	jae	LABEL(nibble_ashr_exit_use)
#endif
	cmp	$9, %ecx
	ja	LABEL(nibble_ashr_6_restart_use)

	jmp	LABEL(nibble_ashr_exit_use)

/*
 * The following cases will be handled by ashr_7
 *  rcx(offset of rsi)  rax(offset of rdi)        relative offset      corresponding case
 *        n(9~15)          n - 9		  6(15 +(n - 9) - n)         ashr_7
 */
	.p2align 4
LABEL(ashr_7):
	pslldq	$9, D(%xmm2)
	TOLOWER (%xmm1, %xmm2)
	pcmpeqb	%xmm1, D(%xmm2)
	psubb	%xmm0, D(%xmm2)
	pmovmskb %xmm2, %r9d
	shr	%cl, %edx
	shr	%cl, %r9d
	sub	%r9d, %edx
	jnz	LABEL(less32bytes)
	movdqa	(%rdi), %xmm3

	UPDATE_STRNCMP_COUNTER

	mov	$16, %rcx	/* index for loads */
	mov	$7, %r9d	/* byte position left over from less32bytes case */
	/*
	 * Setup %r10 value allows us to detect crossing a page boundary.
	 * When %r10 goes positive we have crossed a page boundary and
	 * need to do a nibble.
	 */
	lea	7(%rdi), %r10
	and	$0xfff, %r10	/* offset into 4K page */
	sub	$0x1000, %r10	/* subtract 4K pagesize */
	mov	%rcx, %rdx	/* only for offset of sse4 instruction loop*/

	.p2align 4
LABEL(loop_ashr_7_use):
	add	$16, %r10
	jg	LABEL(nibble_ashr_7_use)

LABEL(nibble_ashr_7_restart_use):
	movdqa	(%rdi, %rdx), %xmm0
	palignr $7, -16(%rdi, %rdx), D(%xmm0)
#if !defined USE_AS_STRCASECMP_L && !defined USE_AS_STRNCASECMP_L
	pcmpistri	$0x1a,(%rsi,%rdx), %xmm0
#else
	movdqa	(%rsi,%rdx), %xmm1
	TOLOWER (%xmm0, %xmm1)
	pcmpistri $0x1a, %xmm1, %xmm0
#endif
	jbe	LABEL(exit_use)
#if defined USE_AS_STRNCMP || defined USE_AS_STRNCASECMP_L
	sub	$16, %r11
	jbe	LABEL(strcmp_exitz)
#endif

	add	$16, %rdx
	add	$16, %r10
	jg	LABEL(nibble_ashr_7_use)

	movdqa	(%rdi, %rdx), %xmm0
	palignr $7, -16(%rdi, %rdx), D(%xmm0)
#if !defined USE_AS_STRCASECMP_L && !defined USE_AS_STRNCASECMP_L
	pcmpistri	$0x1a,(%rsi,%rdx), %xmm0
#else
	movdqa	(%rsi,%rdx), %xmm1
	TOLOWER (%xmm0, %xmm1)
	pcmpistri $0x1a, %xmm1, %xmm0
#endif
	jbe	LABEL(exit_use)
#if defined USE_AS_STRNCMP || defined USE_AS_STRNCASECMP_L
	sub	$16, %r11
	jbe	LABEL(strcmp_exitz)
#endif
	add	$16, %rdx
	jmp	LABEL(loop_ashr_7_use)

	.p2align 4
LABEL(nibble_ashr_7_use):
	sub	$0x1000, %r10
	movdqa	-16(%rdi, %rdx), %xmm0
	psrldq	$7, D(%xmm0)
	pcmpistri      $0x3a,%xmm0, %xmm0
#if defined USE_AS_STRNCMP || defined USE_AS_STRNCASECMP_L
	cmp	%r11, %rcx
	jae	LABEL(nibble_ashr_exit_use)
#endif
	cmp	$8, %ecx
	ja	LABEL(nibble_ashr_7_restart_use)

	jmp	LABEL(nibble_ashr_exit_use)

/*
 *  The following cases will be handled by ashr_8
 *  rcx(offset of rsi)  rax(offset of rdi)        relative offset	 corresponding case
 *        n(8~15)          n - 8		  7(15 +(n - 8) - n)         ashr_8
 */
	.p2align 4
LABEL(ashr_8):
	pslldq	$8, D(%xmm2)
	TOLOWER (%xmm1, %xmm2)
	pcmpeqb	%xmm1, D(%xmm2)
	psubb	%xmm0, D(%xmm2)
	pmovmskb %xmm2, %r9d
	shr	%cl, %edx
	shr	%cl, %r9d
	sub	%r9d, %edx
	jnz	LABEL(less32bytes)
	movdqa	(%rdi), %xmm3

	UPDATE_STRNCMP_COUNTER

	mov	$16, %rcx	/* index for loads */
	mov	$8, %r9d	/* byte position left over from less32bytes case */
	/*
	 * Setup %r10 value allows us to detect crossing a page boundary.
	 * When %r10 goes positive we have crossed a page boundary and
	 * need to do a nibble.
	 */
	lea	8(%rdi), %r10
	and	$0xfff, %r10	/* offset into 4K page */
	sub	$0x1000, %r10	/* subtract 4K pagesize */
	mov	%rcx, %rdx	/* only for offset of sse4 instruction loop*/

	.p2align 4
LABEL(loop_ashr_8_use):
	add	$16, %r10
	jg	LABEL(nibble_ashr_8_use)

LABEL(nibble_ashr_8_restart_use):
	movdqa	(%rdi, %rdx), %xmm0
	palignr $8, -16(%rdi, %rdx), D(%xmm0)
#if !defined USE_AS_STRCASECMP_L && !defined USE_AS_STRNCASECMP_L
	pcmpistri $0x1a, (%rsi,%rdx), %xmm0
#else
	movdqa	(%rsi,%rdx), %xmm1
	TOLOWER (%xmm0, %xmm1)
	pcmpistri $0x1a, %xmm1, %xmm0
#endif
	jbe	LABEL(exit_use)
#if defined USE_AS_STRNCMP || defined USE_AS_STRNCASECMP_L
	sub	$16, %r11
	jbe	LABEL(strcmp_exitz)
#endif

	add	$16, %rdx
	add	$16, %r10
	jg	LABEL(nibble_ashr_8_use)

	movdqa	(%rdi, %rdx), %xmm0
	palignr $8, -16(%rdi, %rdx), D(%xmm0)
#if !defined USE_AS_STRCASECMP_L && !defined USE_AS_STRNCASECMP_L
	pcmpistri $0x1a, (%rsi,%rdx), %xmm0
#else
	movdqa	(%rsi,%rdx), %xmm1
	TOLOWER (%xmm0, %xmm1)
	pcmpistri $0x1a, %xmm1, %xmm0
#endif
	jbe	LABEL(exit_use)
#if defined USE_AS_STRNCMP || defined USE_AS_STRNCASECMP_L
	sub	$16, %r11
	jbe	LABEL(strcmp_exitz)
#endif
	add	$16, %rdx
	jmp	LABEL(loop_ashr_8_use)

	.p2align 4
LABEL(nibble_ashr_8_use):
	sub	$0x1000, %r10
	movdqa	-16(%rdi, %rdx), %xmm0
	psrldq	$8, D(%xmm0)
	pcmpistri      $0x3a,%xmm0, %xmm0
#if defined USE_AS_STRNCMP || defined USE_AS_STRNCASECMP_L
	cmp	%r11, %rcx
	jae	LABEL(nibble_ashr_exit_use)
#endif
	cmp	$7, %ecx
	ja	LABEL(nibble_ashr_8_restart_use)

	jmp	LABEL(nibble_ashr_exit_use)

/*
 *  The following cases will be handled by ashr_9
 *  rcx(offset of rsi)  rax(offset of rdi)        relative offset	 corresponding case
 *        n(7~15)          n - 7		  8(15 +(n - 7) - n)         ashr_9
 */
	.p2align 4
LABEL(ashr_9):
	pslldq	$7, D(%xmm2)
	TOLOWER (%xmm1, %xmm2)
	pcmpeqb	%xmm1, D(%xmm2)
	psubb	%xmm0, D(%xmm2)
	pmovmskb %xmm2, %r9d
	shr	%cl, %edx
	shr	%cl, %r9d
	sub	%r9d, %edx
	jnz	LABEL(less32bytes)
	movdqa	(%rdi), %xmm3

	UPDATE_STRNCMP_COUNTER

	mov	$16, %rcx	/* index for loads */
	mov	$9, %r9d	/* byte position left over from less32bytes case */
	/*
	 * Setup %r10 value allows us to detect crossing a page boundary.
	 * When %r10 goes positive we have crossed a page boundary and
	 * need to do a nibble.
	 */
	lea	9(%rdi), %r10
	and	$0xfff, %r10	/* offset into 4K page */
	sub	$0x1000, %r10	/* subtract 4K pagesize */
	mov	%rcx, %rdx	/* only for offset of sse4 instruction loop*/

	.p2align 4
LABEL(loop_ashr_9_use):
	add	$16, %r10
	jg	LABEL(nibble_ashr_9_use)

LABEL(nibble_ashr_9_restart_use):
	movdqa	(%rdi, %rdx), %xmm0

	palignr $9, -16(%rdi, %rdx), D(%xmm0)
#if !defined USE_AS_STRCASECMP_L && !defined USE_AS_STRNCASECMP_L
	pcmpistri $0x1a, (%rsi,%rdx), %xmm0
#else
	movdqa	(%rsi,%rdx), %xmm1
	TOLOWER (%xmm0, %xmm1)
	pcmpistri $0x1a, %xmm1, %xmm0
#endif
	jbe	LABEL(exit_use)
#if defined USE_AS_STRNCMP || defined USE_AS_STRNCASECMP_L
	sub	$16, %r11
	jbe	LABEL(strcmp_exitz)
#endif

	add	$16, %rdx
	add	$16, %r10
	jg	LABEL(nibble_ashr_9_use)

	movdqa	(%rdi, %rdx), %xmm0
	palignr $9, -16(%rdi, %rdx), D(%xmm0)
#if !defined USE_AS_STRCASECMP_L && !defined USE_AS_STRNCASECMP_L
	pcmpistri $0x1a, (%rsi,%rdx), %xmm0
#else
	movdqa	(%rsi,%rdx), %xmm1
	TOLOWER (%xmm0, %xmm1)
	pcmpistri $0x1a, %xmm1, %xmm0
#endif
	jbe	LABEL(exit_use)
#if defined USE_AS_STRNCMP || defined USE_AS_STRNCASECMP_L
	sub	$16, %r11
	jbe	LABEL(strcmp_exitz)
#endif
	add	$16, %rdx
	jmp	LABEL(loop_ashr_9_use)

	.p2align 4
LABEL(nibble_ashr_9_use):
	sub	$0x1000, %r10
	movdqa	-16(%rdi, %rdx), %xmm0
	psrldq	$9, D(%xmm0)
	pcmpistri      $0x3a,%xmm0, %xmm0
#if defined USE_AS_STRNCMP || defined USE_AS_STRNCASECMP_L
	cmp	%r11, %rcx
	jae	LABEL(nibble_ashr_exit_use)
#endif
	cmp	$6, %ecx
	ja	LABEL(nibble_ashr_9_restart_use)

	jmp	LABEL(nibble_ashr_exit_use)

/*
 *  The following cases will be handled by ashr_10
 *  rcx(offset of rsi)  rax(offset of rdi)        relative offset	 corresponding case
 *        n(6~15)          n - 6		  9(15 +(n - 6) - n)         ashr_10
 */
	.p2align 4
LABEL(ashr_10):
	pslldq	$6, D(%xmm2)
	TOLOWER (%xmm1, %xmm2)
	pcmpeqb	%xmm1, D(%xmm2)
	psubb	%xmm0, D(%xmm2)
	pmovmskb %xmm2, %r9d
	shr	%cl, %edx
	shr	%cl, %r9d
	sub	%r9d, %edx
	jnz	LABEL(less32bytes)
	movdqa	(%rdi), %xmm3

	UPDATE_STRNCMP_COUNTER

	mov	$16, %rcx	/* index for loads */
	mov	$10, %r9d	/* byte position left over from less32bytes case */
	/*
	 * Setup %r10 value allows us to detect crossing a page boundary.
	 * When %r10 goes positive we have crossed a page boundary and
	 * need to do a nibble.
	 */
	lea	10(%rdi), %r10
	and	$0xfff, %r10	/* offset into 4K page */
	sub	$0x1000, %r10	/* subtract 4K pagesize */
	mov	%rcx, %rdx	/* only for offset of sse4 instruction loop*/

	.p2align 4
LABEL(loop_ashr_10_use):
	add	$16, %r10
	jg	LABEL(nibble_ashr_10_use)

LABEL(nibble_ashr_10_restart_use):
	movdqa	(%rdi, %rdx), %xmm0
	palignr $10, -16(%rdi, %rdx), D(%xmm0)
#if !defined USE_AS_STRCASECMP_L && !defined USE_AS_STRNCASECMP_L
	pcmpistri $0x1a, (%rsi,%rdx), %xmm0
#else
	movdqa	(%rsi,%rdx), %xmm1
	TOLOWER (%xmm0, %xmm1)
	pcmpistri $0x1a, %xmm1, %xmm0
#endif
	jbe	LABEL(exit_use)
#if defined USE_AS_STRNCMP || defined USE_AS_STRNCASECMP_L
	sub	$16, %r11
	jbe	LABEL(strcmp_exitz)
#endif

	add	$16, %rdx
	add	$16, %r10
	jg	LABEL(nibble_ashr_10_use)

	movdqa	(%rdi, %rdx), %xmm0
	palignr $10, -16(%rdi, %rdx), D(%xmm0)
#if !defined USE_AS_STRCASECMP_L && !defined USE_AS_STRNCASECMP_L
	pcmpistri $0x1a, (%rsi,%rdx), %xmm0
#else
	movdqa	(%rsi,%rdx), %xmm1
	TOLOWER (%xmm0, %xmm1)
	pcmpistri $0x1a, %xmm1, %xmm0
#endif
	jbe	LABEL(exit_use)
#if defined USE_AS_STRNCMP || defined USE_AS_STRNCASECMP_L
	sub	$16, %r11
	jbe	LABEL(strcmp_exitz)
#endif
	add	$16, %rdx
	jmp	LABEL(loop_ashr_10_use)

	.p2align 4
LABEL(nibble_ashr_10_use):
	sub	$0x1000, %r10
	movdqa	-16(%rdi, %rdx), %xmm0
	psrldq	$10, D(%xmm0)
	pcmpistri      $0x3a,%xmm0, %xmm0
#if defined USE_AS_STRNCMP || defined USE_AS_STRNCASECMP_L
	cmp	%r11, %rcx
	jae	LABEL(nibble_ashr_exit_use)
#endif
	cmp	$5, %ecx
	ja	LABEL(nibble_ashr_10_restart_use)

	jmp	LABEL(nibble_ashr_exit_use)

/*
 *  The following cases will be handled by ashr_11
 *  rcx(offset of rsi)  rax(offset of rdi)        relative offset	 corresponding case
 *        n(5~15)          n - 5		  10(15 +(n - 5) - n)         ashr_11
 */
	.p2align 4
LABEL(ashr_11):
	pslldq	$5, D(%xmm2)
	TOLOWER (%xmm1, %xmm2)
	pcmpeqb	%xmm1, D(%xmm2)
	psubb	%xmm0, D(%xmm2)
	pmovmskb %xmm2, %r9d
	shr	%cl, %edx
	shr	%cl, %r9d
	sub	%r9d, %edx
	jnz	LABEL(less32bytes)
	movdqa	(%rdi), %xmm3

	UPDATE_STRNCMP_COUNTER

	mov	$16, %rcx	/* index for loads */
	mov	$11, %r9d	/* byte position left over from less32bytes case */
	/*
	 * Setup %r10 value allows us to detect crossing a page boundary.
	 * When %r10 goes positive we have crossed a page boundary and
	 * need to do a nibble.
	 */
	lea	11(%rdi), %r10
	and	$0xfff, %r10	/* offset into 4K page */
	sub	$0x1000, %r10	/* subtract 4K pagesize */
	mov	%rcx, %rdx	/* only for offset of sse4 instruction loop*/

	.p2align 4
LABEL(loop_ashr_11_use):
	add	$16, %r10
	jg	LABEL(nibble_ashr_11_use)

LABEL(nibble_ashr_11_restart_use):
	movdqa	(%rdi, %rdx), %xmm0
	palignr $11, -16(%rdi, %rdx), D(%xmm0)
#if !defined USE_AS_STRCASECMP_L && !defined USE_AS_STRNCASECMP_L
	pcmpistri $0x1a, (%rsi,%rdx), %xmm0
#else
	movdqa	(%rsi,%rdx), %xmm1
	TOLOWER (%xmm0, %xmm1)
	pcmpistri $0x1a, %xmm1, %xmm0
#endif
	jbe	LABEL(exit_use)
#if defined USE_AS_STRNCMP || defined USE_AS_STRNCASECMP_L
	sub	$16, %r11
	jbe	LABEL(strcmp_exitz)
#endif

	add	$16, %rdx
	add	$16, %r10
	jg	LABEL(nibble_ashr_11_use)

	movdqa	(%rdi, %rdx), %xmm0
	palignr $11, -16(%rdi, %rdx), D(%xmm0)
#if !defined USE_AS_STRCASECMP_L && !defined USE_AS_STRNCASECMP_L
	pcmpistri $0x1a, (%rsi,%rdx), %xmm0
#else
	movdqa	(%rsi,%rdx), %xmm1
	TOLOWER (%xmm0, %xmm1)
	pcmpistri $0x1a, %xmm1, %xmm0
#endif
	jbe	LABEL(exit_use)
#if defined USE_AS_STRNCMP || defined USE_AS_STRNCASECMP_L
	sub	$16, %r11
	jbe	LABEL(strcmp_exitz)
#endif
	add	$16, %rdx
	jmp	LABEL(loop_ashr_11_use)

	.p2align 4
LABEL(nibble_ashr_11_use):
	sub	$0x1000, %r10
	movdqa	-16(%rdi, %rdx), %xmm0
	psrldq	$11, D(%xmm0)
	pcmpistri      $0x3a,%xmm0, %xmm0
#if defined USE_AS_STRNCMP || defined USE_AS_STRNCASECMP_L
	cmp	%r11, %rcx
	jae	LABEL(nibble_ashr_exit_use)
#endif
	cmp	$4, %ecx
	ja	LABEL(nibble_ashr_11_restart_use)

	jmp	LABEL(nibble_ashr_exit_use)

/*
 *  The following cases will be handled by ashr_12
 *  rcx(offset of rsi)  rax(offset of rdi)        relative offset	 corresponding case
 *        n(4~15)          n - 4		  11(15 +(n - 4) - n)         ashr_12
 */
	.p2align 4
LABEL(ashr_12):
	pslldq	$4, D(%xmm2)
	TOLOWER (%xmm1, %xmm2)
	pcmpeqb	%xmm1, D(%xmm2)
	psubb	%xmm0, D(%xmm2)
	pmovmskb %xmm2, %r9d
	shr	%cl, %edx
	shr	%cl, %r9d
	sub	%r9d, %edx
	jnz	LABEL(less32bytes)
	movdqa	(%rdi), %xmm3

	UPDATE_STRNCMP_COUNTER

	mov	$16, %rcx	/* index for loads */
	mov	$12, %r9d	/* byte position left over from less32bytes case */
	/*
	 * Setup %r10 value allows us to detect crossing a page boundary.
	 * When %r10 goes positive we have crossed a page boundary and
	 * need to do a nibble.
	 */
	lea	12(%rdi), %r10
	and	$0xfff, %r10	/* offset into 4K page */
	sub	$0x1000, %r10	/* subtract 4K pagesize */
	mov	%rcx, %rdx	/* only for offset of sse4 instruction loop*/

	.p2align 4
LABEL(loop_ashr_12_use):
	add	$16, %r10
	jg	LABEL(nibble_ashr_12_use)

LABEL(nibble_ashr_12_restart_use):
	movdqa	(%rdi, %rdx), %xmm0
	palignr $12, -16(%rdi, %rdx), D(%xmm0)
#if !defined USE_AS_STRCASECMP_L && !defined USE_AS_STRNCASECMP_L
	pcmpistri $0x1a, (%rsi,%rdx), %xmm0
#else
	movdqa	(%rsi,%rdx), %xmm1
	TOLOWER (%xmm0, %xmm1)
	pcmpistri $0x1a, %xmm1, %xmm0
#endif
	jbe	LABEL(exit_use)
#if defined USE_AS_STRNCMP || defined USE_AS_STRNCASECMP_L
	sub	$16, %r11
	jbe	LABEL(strcmp_exitz)
#endif

	add	$16, %rdx
	add	$16, %r10
	jg	LABEL(nibble_ashr_12_use)

	movdqa	(%rdi, %rdx), %xmm0
	palignr $12, -16(%rdi, %rdx), D(%xmm0)
#if !defined USE_AS_STRCASECMP_L && !defined USE_AS_STRNCASECMP_L
	pcmpistri $0x1a, (%rsi,%rdx), %xmm0
#else
	movdqa	(%rsi,%rdx), %xmm1
	TOLOWER (%xmm0, %xmm1)
	pcmpistri $0x1a, %xmm1, %xmm0
#endif
	jbe	LABEL(exit_use)
#if defined USE_AS_STRNCMP || defined USE_AS_STRNCASECMP_L
	sub	$16, %r11
	jbe	LABEL(strcmp_exitz)
#endif
	add	$16, %rdx
	jmp	LABEL(loop_ashr_12_use)

	.p2align 4
LABEL(nibble_ashr_12_use):
	sub	$0x1000, %r10
	movdqa	-16(%rdi, %rdx), %xmm0
	psrldq	$12, D(%xmm0)
	pcmpistri      $0x3a,%xmm0, %xmm0
#if defined USE_AS_STRNCMP || defined USE_AS_STRNCASECMP_L
	cmp	%r11, %rcx
	jae	LABEL(nibble_ashr_exit_use)
#endif
	cmp	$3, %ecx
	ja	LABEL(nibble_ashr_12_restart_use)

	jmp	LABEL(nibble_ashr_exit_use)

/*
 *  The following cases will be handled by ashr_13
 *  rcx(offset of rsi)  rax(offset of rdi)        relative offset	 corresponding case
 *        n(3~15)          n - 3		  12(15 +(n - 3) - n)         ashr_13
 */
	.p2align 4
LABEL(ashr_13):
	pslldq	$3, D(%xmm2)
	TOLOWER (%xmm1, %xmm2)
	pcmpeqb	%xmm1, D(%xmm2)
	psubb	%xmm0, D(%xmm2)
	pmovmskb %xmm2, %r9d
	shr	%cl, %edx
	shr	%cl, %r9d
	sub	%r9d, %edx
	jnz	LABEL(less32bytes)
	movdqa	(%rdi), %xmm3

	UPDATE_STRNCMP_COUNTER

	mov	$16, %rcx	/* index for loads */
	mov	$13, %r9d	/* byte position left over from less32bytes case */
	/*
	 * Setup %r10 value allows us to detect crossing a page boundary.
	 * When %r10 goes positive we have crossed a page boundary and
	 * need to do a nibble.
	 */
	lea	13(%rdi), %r10
	and	$0xfff, %r10	/* offset into 4K page */
	sub	$0x1000, %r10	/* subtract 4K pagesize */

	mov	%rcx, %rdx	/* only for offset of sse4 instruction loop*/

	.p2align 4
LABEL(loop_ashr_13_use):
	add	$16, %r10
	jg	LABEL(nibble_ashr_13_use)

LABEL(nibble_ashr_13_restart_use):
	movdqa	(%rdi, %rdx), %xmm0
	palignr $13, -16(%rdi, %rdx), D(%xmm0)
#if !defined USE_AS_STRCASECMP_L && !defined USE_AS_STRNCASECMP_L
	pcmpistri $0x1a, (%rsi,%rdx), %xmm0
#else
	movdqa	(%rsi,%rdx), %xmm1
	TOLOWER (%xmm0, %xmm1)
	pcmpistri $0x1a, %xmm1, %xmm0
#endif
	jbe	LABEL(exit_use)
#if defined USE_AS_STRNCMP || defined USE_AS_STRNCASECMP_L
	sub	$16, %r11
	jbe	LABEL(strcmp_exitz)
#endif

	add	$16, %rdx
	add	$16, %r10
	jg	LABEL(nibble_ashr_13_use)

	movdqa	(%rdi, %rdx), %xmm0
	palignr $13, -16(%rdi, %rdx), D(%xmm0)
#if !defined USE_AS_STRCASECMP_L && !defined USE_AS_STRNCASECMP_L
	pcmpistri $0x1a, (%rsi,%rdx), %xmm0
#else
	movdqa	(%rsi,%rdx), %xmm1
	TOLOWER (%xmm0, %xmm1)
	pcmpistri $0x1a, %xmm1, %xmm0
#endif
	jbe	LABEL(exit_use)
#if defined USE_AS_STRNCMP || defined USE_AS_STRNCASECMP_L
	sub	$16, %r11
	jbe	LABEL(strcmp_exitz)
#endif
	add	$16, %rdx
	jmp	LABEL(loop_ashr_13_use)

	.p2align 4
LABEL(nibble_ashr_13_use):
	sub	$0x1000, %r10
	movdqa	-16(%rdi, %rdx), %xmm0
	psrldq	$13, D(%xmm0)
	pcmpistri      $0x3a,%xmm0, %xmm0
#if defined USE_AS_STRNCMP || defined USE_AS_STRNCASECMP_L
	cmp	%r11, %rcx
	jae	LABEL(nibble_ashr_exit_use)
#endif
	cmp	$2, %ecx
	ja	LABEL(nibble_ashr_13_restart_use)

	jmp	LABEL(nibble_ashr_exit_use)

/*
 *  The following cases will be handled by ashr_14
 *  rcx(offset of rsi)  rax(offset of rdi)        relative offset	 corresponding case
 *        n(2~15)          n - 2		  13(15 +(n - 2) - n)         ashr_14
 */
	.p2align 4
LABEL(ashr_14):
	pslldq  $2, D(%xmm2)
	TOLOWER (%xmm1, %xmm2)
	pcmpeqb	%xmm1, D(%xmm2)
	psubb	%xmm0, D(%xmm2)
	pmovmskb %xmm2, %r9d
	shr	%cl, %edx
	shr	%cl, %r9d
	sub	%r9d, %edx
	jnz	LABEL(less32bytes)
	movdqa	(%rdi), %xmm3

	UPDATE_STRNCMP_COUNTER

	mov	$16, %rcx	/* index for loads */
	mov	$14, %r9d	/* byte position left over from less32bytes case */
	/*
	 * Setup %r10 value allows us to detect crossing a page boundary.
	 * When %r10 goes positive we have crossed a page boundary and
	 * need to do a nibble.
	 */
	lea	14(%rdi), %r10
	and	$0xfff, %r10	/* offset into 4K page */
	sub	$0x1000, %r10	/* subtract 4K pagesize */

	mov	%rcx, %rdx	/* only for offset of sse4 instruction loop*/

	.p2align 4
LABEL(loop_ashr_14_use):
	add	$16, %r10
	jg	LABEL(nibble_ashr_14_use)

LABEL(nibble_ashr_14_restart_use):
	movdqa	(%rdi, %rdx), %xmm0
	palignr $14, -16(%rdi, %rdx), D(%xmm0)
#if !defined USE_AS_STRCASECMP_L && !defined USE_AS_STRNCASECMP_L
	pcmpistri $0x1a, (%rsi,%rdx), %xmm0
#else
	movdqa	(%rsi,%rdx), %xmm1
	TOLOWER (%xmm0, %xmm1)
	pcmpistri $0x1a, %xmm1, %xmm0
#endif
	jbe	LABEL(exit_use)
#if defined USE_AS_STRNCMP || defined USE_AS_STRNCASECMP_L
	sub	$16, %r11
	jbe	LABEL(strcmp_exitz)
#endif

	add	$16, %rdx
	add	$16, %r10
	jg	LABEL(nibble_ashr_14_use)

	movdqa	(%rdi, %rdx), %xmm0
	palignr $14, -16(%rdi, %rdx), D(%xmm0)
#if !defined USE_AS_STRCASECMP_L && !defined USE_AS_STRNCASECMP_L
	pcmpistri $0x1a, (%rsi,%rdx), %xmm0
#else
	movdqa	(%rsi,%rdx), %xmm1
	TOLOWER (%xmm0, %xmm1)
	pcmpistri $0x1a, %xmm1, %xmm0
#endif
	jbe	LABEL(exit_use)
#if defined USE_AS_STRNCMP || defined USE_AS_STRNCASECMP_L
	sub	$16, %r11
	jbe	LABEL(strcmp_exitz)
#endif
	add	$16, %rdx
	jmp	LABEL(loop_ashr_14_use)

	.p2align 4
LABEL(nibble_ashr_14_use):
	sub	$0x1000, %r10
	movdqa	-16(%rdi, %rdx), %xmm0
	psrldq	$14, D(%xmm0)
	pcmpistri      $0x3a,%xmm0, %xmm0
#if defined USE_AS_STRNCMP || defined USE_AS_STRNCASECMP_L
	cmp	%r11, %rcx
	jae	LABEL(nibble_ashr_exit_use)
#endif
	cmp	$1, %ecx
	ja	LABEL(nibble_ashr_14_restart_use)

	jmp	LABEL(nibble_ashr_exit_use)

/*
 *  The following cases will be handled by ashr_15
 *  rcx(offset of rsi)  rax(offset of rdi)        relative offset	 corresponding case
 *        n(1~15)          n - 1		  14(15 +(n - 1) - n)         ashr_15
 */
	.p2align 4
LABEL(ashr_15):
	pslldq	$1, D(%xmm2)
	TOLOWER (%xmm1, %xmm2)
	pcmpeqb	%xmm1, D(%xmm2)
	psubb	%xmm0, D(%xmm2)
	pmovmskb %xmm2, %r9d
	shr	%cl, %edx
	shr	%cl, %r9d
	sub	%r9d, %edx
	jnz	LABEL(less32bytes)

	movdqa	(%rdi), %xmm3

	UPDATE_STRNCMP_COUNTER

	mov	$16, %rcx	/* index for loads */
	mov	$15, %r9d	/* byte position left over from less32bytes case */
	/*
	 * Setup %r10 value allows us to detect crossing a page boundary.
	 * When %r10 goes positive we have crossed a page boundary and
	 * need to do a nibble.
	 */
	lea	15(%rdi), %r10
	and	$0xfff, %r10	/* offset into 4K page */

	sub	$0x1000, %r10	/* subtract 4K pagesize */

	mov	%rcx, %rdx	/* only for offset of sse4 instruction loop*/

	.p2align 4
LABEL(loop_ashr_15_use):
	add	$16, %r10
	jg	LABEL(nibble_ashr_15_use)

LABEL(nibble_ashr_15_restart_use):
	movdqa	(%rdi, %rdx), %xmm0
	palignr $15, -16(%rdi, %rdx), D(%xmm0)
#if !defined USE_AS_STRCASECMP_L && !defined USE_AS_STRNCASECMP_L
	pcmpistri $0x1a, (%rsi,%rdx), %xmm0
#else
	movdqa	(%rsi,%rdx), %xmm1
	TOLOWER (%xmm0, %xmm1)
	pcmpistri $0x1a, %xmm1, %xmm0
#endif
	jbe	LABEL(exit_use)
#if defined USE_AS_STRNCMP || defined USE_AS_STRNCASECMP_L
	sub	$16, %r11
	jbe	LABEL(strcmp_exitz)
#endif

	add	$16, %rdx
	add	$16, %r10
	jg	LABEL(nibble_ashr_15_use)

	movdqa	(%rdi, %rdx), %xmm0
	palignr $15, -16(%rdi, %rdx), D(%xmm0)
#if !defined USE_AS_STRCASECMP_L && !defined USE_AS_STRNCASECMP_L
	pcmpistri $0x1a, (%rsi,%rdx), %xmm0
#else
	movdqa	(%rsi,%rdx), %xmm1
	TOLOWER (%xmm0, %xmm1)
	pcmpistri $0x1a, %xmm1, %xmm0
#endif
	jbe	LABEL(exit_use)
#if defined USE_AS_STRNCMP || defined USE_AS_STRNCASECMP_L
	sub	$16, %r11
	jbe	LABEL(strcmp_exitz)
#endif
	add	$16, %rdx
	jmp	LABEL(loop_ashr_15_use)

	.p2align 4
LABEL(nibble_ashr_15_use):
	sub	$0x1000, %r10
	movdqa	-16(%rdi, %rdx), %xmm0
	psrldq	$15, D(%xmm0)
	pcmpistri      $0x3a,%xmm0, %xmm0
#if defined USE_AS_STRNCMP || defined USE_AS_STRNCASECMP_L
	cmp	%r11, %rcx
	jae	LABEL(nibble_ashr_exit_use)
#endif
	cmp	$0, %ecx
	ja	LABEL(nibble_ashr_15_restart_use)

LABEL(nibble_ashr_exit_use):
#if !defined USE_AS_STRCASECMP_L && !defined USE_AS_STRNCASECMP_L
	pcmpistri      $0x1a,(%rsi,%rdx), %xmm0
#else
	movdqa	(%rsi,%rdx), %xmm1
	TOLOWER (%xmm0, %xmm1)
	pcmpistri $0x1a, %xmm1, %xmm0
#endif
	.p2align 4
LABEL(exit_use):
	jnc	LABEL(strcmp_exitz)
#if defined USE_AS_STRNCMP || defined USE_AS_STRNCASECMP_L
	sub	%rcx, %r11
	jbe	LABEL(strcmp_exitz)
#endif
	add	%rcx, %rdx
	lea	-16(%rdi, %r9), %rdi
	movzbl	(%rdi, %rdx), %eax
	movzbl	(%rsi, %rdx), %edx
	test	%r8d, %r8d
	jz	LABEL(ret_use)
	xchg	%eax, %edx
LABEL(ret_use):
#if defined USE_AS_STRCASECMP_L || defined USE_AS_STRNCASECMP_L
	leaq	_nl_C_LC_CTYPE_tolower+128*4(%rip), %rcx
	movl	(%rcx,%rdx,4), %edx
	movl	(%rcx,%rax,4), %eax
#endif

	sub	%edx, %eax
	ret

LABEL(less32bytes):
	lea	(%rdi, %rax), %rdi	/* locate the exact address for first operand(rdi) */
	lea	(%rsi, %rcx), %rsi	/* locate the exact address for second operand(rsi) */
	test	%r8d, %r8d
	jz	LABEL(ret)
	xchg	%rsi, %rdi		/* recover original order according to flag(%r8d) */

	.p2align 4
LABEL(ret):
LABEL(less16bytes):
	bsf	%rdx, %rdx		/* find and store bit index in %rdx */

#if defined USE_AS_STRNCMP || defined USE_AS_STRNCASECMP_L
	sub	%rdx, %r11
	jbe	LABEL(strcmp_exitz)
#endif
	movzbl	(%rsi, %rdx), %ecx
	movzbl	(%rdi, %rdx), %eax

#if defined USE_AS_STRCASECMP_L || defined USE_AS_STRNCASECMP_L
	leaq	_nl_C_LC_CTYPE_tolower+128*4(%rip), %rdx
	movl	(%rdx,%rcx,4), %ecx
	movl	(%rdx,%rax,4), %eax
#endif

	sub	%ecx, %eax
	ret

LABEL(strcmp_exitz):
	xor	%eax, %eax
	ret

	.p2align 4
	// XXX Same as code above
LABEL(Byte0):
	movzbl	(%rsi),%ecx
	movzbl	(%rdi),%eax

#if defined USE_AS_STRCASECMP_L || defined USE_AS_STRNCASECMP_L
	leaq	_nl_C_LC_CTYPE_tolower+128*4(%rip), %rdx
	movl	(%rdx,%rcx,4), %ecx
	movl	(%rdx,%rax,4), %eax
#endif

	sub	%ecx, %eax
	ret
	cfi_endproc
	.size	STRCMP_SSE42, .-STRCMP_SSE42

#undef UCLOW_reg
#undef UCHIGH_reg
#undef LCQWORD_reg
#undef TOLOWER

	/* Put all SSE 4.2 functions together.  */
	.section .rodata.SECTION,"a",@progbits
	.p2align 3
LABEL(unaligned_table):
	.int	LABEL(ashr_1) - LABEL(unaligned_table)
	.int	LABEL(ashr_2) - LABEL(unaligned_table)
	.int	LABEL(ashr_3) - LABEL(unaligned_table)
	.int	LABEL(ashr_4) - LABEL(unaligned_table)
	.int	LABEL(ashr_5) - LABEL(unaligned_table)
	.int	LABEL(ashr_6) - LABEL(unaligned_table)
	.int	LABEL(ashr_7) - LABEL(unaligned_table)
	.int	LABEL(ashr_8) - LABEL(unaligned_table)
	.int	LABEL(ashr_9) - LABEL(unaligned_table)
	.int	LABEL(ashr_10) - LABEL(unaligned_table)
	.int	LABEL(ashr_11) - LABEL(unaligned_table)
	.int	LABEL(ashr_12) - LABEL(unaligned_table)
	.int	LABEL(ashr_13) - LABEL(unaligned_table)
	.int	LABEL(ashr_14) - LABEL(unaligned_table)
	.int	LABEL(ashr_15) - LABEL(unaligned_table)
	.int	LABEL(ashr_0) - LABEL(unaligned_table)

#undef LABEL
#undef GLABEL
#undef SECTION
#undef movdqa
#undef movdqu
#undef pmovmskb
#undef pcmpistri
#undef psubb
#undef pcmpeqb
#undef psrldq
#undef pslldq
#undef palignr
#undef pxor
#undef D
