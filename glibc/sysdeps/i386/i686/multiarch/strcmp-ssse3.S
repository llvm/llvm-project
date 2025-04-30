/* strcmp with SSSE3
   Copyright (C) 2010-2021 Free Software Foundation, Inc.
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

#if IS_IN (libc)

#include <sysdep.h>
#include "asm-syntax.h"

#define CFI_PUSH(REG)						\
  cfi_adjust_cfa_offset (4);					\
  cfi_rel_offset (REG, 0)

#define CFI_POP(REG)						\
  cfi_adjust_cfa_offset (-4);					\
  cfi_restore (REG)

#define PUSH(REG)	pushl REG; CFI_PUSH (REG)
#define POP(REG)	popl REG; CFI_POP (REG)

#ifdef USE_AS_STRNCMP
# ifndef STRCMP
#  define STRCMP	__strncmp_ssse3
# endif
# define STR1		8
# define STR2		STR1+4
# define CNT		STR2+4
# define RETURN		POP (REM); ret; .p2align 4; CFI_PUSH (REM)
# define UPDATE_STRNCMP_COUNTER				\
	/* calculate left number to compare */		\
	mov	$16, %esi;				\
	sub	%ecx, %esi;				\
	cmp	%esi, REM;				\
	jbe	L(more8byteseq);			\
	sub	%esi, REM
# define FLAGS		%ebx
# define REM		%ebp
#elif defined USE_AS_STRCASECMP_L
# include "locale-defines.h"
# ifndef STRCMP
#  define STRCMP	__strcasecmp_l_ssse3
# endif
# ifdef PIC
#  define STR1		8
# else
#  define STR1		4
# endif
# define STR2		STR1+4
# define LOCALE		12	/* Loaded before the adjustment.  */
# ifdef PIC
#  define RETURN	POP (%ebx); ret; .p2align 4; CFI_PUSH (%ebx)
# else
#  define RETURN	ret; .p2align 4
# endif
# define UPDATE_STRNCMP_COUNTER
# define FLAGS		(%esp)
# define NONASCII	__strcasecmp_nonascii
#elif defined USE_AS_STRNCASECMP_L
# include "locale-defines.h"
# ifndef STRCMP
#  define STRCMP	__strncasecmp_l_ssse3
# endif
# ifdef PIC
#  define STR1		12
# else
#  define STR1		8
# endif
# define STR2		STR1+4
# define CNT		STR2+4
# define LOCALE		16	/* Loaded before the adjustment.  */
# ifdef PIC
#  define RETURN	POP (REM); POP (%ebx); ret; \
			.p2align 4; CFI_PUSH (%ebx); CFI_PUSH (REM)
# else
#  define RETURN	POP (REM); ret; .p2align 4; CFI_PUSH (REM)
# endif
# define UPDATE_STRNCMP_COUNTER				\
	/* calculate left number to compare */		\
	mov	$16, %esi;				\
	sub	%ecx, %esi;				\
	cmp	%esi, REM;				\
	jbe	L(more8byteseq);			\
	sub	%esi, REM
# define FLAGS		(%esp)
# define REM		%ebp
# define NONASCII	__strncasecmp_nonascii
#else
# ifndef STRCMP
#  define STRCMP	__strcmp_ssse3
# endif
# define STR1		4
# define STR2		STR1+4
# define RETURN		ret; .p2align 4
# define UPDATE_STRNCMP_COUNTER
# define FLAGS		%ebx
#endif

	.section .text.ssse3,"ax",@progbits

#ifdef USE_AS_STRCASECMP_L
ENTRY (__strcasecmp_ssse3)
# ifdef PIC
	PUSH	(%ebx)
	LOAD_PIC_REG(bx)
	movl	__libc_tsd_LOCALE@GOTNTPOFF(%ebx), %eax
	movl	%gs:(%eax), %eax
# else
	movl	%gs:__libc_tsd_LOCALE@NTPOFF, %eax
# endif
# if LOCALE_T___LOCALES != 0 || LC_CTYPE != 0
	movl	LOCALE_T___LOCALES+LC_CTYPE*4(%eax), %eax
# else
	movl	(%eax), %eax
# endif
	testl	$1, LOCALE_DATA_VALUES+_NL_CTYPE_NONASCII_CASE*SIZEOF_VALUES(%eax)
# ifdef PIC
	je	L(ascii)
	POP	(%ebx)
	jmp	__strcasecmp_nonascii
# else
	jne	__strcasecmp_nonascii
	jmp	L(ascii)
# endif
END (__strcasecmp_ssse3)
#endif

#ifdef USE_AS_STRNCASECMP_L
ENTRY (__strncasecmp_ssse3)
# ifdef PIC
	PUSH	(%ebx)
	LOAD_PIC_REG(bx)
	movl	__libc_tsd_LOCALE@GOTNTPOFF(%ebx), %eax
	movl	%gs:(%eax), %eax
# else
	movl	%gs:__libc_tsd_LOCALE@NTPOFF, %eax
# endif
# if LOCALE_T___LOCALES != 0 || LC_CTYPE != 0
	movl	LOCALE_T___LOCALES+LC_CTYPE*4(%eax), %eax
# else
	movl	(%eax), %eax
# endif
	testl	$1, LOCALE_DATA_VALUES+_NL_CTYPE_NONASCII_CASE*SIZEOF_VALUES(%eax)
# ifdef PIC
	je	L(ascii)
	POP	(%ebx)
	jmp	__strncasecmp_nonascii
# else
	jne	__strncasecmp_nonascii
	jmp	L(ascii)
# endif
END (__strncasecmp_ssse3)
#endif

ENTRY (STRCMP)
#if defined USE_AS_STRCASECMP_L || defined USE_AS_STRNCASECMP_L
	movl	LOCALE(%esp), %eax
# if LOCALE_T___LOCALES != 0 || LC_CTYPE != 0
	movl	LOCALE_T___LOCALES+LC_CTYPE*4(%eax), %eax
# else
	movl	(%eax), %eax
# endif
	testl	$1, LOCALE_DATA_VALUES+_NL_CTYPE_NONASCII_CASE*SIZEOF_VALUES(%eax)
	jne	NONASCII

# ifdef PIC
	PUSH	(%ebx)
	LOAD_PIC_REG(bx)
# endif
L(ascii):
	.section .rodata.cst16,"aM",@progbits,16
	.align 16
.Lbelowupper:
	.quad	0x4040404040404040
	.quad	0x4040404040404040
.Ltopupper:
	.quad	0x5b5b5b5b5b5b5b5b
	.quad	0x5b5b5b5b5b5b5b5b
.Ltouppermask:
	.quad	0x2020202020202020
	.quad	0x2020202020202020
	.previous

# ifdef PIC
#  define UCLOW_reg .Lbelowupper@GOTOFF(%ebx)
#  define UCHIGH_reg .Ltopupper@GOTOFF(%ebx)
#  define LCQWORD_reg .Ltouppermask@GOTOFF(%ebx)
# else
#  define UCLOW_reg .Lbelowupper
#  define UCHIGH_reg .Ltopupper
#  define LCQWORD_reg .Ltouppermask
# endif
#endif

#if defined USE_AS_STRNCMP || defined USE_AS_STRNCASECMP_L
	PUSH	(REM)
#endif

	movl	STR1(%esp), %edx
	movl	STR2(%esp), %eax
#if defined USE_AS_STRNCMP || defined USE_AS_STRNCASECMP_L
	movl	CNT(%esp), REM
	cmp	$16, REM
	jb	L(less16bytes_sncmp)
#elif !defined USE_AS_STRCASECMP_L
	movzbl	(%eax), %ecx
	cmpb	%cl, (%edx)
	jne	L(neq)
	cmpl	$0, %ecx
	je	L(eq)

	movzbl	1(%eax), %ecx
	cmpb	%cl, 1(%edx)
	jne	L(neq)
	cmpl	$0, %ecx
	je	L(eq)

	movzbl	2(%eax), %ecx
	cmpb	%cl, 2(%edx)
	jne	L(neq)
	cmpl	$0, %ecx
	je	L(eq)

	movzbl	3(%eax), %ecx
	cmpb	%cl, 3(%edx)
	jne	L(neq)
	cmpl	$0, %ecx
	je	L(eq)

	movzbl	4(%eax), %ecx
	cmpb	%cl, 4(%edx)
	jne	L(neq)
	cmpl	$0, %ecx
	je	L(eq)

	movzbl	5(%eax), %ecx
	cmpb	%cl, 5(%edx)
	jne	L(neq)
	cmpl	$0, %ecx
	je	L(eq)

	movzbl	6(%eax), %ecx
	cmpb	%cl, 6(%edx)
	jne	L(neq)
	cmpl	$0, %ecx
	je	L(eq)

	movzbl	7(%eax), %ecx
	cmpb	%cl, 7(%edx)
	jne	L(neq)
	cmpl	$0, %ecx
	je	L(eq)

	add	$8, %edx
	add	$8, %eax
#endif
	movl	%edx, %ecx
	and	$0xfff, %ecx
	cmp	$0xff0, %ecx
	ja	L(crosspage)
	mov	%eax, %ecx
	and	$0xfff, %ecx
	cmp	$0xff0, %ecx
	ja	L(crosspage)
	pxor	%xmm0, %xmm0
	movlpd	(%eax), %xmm1
	movlpd	(%edx), %xmm2
	movhpd	8(%eax), %xmm1
	movhpd	8(%edx), %xmm2
#if defined USE_AS_STRCASECMP_L || defined USE_AS_STRNCASECMP_L
# define TOLOWER(reg1, reg2) \
	movdqa	reg1, %xmm5;					\
	movdqa	reg2, %xmm7;					\
	movdqa	UCHIGH_reg, %xmm6;				\
	pcmpgtb	UCLOW_reg, %xmm5;				\
	pcmpgtb	UCLOW_reg, %xmm7;				\
	pcmpgtb	reg1, %xmm6;					\
	pand	%xmm6, %xmm5;					\
	movdqa	UCHIGH_reg, %xmm6;				\
	pcmpgtb	reg2, %xmm6;					\
	pand	%xmm6, %xmm7;					\
	pand	LCQWORD_reg, %xmm5;				\
	por	%xmm5, reg1;					\
	pand	LCQWORD_reg, %xmm7;				\
	por	%xmm7, reg2
	TOLOWER (%xmm1, %xmm2)
#else
# define TOLOWER(reg1, reg2)
#endif
	pcmpeqb	%xmm1, %xmm0
	pcmpeqb	%xmm2, %xmm1
	psubb	%xmm0, %xmm1
	pmovmskb %xmm1, %ecx
	sub	$0xffff, %ecx
	jnz	L(less16bytes)
#if defined USE_AS_STRNCMP || defined USE_AS_STRNCASECMP_L
	cmp	$16, REM
	lea	-16(REM), REM
	jbe	L(eq)
#endif
	add	$16, %eax
	add	$16, %edx

L(crosspage):

#if !defined USE_AS_STRCASECMP_L && !defined USE_AS_STRNCASECMP_L
	PUSH	(FLAGS)
#endif
	PUSH	(%edi)
	PUSH	(%esi)
#if defined USE_AS_STRCASECMP_L || defined USE_AS_STRNCASECMP_L
	pushl	$0
	cfi_adjust_cfa_offset (4)
#endif
#if defined USE_AS_STRNCMP || defined USE_AS_STRNCASECMP_L
	cfi_remember_state
#endif

	movl	%edx, %edi
	movl	%eax, %ecx
	and	$0xf, %ecx
	and	$0xf, %edi
	xor	%ecx, %eax
	xor	%edi, %edx
#if !defined USE_AS_STRCASECMP_L && !defined USE_AS_STRNCASECMP_L
	xor	FLAGS, FLAGS
#endif
	cmp	%edi, %ecx
	je	L(ashr_0)
	ja	L(bigger)
	orl	$0x20, FLAGS
	xchg	%edx, %eax
	xchg	%ecx, %edi
L(bigger):
	lea	15(%edi), %edi
	sub	%ecx, %edi
	cmp	$8, %edi
	jle	L(ashr_less_8)
	cmp	$14, %edi
	je	L(ashr_15)
	cmp	$13, %edi
	je	L(ashr_14)
	cmp	$12, %edi
	je	L(ashr_13)
	cmp	$11, %edi
	je	L(ashr_12)
	cmp	$10, %edi
	je	L(ashr_11)
	cmp	$9, %edi
	je	L(ashr_10)
L(ashr_less_8):
	je	L(ashr_9)
	cmp	$7, %edi
	je	L(ashr_8)
	cmp	$6, %edi
	je	L(ashr_7)
	cmp	$5, %edi
	je	L(ashr_6)
	cmp	$4, %edi
	je	L(ashr_5)
	cmp	$3, %edi
	je	L(ashr_4)
	cmp	$2, %edi
	je	L(ashr_3)
	cmp	$1, %edi
	je	L(ashr_2)
	cmp	$0, %edi
	je	L(ashr_1)

/*
 * The following cases will be handled by ashr_0
 *  ecx(offset of esi)  eax(offset of edi)  relative offset  corresponding case
 *        n(0~15)            n(0~15)           15(15+ n-n)         ashr_0
 */
	.p2align 4
L(ashr_0):
	mov	$0xffff, %esi
	movdqa	(%eax), %xmm1
	pxor	%xmm0, %xmm0
	pcmpeqb	%xmm1, %xmm0
#if defined USE_AS_STRCASECMP_L || defined USE_AS_STRNCASECMP_L
	movdqa	(%edx), %xmm2
	TOLOWER (%xmm1, %xmm2)
	pcmpeqb	%xmm2, %xmm1
#else
	pcmpeqb	(%edx), %xmm1
#endif
	psubb	%xmm0, %xmm1
	pmovmskb %xmm1, %edi
	shr	%cl, %esi
	shr	%cl, %edi
	sub	%edi, %esi
	mov	%ecx, %edi
	jne	L(less32bytes)
	UPDATE_STRNCMP_COUNTER
	movl	$0x10, FLAGS
	mov	$0x10, %ecx
	pxor	%xmm0, %xmm0
	.p2align 4
L(loop_ashr_0):
	movdqa	(%eax, %ecx), %xmm1
#if defined USE_AS_STRCASECMP_L || defined USE_AS_STRNCASECMP_L
	movdqa	(%edx, %ecx), %xmm2
	TOLOWER (%xmm1, %xmm2)

	pcmpeqb	%xmm1, %xmm0
	pcmpeqb	%xmm2, %xmm1
#else
	pcmpeqb	%xmm1, %xmm0
	pcmpeqb	(%edx, %ecx), %xmm1
#endif
	psubb	%xmm0, %xmm1
	pmovmskb %xmm1, %esi
	sub	$0xffff, %esi
	jnz	L(exit)
#if defined USE_AS_STRNCMP || defined USE_AS_STRNCASECMP_L
	cmp	$16, REM
	lea	-16(REM), REM
	jbe	L(more8byteseq)
#endif
	add	$16, %ecx
	jmp	L(loop_ashr_0)

/*
 * The following cases will be handled by ashr_1
 * ecx(offset of esi)  eax(offset of edi)   relative offset	corresponding case
 *        n(15)            n -15            0(15 +(n-15) - n)         ashr_1
 */
	.p2align 4
L(ashr_1):
	mov	$0xffff, %esi
	pxor	%xmm0, %xmm0
	movdqa	(%edx), %xmm2
	movdqa	(%eax), %xmm1
	pcmpeqb	%xmm1, %xmm0
	pslldq	$15, %xmm2
	TOLOWER (%xmm1, %xmm2)
	pcmpeqb	%xmm1, %xmm2
	psubb	%xmm0, %xmm2
	pmovmskb %xmm2, %edi
	shr	%cl, %esi
	shr	%cl, %edi
	sub	%edi, %esi
	lea	-15(%ecx), %edi
	jnz	L(less32bytes)

	UPDATE_STRNCMP_COUNTER

	movdqa	(%edx), %xmm3
	pxor	%xmm0, %xmm0
	mov	$16, %ecx
	orl	$1, FLAGS
	lea	1(%edx), %edi
	and	$0xfff, %edi
	sub	$0x1000, %edi

	.p2align 4
L(loop_ashr_1):
	add	$16, %edi
	jg	L(nibble_ashr_1)

L(gobble_ashr_1):
	movdqa	(%eax, %ecx), %xmm1
	movdqa	(%edx, %ecx), %xmm2
	movdqa	%xmm2, %xmm4

	palignr	$1, %xmm3, %xmm2
	TOLOWER (%xmm1, %xmm2)

	pcmpeqb	%xmm1, %xmm0
	pcmpeqb	%xmm2, %xmm1
	psubb	%xmm0, %xmm1
	pmovmskb %xmm1, %esi
	sub	$0xffff, %esi
	jnz	L(exit)
#if defined USE_AS_STRNCMP || defined USE_AS_STRNCASECMP_L
	cmp	$16, REM
	lea	-16(REM), REM
	jbe	L(more8byteseq)
#endif

	add	$16, %ecx
	movdqa	%xmm4, %xmm3

	add	$16, %edi
	jg	L(nibble_ashr_1)

	movdqa	(%eax, %ecx), %xmm1
	movdqa	(%edx, %ecx), %xmm2
	movdqa	%xmm2, %xmm4

	palignr	$1, %xmm3, %xmm2
	TOLOWER (%xmm1, %xmm2)

	pcmpeqb	%xmm1, %xmm0
	pcmpeqb	%xmm2, %xmm1
	psubb	%xmm0, %xmm1
	pmovmskb %xmm1, %esi
	sub	$0xffff, %esi
	jnz	L(exit)

#if defined USE_AS_STRNCMP || defined USE_AS_STRNCASECMP_L
	cmp	$16, REM
	lea	-16(REM), REM
	jbe	L(more8byteseq)
#endif
	add	$16, %ecx
	movdqa	%xmm4, %xmm3
	jmp	L(loop_ashr_1)

	.p2align 4
L(nibble_ashr_1):
	pcmpeqb	%xmm3, %xmm0
	pmovmskb %xmm0, %esi
	test	$0xfffe, %esi
	jnz	L(ashr_1_exittail)

#if defined USE_AS_STRNCMP || defined USE_AS_STRNCASECMP_L
	cmp	$15, REM
	jbe	L(ashr_1_exittail)
#endif
	pxor	%xmm0, %xmm0
	sub	$0x1000, %edi
	jmp	L(gobble_ashr_1)

	.p2align 4
L(ashr_1_exittail):
	movdqa	(%eax, %ecx), %xmm1
	psrldq	$1, %xmm0
	psrldq	$1, %xmm3
	jmp	L(aftertail)

/*
 * The following cases will be handled by ashr_2
 * ecx(offset of esi)  eax(offset of edi)   relative offset	corresponding case
 *        n(14~15)            n -14            1(15 +(n-14) - n)         ashr_2
 */
	.p2align 4
L(ashr_2):
	mov	$0xffff, %esi
	pxor	%xmm0, %xmm0
	movdqa	(%edx), %xmm2
	movdqa	(%eax), %xmm1
	pcmpeqb	%xmm1, %xmm0
	pslldq	$14, %xmm2
	TOLOWER (%xmm1, %xmm2)
	pcmpeqb	%xmm1, %xmm2
	psubb	%xmm0, %xmm2
	pmovmskb %xmm2, %edi
	shr	%cl, %esi
	shr	%cl, %edi
	sub	%edi, %esi
	lea	-14(%ecx), %edi
	jnz	L(less32bytes)

	UPDATE_STRNCMP_COUNTER

	movdqa	(%edx), %xmm3
	pxor	%xmm0, %xmm0
	mov	$16, %ecx
	orl	$2, FLAGS
	lea	2(%edx), %edi
	and	$0xfff, %edi
	sub	$0x1000, %edi

	.p2align 4
L(loop_ashr_2):
	add	$16, %edi
	jg	L(nibble_ashr_2)

L(gobble_ashr_2):
	movdqa	(%eax, %ecx), %xmm1
	movdqa	(%edx, %ecx), %xmm2
	movdqa	%xmm2, %xmm4

	palignr	$2, %xmm3, %xmm2
	TOLOWER (%xmm1, %xmm2)

	pcmpeqb	%xmm1, %xmm0
	pcmpeqb	%xmm2, %xmm1
	psubb	%xmm0, %xmm1
	pmovmskb %xmm1, %esi
	sub	$0xffff, %esi
	jnz	L(exit)

#if defined USE_AS_STRNCMP || defined USE_AS_STRNCASECMP_L
	cmp	$16, REM
	lea	-16(REM), REM
	jbe	L(more8byteseq)
#endif
	add	$16, %ecx
	movdqa	%xmm4, %xmm3

	add	$16, %edi
	jg	L(nibble_ashr_2)

	movdqa	(%eax, %ecx), %xmm1
	movdqa	(%edx, %ecx), %xmm2
	movdqa	%xmm2, %xmm4

	palignr	$2, %xmm3, %xmm2
	TOLOWER (%xmm1, %xmm2)

	pcmpeqb	%xmm1, %xmm0
	pcmpeqb	%xmm2, %xmm1
	psubb	%xmm0, %xmm1
	pmovmskb %xmm1, %esi
	sub	$0xffff, %esi
	jnz	L(exit)

#if defined USE_AS_STRNCMP || defined USE_AS_STRNCASECMP_L
	cmp	$16, REM
	lea	-16(REM), REM
	jbe	L(more8byteseq)
#endif
	add	$16, %ecx
	movdqa	%xmm4, %xmm3
	jmp	L(loop_ashr_2)

	.p2align 4
L(nibble_ashr_2):
	pcmpeqb	%xmm3, %xmm0
	pmovmskb %xmm0, %esi
	test	$0xfffc, %esi
	jnz	L(ashr_2_exittail)

#if defined USE_AS_STRNCMP || defined USE_AS_STRNCASECMP_L
	cmp	$14, REM
	jbe	L(ashr_2_exittail)
#endif

	pxor	%xmm0, %xmm0
	sub	$0x1000, %edi
	jmp	L(gobble_ashr_2)

	.p2align 4
L(ashr_2_exittail):
	movdqa	(%eax, %ecx), %xmm1
	psrldq	$2, %xmm0
	psrldq	$2, %xmm3
	jmp	L(aftertail)

/*
 * The following cases will be handled by ashr_3
 * ecx(offset of esi)  eax(offset of edi)   relative offset	corresponding case
 *        n(13~15)            n -13            2(15 +(n-13) - n)         ashr_3
 */
	.p2align 4
L(ashr_3):
	mov	$0xffff, %esi
	pxor	%xmm0, %xmm0
	movdqa	(%edx), %xmm2
	movdqa	(%eax), %xmm1
	pcmpeqb	%xmm1, %xmm0
	pslldq	$13, %xmm2
	TOLOWER (%xmm1, %xmm2)
	pcmpeqb	%xmm1, %xmm2
	psubb	%xmm0, %xmm2
	pmovmskb %xmm2, %edi
	shr	%cl, %esi
	shr	%cl, %edi
	sub	%edi, %esi
	lea	-13(%ecx), %edi
	jnz	L(less32bytes)

	UPDATE_STRNCMP_COUNTER

	movdqa	(%edx), %xmm3
	pxor	%xmm0, %xmm0
	mov	$16, %ecx
	orl	$3, FLAGS
	lea	3(%edx), %edi
	and	$0xfff, %edi
	sub	$0x1000, %edi

	.p2align 4
L(loop_ashr_3):
	add	$16, %edi
	jg	L(nibble_ashr_3)

L(gobble_ashr_3):
	movdqa	(%eax, %ecx), %xmm1
	movdqa	(%edx, %ecx), %xmm2
	movdqa	%xmm2, %xmm4

	palignr	$3, %xmm3, %xmm2
	TOLOWER (%xmm1, %xmm2)

	pcmpeqb	%xmm1, %xmm0
	pcmpeqb	%xmm2, %xmm1
	psubb	%xmm0, %xmm1
	pmovmskb %xmm1, %esi
	sub	$0xffff, %esi
	jnz	L(exit)

#if defined USE_AS_STRNCMP || defined USE_AS_STRNCASECMP_L
	cmp	$16, REM
	lea	-16(REM), REM
	jbe	L(more8byteseq)
#endif
	add	$16, %ecx
	movdqa	%xmm4, %xmm3

	add	$16, %edi
	jg	L(nibble_ashr_3)

	movdqa	(%eax, %ecx), %xmm1
	movdqa	(%edx, %ecx), %xmm2
	movdqa	%xmm2, %xmm4

	palignr	$3, %xmm3, %xmm2
	TOLOWER (%xmm1, %xmm2)

	pcmpeqb	%xmm1, %xmm0
	pcmpeqb	%xmm2, %xmm1
	psubb	%xmm0, %xmm1
	pmovmskb %xmm1, %esi
	sub	$0xffff, %esi
	jnz	L(exit)

#if defined USE_AS_STRNCMP || defined USE_AS_STRNCASECMP_L
	cmp	$16, REM
	lea	-16(REM), REM
	jbe	L(more8byteseq)
#endif
	add	$16, %ecx
	movdqa	%xmm4, %xmm3
	jmp	L(loop_ashr_3)

	.p2align 4
L(nibble_ashr_3):
	pcmpeqb	%xmm3, %xmm0
	pmovmskb %xmm0, %esi
	test	$0xfff8, %esi
	jnz	L(ashr_3_exittail)

#if defined USE_AS_STRNCMP || defined USE_AS_STRNCASECMP_L
	cmp	$13, REM
	jbe	L(ashr_3_exittail)
#endif
	pxor	%xmm0, %xmm0
	sub	$0x1000, %edi
	jmp	L(gobble_ashr_3)

	.p2align 4
L(ashr_3_exittail):
	movdqa	(%eax, %ecx), %xmm1
	psrldq	$3, %xmm0
	psrldq	$3, %xmm3
	jmp	L(aftertail)

/*
 * The following cases will be handled by ashr_4
 * ecx(offset of esi)  eax(offset of edi)   relative offset	corresponding case
 *        n(12~15)            n -12            3(15 +(n-12) - n)         ashr_4
 */
	.p2align 4
L(ashr_4):
	mov	$0xffff, %esi
	pxor	%xmm0, %xmm0
	movdqa	(%edx), %xmm2
	movdqa	(%eax), %xmm1
	pcmpeqb	%xmm1, %xmm0
	pslldq	$12, %xmm2
	TOLOWER (%xmm1, %xmm2)
	pcmpeqb	%xmm1, %xmm2
	psubb	%xmm0, %xmm2
	pmovmskb %xmm2, %edi
	shr	%cl, %esi
	shr	%cl, %edi
	sub	%edi, %esi
	lea	-12(%ecx), %edi
	jnz	L(less32bytes)

	UPDATE_STRNCMP_COUNTER

	movdqa	(%edx), %xmm3
	pxor	%xmm0, %xmm0
	mov	$16, %ecx
	orl	$4, FLAGS
	lea	4(%edx), %edi
	and	$0xfff, %edi
	sub	$0x1000, %edi

	.p2align 4
L(loop_ashr_4):
	add	$16, %edi
	jg	L(nibble_ashr_4)

L(gobble_ashr_4):
	movdqa	(%eax, %ecx), %xmm1
	movdqa	(%edx, %ecx), %xmm2
	movdqa	%xmm2, %xmm4

	palignr	$4, %xmm3, %xmm2
	TOLOWER (%xmm1, %xmm2)

	pcmpeqb	%xmm1, %xmm0
	pcmpeqb	%xmm2, %xmm1
	psubb	%xmm0, %xmm1
	pmovmskb %xmm1, %esi
	sub	$0xffff, %esi
	jnz	L(exit)

#if defined USE_AS_STRNCMP || defined USE_AS_STRNCASECMP_L
	cmp	$16, REM
	lea	-16(REM), REM
	jbe	L(more8byteseq)
#endif

	add	$16, %ecx
	movdqa	%xmm4, %xmm3

	add	$16, %edi
	jg	L(nibble_ashr_4)

	movdqa	(%eax, %ecx), %xmm1
	movdqa	(%edx, %ecx), %xmm2
	movdqa	%xmm2, %xmm4

	palignr	$4, %xmm3, %xmm2
	TOLOWER (%xmm1, %xmm2)

	pcmpeqb	%xmm1, %xmm0
	pcmpeqb	%xmm2, %xmm1
	psubb	%xmm0, %xmm1
	pmovmskb %xmm1, %esi
	sub	$0xffff, %esi
	jnz	L(exit)

#if defined USE_AS_STRNCMP || defined USE_AS_STRNCASECMP_L
	cmp	$16, REM
	lea	-16(REM), REM
	jbe	L(more8byteseq)
#endif

	add	$16, %ecx
	movdqa	%xmm4, %xmm3
	jmp	L(loop_ashr_4)

	.p2align 4
L(nibble_ashr_4):
	pcmpeqb	%xmm3, %xmm0
	pmovmskb %xmm0, %esi
	test	$0xfff0, %esi
	jnz	L(ashr_4_exittail)

#if defined USE_AS_STRNCMP || defined USE_AS_STRNCASECMP_L
	cmp	$12, REM
	jbe	L(ashr_4_exittail)
#endif

	pxor	%xmm0, %xmm0
	sub	$0x1000, %edi
	jmp	L(gobble_ashr_4)

	.p2align 4
L(ashr_4_exittail):
	movdqa	(%eax, %ecx), %xmm1
	psrldq	$4, %xmm0
	psrldq	$4, %xmm3
	jmp	L(aftertail)

/*
 * The following cases will be handled by ashr_5
 * ecx(offset of esi)  eax(offset of edi)   relative offset	corresponding case
 *        n(11~15)            n -11            4(15 +(n-11) - n)         ashr_5
 */
	.p2align 4
L(ashr_5):
	mov	$0xffff, %esi
	pxor	%xmm0, %xmm0
	movdqa	(%edx), %xmm2
	movdqa	(%eax), %xmm1
	pcmpeqb	%xmm1, %xmm0
	pslldq	$11, %xmm2
	TOLOWER (%xmm1, %xmm2)
	pcmpeqb	%xmm1, %xmm2
	psubb	%xmm0, %xmm2
	pmovmskb %xmm2, %edi
	shr	%cl, %esi
	shr	%cl, %edi
	sub	%edi, %esi
	lea	-11(%ecx), %edi
	jnz	L(less32bytes)

	UPDATE_STRNCMP_COUNTER

	movdqa	(%edx), %xmm3
	pxor	%xmm0, %xmm0
	mov	$16, %ecx
	orl	$5, FLAGS
	lea	5(%edx), %edi
	and	$0xfff, %edi
	sub	$0x1000, %edi

	.p2align 4
L(loop_ashr_5):
	add	$16, %edi
	jg	L(nibble_ashr_5)

L(gobble_ashr_5):
	movdqa	(%eax, %ecx), %xmm1
	movdqa	(%edx, %ecx), %xmm2
	movdqa	%xmm2, %xmm4

	palignr	$5, %xmm3, %xmm2
	TOLOWER (%xmm1, %xmm2)

	pcmpeqb	%xmm1, %xmm0
	pcmpeqb	%xmm2, %xmm1
	psubb	%xmm0, %xmm1
	pmovmskb %xmm1, %esi
	sub	$0xffff, %esi
	jnz	L(exit)

#if defined USE_AS_STRNCMP || defined USE_AS_STRNCASECMP_L
	cmp	$16, REM
	lea	-16(REM), REM
	jbe	L(more8byteseq)
#endif
	add	$16, %ecx
	movdqa	%xmm4, %xmm3

	add	$16, %edi
	jg	L(nibble_ashr_5)

	movdqa	(%eax, %ecx), %xmm1
	movdqa	(%edx, %ecx), %xmm2
	movdqa	%xmm2, %xmm4

	palignr	$5, %xmm3, %xmm2
	TOLOWER (%xmm1, %xmm2)

	pcmpeqb	%xmm1, %xmm0
	pcmpeqb	%xmm2, %xmm1
	psubb	%xmm0, %xmm1
	pmovmskb %xmm1, %esi
	sub	$0xffff, %esi
	jnz	L(exit)

#if defined USE_AS_STRNCMP || defined USE_AS_STRNCASECMP_L
	cmp	$16, REM
	lea	-16(REM), REM
	jbe	L(more8byteseq)
#endif
	add	$16, %ecx
	movdqa	%xmm4, %xmm3
	jmp	L(loop_ashr_5)

	.p2align 4
L(nibble_ashr_5):
	pcmpeqb	%xmm3, %xmm0
	pmovmskb %xmm0, %esi
	test	$0xffe0, %esi
	jnz	L(ashr_5_exittail)

#if defined USE_AS_STRNCMP || defined USE_AS_STRNCASECMP_L
	cmp	$11, REM
	jbe	L(ashr_5_exittail)
#endif
	pxor	%xmm0, %xmm0
	sub	$0x1000, %edi
	jmp	L(gobble_ashr_5)

	.p2align 4
L(ashr_5_exittail):
	movdqa	(%eax, %ecx), %xmm1
	psrldq	$5, %xmm0
	psrldq	$5, %xmm3
	jmp	L(aftertail)

/*
 * The following cases will be handled by ashr_6
 * ecx(offset of esi)  eax(offset of edi)   relative offset	corresponding case
 *        n(10~15)            n -10            5(15 +(n-10) - n)         ashr_6
 */

	.p2align 4
L(ashr_6):
	mov	$0xffff, %esi
	pxor	%xmm0, %xmm0
	movdqa	(%edx), %xmm2
	movdqa	(%eax), %xmm1
	pcmpeqb	%xmm1, %xmm0
	pslldq	$10, %xmm2
	TOLOWER (%xmm1, %xmm2)
	pcmpeqb	%xmm1, %xmm2
	psubb	%xmm0, %xmm2
	pmovmskb %xmm2, %edi
	shr	%cl, %esi
	shr	%cl, %edi
	sub	%edi, %esi
	lea	-10(%ecx), %edi
	jnz	L(less32bytes)

	UPDATE_STRNCMP_COUNTER

	movdqa	(%edx), %xmm3
	pxor	%xmm0, %xmm0
	mov	$16, %ecx
	orl	$6, FLAGS
	lea	6(%edx), %edi
	and	$0xfff, %edi
	sub	$0x1000, %edi

	.p2align 4
L(loop_ashr_6):
	add	$16, %edi
	jg	L(nibble_ashr_6)

L(gobble_ashr_6):
	movdqa	(%eax, %ecx), %xmm1
	movdqa	(%edx, %ecx), %xmm2
	movdqa	%xmm2, %xmm4

	palignr	$6, %xmm3, %xmm2
	TOLOWER (%xmm1, %xmm2)

	pcmpeqb	%xmm1, %xmm0
	pcmpeqb	%xmm2, %xmm1
	psubb	%xmm0, %xmm1
	pmovmskb %xmm1, %esi
	sub	$0xffff, %esi
	jnz	L(exit)

#if defined USE_AS_STRNCMP || defined USE_AS_STRNCASECMP_L
	cmp	$16, REM
	lea	-16(REM), REM
	jbe	L(more8byteseq)
#endif

	add	$16, %ecx
	movdqa	%xmm4, %xmm3

	add	$16, %edi
	jg	L(nibble_ashr_6)

	movdqa	(%eax, %ecx), %xmm1
	movdqa	(%edx, %ecx), %xmm2
	movdqa	%xmm2, %xmm4

	palignr	$6, %xmm3, %xmm2
	TOLOWER (%xmm1, %xmm2)

	pcmpeqb	%xmm1, %xmm0
	pcmpeqb	%xmm2, %xmm1
	psubb	%xmm0, %xmm1
	pmovmskb %xmm1, %esi
	sub	$0xffff, %esi
	jnz	L(exit)
#if defined USE_AS_STRNCMP || defined USE_AS_STRNCASECMP_L
	cmp	$16, REM
	lea	-16(REM), REM
	jbe	L(more8byteseq)
#endif

	add	$16, %ecx
	movdqa	%xmm4, %xmm3
	jmp	L(loop_ashr_6)

	.p2align 4
L(nibble_ashr_6):
	pcmpeqb	%xmm3, %xmm0
	pmovmskb %xmm0, %esi
	test	$0xffc0, %esi
	jnz	L(ashr_6_exittail)

#if defined USE_AS_STRNCMP || defined USE_AS_STRNCASECMP_L
	cmp	$10, REM
	jbe	L(ashr_6_exittail)
#endif
	pxor	%xmm0, %xmm0
	sub	$0x1000, %edi
	jmp	L(gobble_ashr_6)

	.p2align 4
L(ashr_6_exittail):
	movdqa	(%eax, %ecx), %xmm1
	psrldq	$6, %xmm0
	psrldq	$6, %xmm3
	jmp	L(aftertail)

/*
 * The following cases will be handled by ashr_7
 * ecx(offset of esi)  eax(offset of edi)   relative offset	corresponding case
 *        n(9~15)            n - 9            6(15 +(n-9) - n)         ashr_7
 */

	.p2align 4
L(ashr_7):
	mov	$0xffff, %esi
	pxor	%xmm0, %xmm0
	movdqa	(%edx), %xmm2
	movdqa	(%eax), %xmm1
	pcmpeqb	%xmm1, %xmm0
	pslldq	$9, %xmm2
	TOLOWER (%xmm1, %xmm2)
	pcmpeqb	%xmm1, %xmm2
	psubb	%xmm0, %xmm2
	pmovmskb %xmm2, %edi
	shr	%cl, %esi
	shr	%cl, %edi
	sub	%edi, %esi
	lea	-9(%ecx), %edi
	jnz	L(less32bytes)

	UPDATE_STRNCMP_COUNTER

	movdqa	(%edx), %xmm3
	pxor	%xmm0, %xmm0
	mov	$16, %ecx
	orl	$7, FLAGS
	lea	8(%edx), %edi
	and	$0xfff, %edi
	sub	$0x1000, %edi

	.p2align 4
L(loop_ashr_7):
	add	$16, %edi
	jg	L(nibble_ashr_7)

L(gobble_ashr_7):
	movdqa	(%eax, %ecx), %xmm1
	movdqa	(%edx, %ecx), %xmm2
	movdqa	%xmm2, %xmm4

	palignr	$7, %xmm3, %xmm2
	TOLOWER (%xmm1, %xmm2)

	pcmpeqb	%xmm1, %xmm0
	pcmpeqb	%xmm2, %xmm1
	psubb	%xmm0, %xmm1
	pmovmskb %xmm1, %esi
	sub	$0xffff, %esi
	jnz	L(exit)

#if defined USE_AS_STRNCMP || defined USE_AS_STRNCASECMP_L
	cmp	$16, REM
	lea	-16(REM), REM
	jbe	L(more8byteseq)
#endif

	add	$16, %ecx
	movdqa	%xmm4, %xmm3

	add	$16, %edi
	jg	L(nibble_ashr_7)

	movdqa	(%eax, %ecx), %xmm1
	movdqa	(%edx, %ecx), %xmm2
	movdqa	%xmm2, %xmm4

	palignr	$7, %xmm3, %xmm2
	TOLOWER (%xmm1, %xmm2)

	pcmpeqb	%xmm1, %xmm0
	pcmpeqb	%xmm2, %xmm1
	psubb	%xmm0, %xmm1
	pmovmskb %xmm1, %esi
	sub	$0xffff, %esi
	jnz	L(exit)

#if defined USE_AS_STRNCMP || defined USE_AS_STRNCASECMP_L
	cmp	$16, REM
	lea	-16(REM), REM
	jbe	L(more8byteseq)
#endif

	add	$16, %ecx
	movdqa	%xmm4, %xmm3
	jmp	L(loop_ashr_7)

	.p2align 4
L(nibble_ashr_7):
	pcmpeqb	%xmm3, %xmm0
	pmovmskb %xmm0, %esi
	test	$0xff80, %esi
	jnz	L(ashr_7_exittail)

#if defined USE_AS_STRNCMP || defined USE_AS_STRNCASECMP_L
	cmp	$9, REM
	jbe	L(ashr_7_exittail)
#endif
	pxor	%xmm0, %xmm0
	pxor	%xmm0, %xmm0
	sub	$0x1000, %edi
	jmp	L(gobble_ashr_7)

	.p2align 4
L(ashr_7_exittail):
	movdqa	(%eax, %ecx), %xmm1
	psrldq	$7, %xmm0
	psrldq	$7, %xmm3
	jmp	L(aftertail)

/*
 * The following cases will be handled by ashr_8
 * ecx(offset of esi)  eax(offset of edi)   relative offset	corresponding case
 *        n(8~15)            n - 8            7(15 +(n-8) - n)         ashr_8
 */
	.p2align 4
L(ashr_8):
	mov	$0xffff, %esi
	pxor	%xmm0, %xmm0
	movdqa	(%edx), %xmm2
	movdqa	(%eax), %xmm1
	pcmpeqb	%xmm1, %xmm0
	pslldq	$8, %xmm2
	TOLOWER (%xmm1, %xmm2)
	pcmpeqb	%xmm1, %xmm2
	psubb	%xmm0, %xmm2
	pmovmskb %xmm2, %edi
	shr	%cl, %esi
	shr	%cl, %edi
	sub	%edi, %esi
	lea	-8(%ecx), %edi
	jnz	L(less32bytes)

	UPDATE_STRNCMP_COUNTER

	movdqa	(%edx), %xmm3
	pxor	%xmm0, %xmm0
	mov	$16, %ecx
	orl	$8, FLAGS
	lea	8(%edx), %edi
	and	$0xfff, %edi
	sub	$0x1000, %edi

	.p2align 4
L(loop_ashr_8):
	add	$16, %edi
	jg	L(nibble_ashr_8)

L(gobble_ashr_8):
	movdqa	(%eax, %ecx), %xmm1
	movdqa	(%edx, %ecx), %xmm2
	movdqa	%xmm2, %xmm4

	palignr	$8, %xmm3, %xmm2
	TOLOWER (%xmm1, %xmm2)

	pcmpeqb	%xmm1, %xmm0
	pcmpeqb	%xmm2, %xmm1
	psubb	%xmm0, %xmm1
	pmovmskb %xmm1, %esi
	sub	$0xffff, %esi
	jnz	L(exit)

#if defined USE_AS_STRNCMP || defined USE_AS_STRNCASECMP_L
	cmp	$16, REM
	lea	-16(REM), REM
	jbe	L(more8byteseq)
#endif
	add	$16, %ecx
	movdqa	%xmm4, %xmm3

	add	$16, %edi
	jg	L(nibble_ashr_8)

	movdqa	(%eax, %ecx), %xmm1
	movdqa	(%edx, %ecx), %xmm2
	movdqa	%xmm2, %xmm4

	palignr	$8, %xmm3, %xmm2
	TOLOWER (%xmm1, %xmm2)

	pcmpeqb	%xmm1, %xmm0
	pcmpeqb	%xmm2, %xmm1
	psubb	%xmm0, %xmm1
	pmovmskb %xmm1, %esi
	sub	$0xffff, %esi
	jnz	L(exit)

#if defined USE_AS_STRNCMP || defined USE_AS_STRNCASECMP_L
	cmp	$16, REM
	lea	-16(REM), REM
	jbe	L(more8byteseq)
#endif
	add	$16, %ecx
	movdqa	%xmm4, %xmm3
	jmp	L(loop_ashr_8)

	.p2align 4
L(nibble_ashr_8):
	pcmpeqb	%xmm3, %xmm0
	pmovmskb %xmm0, %esi
	test	$0xff00, %esi
	jnz	L(ashr_8_exittail)

#if defined USE_AS_STRNCMP || defined USE_AS_STRNCASECMP_L
	cmp	$8, REM
	jbe	L(ashr_8_exittail)
#endif
	pxor	%xmm0, %xmm0
	pxor	%xmm0, %xmm0
	sub	$0x1000, %edi
	jmp	L(gobble_ashr_8)

	.p2align 4
L(ashr_8_exittail):
	movdqa	(%eax, %ecx), %xmm1
	psrldq	$8, %xmm0
	psrldq	$8, %xmm3
	jmp	L(aftertail)

/*
 * The following cases will be handled by ashr_9
 * ecx(offset of esi)  eax(offset of edi)   relative offset	corresponding case
 *        n(7~15)            n - 7            8(15 +(n-7) - n)         ashr_9
 */
	.p2align 4
L(ashr_9):
	mov	$0xffff, %esi
	pxor	%xmm0, %xmm0
	movdqa	(%edx), %xmm2
	movdqa	(%eax), %xmm1
	pcmpeqb	%xmm1, %xmm0
	pslldq	$7, %xmm2
	TOLOWER (%xmm1, %xmm2)
	pcmpeqb	%xmm1, %xmm2
	psubb	%xmm0, %xmm2
	pmovmskb %xmm2, %edi
	shr	%cl, %esi
	shr	%cl, %edi
	sub	%edi, %esi
	lea	-7(%ecx), %edi
	jnz	L(less32bytes)

	UPDATE_STRNCMP_COUNTER

	movdqa	(%edx), %xmm3
	pxor	%xmm0, %xmm0
	mov	$16, %ecx
	orl	$9, FLAGS
	lea	9(%edx), %edi
	and	$0xfff, %edi
	sub	$0x1000, %edi

	.p2align 4
L(loop_ashr_9):
	add	$16, %edi
	jg	L(nibble_ashr_9)

L(gobble_ashr_9):
	movdqa	(%eax, %ecx), %xmm1
	movdqa	(%edx, %ecx), %xmm2
	movdqa	%xmm2, %xmm4

	palignr	$9, %xmm3, %xmm2
	TOLOWER (%xmm1, %xmm2)

	pcmpeqb	%xmm1, %xmm0
	pcmpeqb	%xmm2, %xmm1
	psubb	%xmm0, %xmm1
	pmovmskb %xmm1, %esi
	sub	$0xffff, %esi
	jnz	L(exit)

#if defined USE_AS_STRNCMP || defined USE_AS_STRNCASECMP_L
	cmp	$16, REM
	lea	-16(REM), REM
	jbe	L(more8byteseq)
#endif
	add	$16, %ecx
	movdqa	%xmm4, %xmm3

	add	$16, %edi
	jg	L(nibble_ashr_9)

	movdqa	(%eax, %ecx), %xmm1
	movdqa	(%edx, %ecx), %xmm2
	movdqa	%xmm2, %xmm4

	palignr	$9, %xmm3, %xmm2
	TOLOWER (%xmm1, %xmm2)

	pcmpeqb	%xmm1, %xmm0
	pcmpeqb	%xmm2, %xmm1
	psubb	%xmm0, %xmm1
	pmovmskb %xmm1, %esi
	sub	$0xffff, %esi
	jnz	L(exit)

#if defined USE_AS_STRNCMP || defined USE_AS_STRNCASECMP_L
	cmp	$16, REM
	lea	-16(REM), REM
	jbe	L(more8byteseq)
#endif
	add	$16, %ecx
	movdqa	%xmm4, %xmm3
	jmp	L(loop_ashr_9)

	.p2align 4
L(nibble_ashr_9):
	pcmpeqb	%xmm3, %xmm0
	pmovmskb %xmm0, %esi
	test	$0xfe00, %esi
	jnz	L(ashr_9_exittail)

#if defined USE_AS_STRNCMP || defined USE_AS_STRNCASECMP_L
	cmp	$7, REM
	jbe	L(ashr_9_exittail)
#endif
	pxor	%xmm0, %xmm0
	sub	$0x1000, %edi
	jmp	L(gobble_ashr_9)

	.p2align 4
L(ashr_9_exittail):
	movdqa	(%eax, %ecx), %xmm1
	psrldq	$9, %xmm0
	psrldq	$9, %xmm3
	jmp	L(aftertail)

/*
 * The following cases will be handled by ashr_10
 * ecx(offset of esi)  eax(offset of edi)   relative offset	corresponding case
 *        n(6~15)            n - 6            9(15 +(n-6) - n)         ashr_10
 */
	.p2align 4
L(ashr_10):
	mov	$0xffff, %esi
	pxor	%xmm0, %xmm0
	movdqa	(%edx), %xmm2
	movdqa	(%eax), %xmm1
	pcmpeqb	%xmm1, %xmm0
	pslldq	$6, %xmm2
	TOLOWER (%xmm1, %xmm2)
	pcmpeqb	%xmm1, %xmm2
	psubb	%xmm0, %xmm2
	pmovmskb %xmm2, %edi
	shr	%cl, %esi
	shr	%cl, %edi
	sub	%edi, %esi
	lea	-6(%ecx), %edi
	jnz	L(less32bytes)

	UPDATE_STRNCMP_COUNTER

	movdqa	(%edx), %xmm3
	pxor	%xmm0, %xmm0
	mov	$16, %ecx
	orl	$10, FLAGS
	lea	10(%edx), %edi
	and	$0xfff, %edi
	sub	$0x1000, %edi

	.p2align 4
L(loop_ashr_10):
	add	$16, %edi
	jg	L(nibble_ashr_10)

L(gobble_ashr_10):
	movdqa	(%eax, %ecx), %xmm1
	movdqa	(%edx, %ecx), %xmm2
	movdqa	%xmm2, %xmm4

	palignr	$10, %xmm3, %xmm2
	TOLOWER (%xmm1, %xmm2)

	pcmpeqb	%xmm1, %xmm0
	pcmpeqb	%xmm2, %xmm1
	psubb	%xmm0, %xmm1
	pmovmskb %xmm1, %esi
	sub	$0xffff, %esi
	jnz	L(exit)

#if defined USE_AS_STRNCMP || defined USE_AS_STRNCASECMP_L
	cmp	$16, REM
	lea	-16(REM), REM
	jbe	L(more8byteseq)
#endif
	add	$16, %ecx
	movdqa	%xmm4, %xmm3

	add	$16, %edi
	jg	L(nibble_ashr_10)

	movdqa	(%eax, %ecx), %xmm1
	movdqa	(%edx, %ecx), %xmm2
	movdqa	%xmm2, %xmm4

	palignr	$10, %xmm3, %xmm2
	TOLOWER (%xmm1, %xmm2)

	pcmpeqb	%xmm1, %xmm0
	pcmpeqb	%xmm2, %xmm1
	psubb	%xmm0, %xmm1
	pmovmskb %xmm1, %esi
	sub	$0xffff, %esi
	jnz	L(exit)

#if defined USE_AS_STRNCMP || defined USE_AS_STRNCASECMP_L
	cmp	$16, REM
	lea	-16(REM), REM
	jbe	L(more8byteseq)
#endif
	add	$16, %ecx
	movdqa	%xmm4, %xmm3
	jmp	L(loop_ashr_10)

	.p2align 4
L(nibble_ashr_10):
	pcmpeqb	%xmm3, %xmm0
	pmovmskb %xmm0, %esi
	test	$0xfc00, %esi
	jnz	L(ashr_10_exittail)

#if defined USE_AS_STRNCMP || defined USE_AS_STRNCASECMP_L
	cmp	$6, REM
	jbe	L(ashr_10_exittail)
#endif
	pxor	%xmm0, %xmm0
	sub	$0x1000, %edi
	jmp	L(gobble_ashr_10)

	.p2align 4
L(ashr_10_exittail):
	movdqa	(%eax, %ecx), %xmm1
	psrldq	$10, %xmm0
	psrldq	$10, %xmm3
	jmp	L(aftertail)

/*
 * The following cases will be handled by ashr_11
 * ecx(offset of esi)  eax(offset of edi)   relative offset	corresponding case
 *        n(5~15)            n - 5            10(15 +(n-5) - n)         ashr_11
 */
	.p2align 4
L(ashr_11):
	mov	$0xffff, %esi
	pxor	%xmm0, %xmm0
	movdqa	(%edx), %xmm2
	movdqa	(%eax), %xmm1
	pcmpeqb	%xmm1, %xmm0
	pslldq	$5, %xmm2
	TOLOWER (%xmm1, %xmm2)
	pcmpeqb	%xmm1, %xmm2
	psubb	%xmm0, %xmm2
	pmovmskb %xmm2, %edi
	shr	%cl, %esi
	shr	%cl, %edi
	sub	%edi, %esi
	lea	-5(%ecx), %edi
	jnz	L(less32bytes)

	UPDATE_STRNCMP_COUNTER

	movdqa	(%edx), %xmm3
	pxor	%xmm0, %xmm0
	mov	$16, %ecx
	orl	$11, FLAGS
	lea	11(%edx), %edi
	and	$0xfff, %edi
	sub	$0x1000, %edi

	.p2align 4
L(loop_ashr_11):
	add	$16, %edi
	jg	L(nibble_ashr_11)

L(gobble_ashr_11):
	movdqa	(%eax, %ecx), %xmm1
	movdqa	(%edx, %ecx), %xmm2
	movdqa	%xmm2, %xmm4

	palignr	$11, %xmm3, %xmm2
	TOLOWER (%xmm1, %xmm2)

	pcmpeqb	%xmm1, %xmm0
	pcmpeqb	%xmm2, %xmm1
	psubb	%xmm0, %xmm1
	pmovmskb %xmm1, %esi
	sub	$0xffff, %esi
	jnz	L(exit)

#if defined USE_AS_STRNCMP || defined USE_AS_STRNCASECMP_L
	cmp	$16, REM
	lea	-16(REM), REM
	jbe	L(more8byteseq)
#endif
	add	$16, %ecx
	movdqa	%xmm4, %xmm3

	add	$16, %edi
	jg	L(nibble_ashr_11)

	movdqa	(%eax, %ecx), %xmm1
	movdqa	(%edx, %ecx), %xmm2
	movdqa	%xmm2, %xmm4

	palignr	$11, %xmm3, %xmm2
	TOLOWER (%xmm1, %xmm2)

	pcmpeqb	%xmm1, %xmm0
	pcmpeqb	%xmm2, %xmm1
	psubb	%xmm0, %xmm1
	pmovmskb %xmm1, %esi
	sub	$0xffff, %esi
	jnz	L(exit)

#if defined USE_AS_STRNCMP || defined USE_AS_STRNCASECMP_L
	cmp	$16, REM
	lea	-16(REM), REM
	jbe	L(more8byteseq)
#endif
	add	$16, %ecx
	movdqa	%xmm4, %xmm3
	jmp	L(loop_ashr_11)

	.p2align 4
L(nibble_ashr_11):
	pcmpeqb	%xmm3, %xmm0
	pmovmskb %xmm0, %esi
	test	$0xf800, %esi
	jnz	L(ashr_11_exittail)

#if defined USE_AS_STRNCMP || defined USE_AS_STRNCASECMP_L
	cmp	$5, REM
	jbe	L(ashr_11_exittail)
#endif
	pxor	%xmm0, %xmm0
	sub	$0x1000, %edi
	jmp	L(gobble_ashr_11)

	.p2align 4
L(ashr_11_exittail):
	movdqa	(%eax, %ecx), %xmm1
	psrldq	$11, %xmm0
	psrldq	$11, %xmm3
	jmp	L(aftertail)

/*
 * The following cases will be handled by ashr_12
 * ecx(offset of esi)  eax(offset of edi)   relative offset	corresponding case
 *        n(4~15)            n - 4            11(15 +(n-4) - n)         ashr_12
 */
	.p2align 4
L(ashr_12):
	mov	$0xffff, %esi
	pxor	%xmm0, %xmm0
	movdqa	(%edx), %xmm2
	movdqa	(%eax), %xmm1
	pcmpeqb	%xmm1, %xmm0
	pslldq	$4, %xmm2
	TOLOWER (%xmm1, %xmm2)
	pcmpeqb	%xmm1, %xmm2
	psubb	%xmm0, %xmm2
	pmovmskb %xmm2, %edi
	shr	%cl, %esi
	shr	%cl, %edi
	sub	%edi, %esi
	lea	-4(%ecx), %edi
	jnz	L(less32bytes)

	UPDATE_STRNCMP_COUNTER

	movdqa	(%edx), %xmm3
	pxor	%xmm0, %xmm0
	mov	$16, %ecx
	orl	$12, FLAGS
	lea	12(%edx), %edi
	and	$0xfff, %edi
	sub	$0x1000, %edi

	.p2align 4
L(loop_ashr_12):
	add	$16, %edi
	jg	L(nibble_ashr_12)

L(gobble_ashr_12):
	movdqa	(%eax, %ecx), %xmm1
	movdqa	(%edx, %ecx), %xmm2
	movdqa	%xmm2, %xmm4

	palignr	$12, %xmm3, %xmm2
	TOLOWER (%xmm1, %xmm2)

	pcmpeqb	%xmm1, %xmm0
	pcmpeqb	%xmm2, %xmm1
	psubb	%xmm0, %xmm1
	pmovmskb %xmm1, %esi
	sub	$0xffff, %esi
	jnz	L(exit)

#if defined USE_AS_STRNCMP || defined USE_AS_STRNCASECMP_L
	cmp	$16, REM
	lea	-16(REM), REM
	jbe	L(more8byteseq)
#endif

	add	$16, %ecx
	movdqa	%xmm4, %xmm3

	add	$16, %edi
	jg	L(nibble_ashr_12)

	movdqa	(%eax, %ecx), %xmm1
	movdqa	(%edx, %ecx), %xmm2
	movdqa	%xmm2, %xmm4

	palignr	$12, %xmm3, %xmm2
	TOLOWER (%xmm1, %xmm2)

	pcmpeqb	%xmm1, %xmm0
	pcmpeqb	%xmm2, %xmm1
	psubb	%xmm0, %xmm1
	pmovmskb %xmm1, %esi
	sub	$0xffff, %esi
	jnz	L(exit)

#if defined USE_AS_STRNCMP || defined USE_AS_STRNCASECMP_L
	cmp	$16, REM
	lea	-16(REM), REM
	jbe	L(more8byteseq)
#endif
	add	$16, %ecx
	movdqa	%xmm4, %xmm3
	jmp	L(loop_ashr_12)

	.p2align 4
L(nibble_ashr_12):
	pcmpeqb	%xmm3, %xmm0
	pmovmskb %xmm0, %esi
	test	$0xf000, %esi
	jnz	L(ashr_12_exittail)

#if defined USE_AS_STRNCMP || defined USE_AS_STRNCASECMP_L
	cmp	$4, REM
	jbe	L(ashr_12_exittail)
#endif
	pxor	%xmm0, %xmm0
	sub	$0x1000, %edi
	jmp	L(gobble_ashr_12)

	.p2align 4
L(ashr_12_exittail):
	movdqa	(%eax, %ecx), %xmm1
	psrldq	$12, %xmm0
	psrldq	$12, %xmm3
	jmp	L(aftertail)

/*
 * The following cases will be handled by ashr_13
 * ecx(offset of esi)  eax(offset of edi)   relative offset	corresponding case
 *        n(3~15)            n - 3            12(15 +(n-3) - n)         ashr_13
 */
	.p2align 4
L(ashr_13):
	mov	$0xffff, %esi
	pxor	%xmm0, %xmm0
	movdqa	(%edx), %xmm2
	movdqa	(%eax), %xmm1
	pcmpeqb	%xmm1, %xmm0
	pslldq	$3, %xmm2
	TOLOWER (%xmm1, %xmm2)
	pcmpeqb	%xmm1, %xmm2
	psubb	%xmm0, %xmm2
	pmovmskb %xmm2, %edi
	shr	%cl, %esi
	shr	%cl, %edi
	sub	%edi, %esi
	lea	-3(%ecx), %edi
	jnz	L(less32bytes)

	UPDATE_STRNCMP_COUNTER

	movdqa	(%edx), %xmm3
	pxor	%xmm0, %xmm0
	mov	$16, %ecx
	orl	$13, FLAGS
	lea	13(%edx), %edi
	and	$0xfff, %edi
	sub	$0x1000, %edi

	.p2align 4
L(loop_ashr_13):
	add	$16, %edi
	jg	L(nibble_ashr_13)

L(gobble_ashr_13):
	movdqa	(%eax, %ecx), %xmm1
	movdqa	(%edx, %ecx), %xmm2
	movdqa	%xmm2, %xmm4

	palignr	$13, %xmm3, %xmm2
	TOLOWER (%xmm1, %xmm2)

	pcmpeqb	%xmm1, %xmm0
	pcmpeqb	%xmm2, %xmm1
	psubb	%xmm0, %xmm1
	pmovmskb %xmm1, %esi
	sub	$0xffff, %esi
	jnz	L(exit)

#if defined USE_AS_STRNCMP || defined USE_AS_STRNCASECMP_L
	cmp	$16, REM
	lea	-16(REM), REM
	jbe	L(more8byteseq)
#endif
	add	$16, %ecx
	movdqa	%xmm4, %xmm3

	add	$16, %edi
	jg	L(nibble_ashr_13)

	movdqa	(%eax, %ecx), %xmm1
	movdqa	(%edx, %ecx), %xmm2
	movdqa	%xmm2, %xmm4

	palignr	$13, %xmm3, %xmm2
	TOLOWER (%xmm1, %xmm2)

	pcmpeqb	%xmm1, %xmm0
	pcmpeqb	%xmm2, %xmm1
	psubb	%xmm0, %xmm1
	pmovmskb %xmm1, %esi
	sub	$0xffff, %esi
	jnz	L(exit)

#if defined USE_AS_STRNCMP || defined USE_AS_STRNCASECMP_L
	cmp	$16, REM
	lea	-16(REM), REM
	jbe	L(more8byteseq)
#endif
	add	$16, %ecx
	movdqa	%xmm4, %xmm3
	jmp	L(loop_ashr_13)

	.p2align 4
L(nibble_ashr_13):
	pcmpeqb	%xmm3, %xmm0
	pmovmskb %xmm0, %esi
	test	$0xe000, %esi
	jnz	L(ashr_13_exittail)

#if defined USE_AS_STRNCMP || defined USE_AS_STRNCASECMP_L
	cmp	$3, REM
	jbe	L(ashr_13_exittail)
#endif
	pxor	%xmm0, %xmm0
	sub	$0x1000, %edi
	jmp	L(gobble_ashr_13)

	.p2align 4
L(ashr_13_exittail):
	movdqa	(%eax, %ecx), %xmm1
	psrldq	$13, %xmm0
	psrldq	$13, %xmm3
	jmp	L(aftertail)

/*
 * The following cases will be handled by ashr_14
 * ecx(offset of esi)  eax(offset of edi)   relative offset	corresponding case
 *        n(2~15)            n - 2            13(15 +(n-2) - n)         ashr_14
 */
	.p2align 4
L(ashr_14):
	mov	$0xffff, %esi
	pxor	%xmm0, %xmm0
	movdqa	(%edx), %xmm2
	movdqa	(%eax), %xmm1
	pcmpeqb	%xmm1, %xmm0
	pslldq	$2, %xmm2
	TOLOWER (%xmm1, %xmm2)
	pcmpeqb	%xmm1, %xmm2
	psubb	%xmm0, %xmm2
	pmovmskb %xmm2, %edi
	shr	%cl, %esi
	shr	%cl, %edi
	sub	%edi, %esi
	lea	-2(%ecx), %edi
	jnz	L(less32bytes)

	UPDATE_STRNCMP_COUNTER

	movdqa	(%edx), %xmm3
	pxor	%xmm0, %xmm0
	mov	$16, %ecx
	orl	$14, FLAGS
	lea	14(%edx), %edi
	and	$0xfff, %edi
	sub	$0x1000, %edi

	.p2align 4
L(loop_ashr_14):
	add	$16, %edi
	jg	L(nibble_ashr_14)

L(gobble_ashr_14):
	movdqa	(%eax, %ecx), %xmm1
	movdqa	(%edx, %ecx), %xmm2
	movdqa	%xmm2, %xmm4

	palignr	$14, %xmm3, %xmm2
	TOLOWER (%xmm1, %xmm2)

	pcmpeqb	%xmm1, %xmm0
	pcmpeqb	%xmm2, %xmm1
	psubb	%xmm0, %xmm1
	pmovmskb %xmm1, %esi
	sub	$0xffff, %esi
	jnz	L(exit)

#if defined USE_AS_STRNCMP || defined USE_AS_STRNCASECMP_L
	cmp	$16, REM
	lea	-16(REM), REM
	jbe	L(more8byteseq)
#endif
	add	$16, %ecx
	movdqa	%xmm4, %xmm3

	add	$16, %edi
	jg	L(nibble_ashr_14)

	movdqa	(%eax, %ecx), %xmm1
	movdqa	(%edx, %ecx), %xmm2
	movdqa	%xmm2, %xmm4

	palignr	$14, %xmm3, %xmm2
	TOLOWER (%xmm1, %xmm2)

	pcmpeqb	%xmm1, %xmm0
	pcmpeqb	%xmm2, %xmm1
	psubb	%xmm0, %xmm1
	pmovmskb %xmm1, %esi
	sub	$0xffff, %esi
	jnz	L(exit)

#if defined USE_AS_STRNCMP || defined USE_AS_STRNCASECMP_L
	cmp	$16, REM
	lea	-16(REM), REM
	jbe	L(more8byteseq)
#endif
	add	$16, %ecx
	movdqa	%xmm4, %xmm3
	jmp	L(loop_ashr_14)

	.p2align 4
L(nibble_ashr_14):
	pcmpeqb	%xmm3, %xmm0
	pmovmskb %xmm0, %esi
	test	$0xc000, %esi
	jnz	L(ashr_14_exittail)

#if defined USE_AS_STRNCMP || defined USE_AS_STRNCASECMP_L
	cmp	$2, REM
	jbe	L(ashr_14_exittail)
#endif
	pxor	%xmm0, %xmm0
	sub	$0x1000, %edi
	jmp	L(gobble_ashr_14)

	.p2align 4
L(ashr_14_exittail):
	movdqa	(%eax, %ecx), %xmm1
	psrldq	$14, %xmm0
	psrldq	$14, %xmm3
	jmp	L(aftertail)

/*
 * The following cases will be handled by ashr_14
 * ecx(offset of esi)  eax(offset of edi)   relative offset	corresponding case
 *        n(1~15)            n - 1            14(15 +(n-1) - n)         ashr_15
 */

	.p2align 4
L(ashr_15):
	mov	$0xffff, %esi
	pxor	%xmm0, %xmm0
	movdqa	(%edx), %xmm2
	movdqa	(%eax), %xmm1
	pcmpeqb	%xmm1, %xmm0
	pslldq	$1, %xmm2
	TOLOWER (%xmm1, %xmm2)
	pcmpeqb	%xmm1, %xmm2
	psubb	%xmm0, %xmm2
	pmovmskb %xmm2, %edi
	shr	%cl, %esi
	shr	%cl, %edi
	sub	%edi, %esi
	lea	-1(%ecx), %edi
	jnz	L(less32bytes)

	UPDATE_STRNCMP_COUNTER

	movdqa	(%edx), %xmm3
	pxor	%xmm0, %xmm0
	mov	$16, %ecx
	orl	$15, FLAGS
	lea	15(%edx), %edi
	and	$0xfff, %edi
	sub	$0x1000, %edi

	.p2align 4
L(loop_ashr_15):
	add	$16, %edi
	jg	L(nibble_ashr_15)

L(gobble_ashr_15):
	movdqa	(%eax, %ecx), %xmm1
	movdqa	(%edx, %ecx), %xmm2
	movdqa	%xmm2, %xmm4

	palignr	$15, %xmm3, %xmm2
	TOLOWER (%xmm1, %xmm2)

	pcmpeqb	%xmm1, %xmm0
	pcmpeqb	%xmm2, %xmm1
	psubb	%xmm0, %xmm1
	pmovmskb %xmm1, %esi
	sub	$0xffff, %esi
	jnz	L(exit)

#if defined USE_AS_STRNCMP || defined USE_AS_STRNCASECMP_L
	cmp	$16, REM
	lea	-16(REM), REM
	jbe	L(more8byteseq)
#endif
	add	$16, %ecx
	movdqa	%xmm4, %xmm3

	add	$16, %edi
	jg	L(nibble_ashr_15)

	movdqa	(%eax, %ecx), %xmm1
	movdqa	(%edx, %ecx), %xmm2
	movdqa	%xmm2, %xmm4

	palignr	$15, %xmm3, %xmm2
	TOLOWER (%xmm1, %xmm2)

	pcmpeqb	%xmm1, %xmm0
	pcmpeqb	%xmm2, %xmm1
	psubb	%xmm0, %xmm1
	pmovmskb %xmm1, %esi
	sub	$0xffff, %esi
	jnz	L(exit)

#if defined USE_AS_STRNCMP || defined USE_AS_STRNCASECMP_L
	cmp	$16, REM
	lea	-16(REM), REM
	jbe	L(more8byteseq)
#endif
	add	$16, %ecx
	movdqa	%xmm4, %xmm3
	jmp	L(loop_ashr_15)

	.p2align 4
L(nibble_ashr_15):
	pcmpeqb	%xmm3, %xmm0
	pmovmskb %xmm0, %esi
	test	$0x8000, %esi
	jnz	L(ashr_15_exittail)

#if defined USE_AS_STRNCMP || defined USE_AS_STRNCASECMP_L
	cmp	$1, REM
	jbe	L(ashr_15_exittail)
#endif
	pxor	%xmm0, %xmm0
	sub	$0x1000, %edi
	jmp	L(gobble_ashr_15)

	.p2align 4
L(ashr_15_exittail):
	movdqa	(%eax, %ecx), %xmm1
	psrldq	$15, %xmm0
	psrldq	$15, %xmm3
	jmp	L(aftertail)

	.p2align 4
L(aftertail):
	TOLOWER (%xmm1, %xmm3)
	pcmpeqb	%xmm3, %xmm1
	psubb	%xmm0, %xmm1
	pmovmskb %xmm1, %esi
	not	%esi
L(exit):
	mov	FLAGS, %edi
	and	$0x1f, %edi
	lea	-16(%edi, %ecx), %edi
L(less32bytes):
	add	%edi, %edx
	add	%ecx, %eax
	testl	$0x20, FLAGS
	jz	L(ret2)
	xchg	%eax, %edx

	.p2align 4
L(ret2):
	mov	%esi, %ecx
#if defined USE_AS_STRCASECMP_L || defined USE_AS_STRNCASECMP_L
	addl	$4, %esp
	cfi_adjust_cfa_offset (-4)
#endif
	POP	(%esi)
	POP	(%edi)
#if !defined USE_AS_STRCASECMP_L && !defined USE_AS_STRNCASECMP_L
	POP	(FLAGS)
#endif
L(less16bytes):
	test	%cl, %cl
	jz	L(2next_8_bytes)

	test	$0x01, %cl
	jnz	L(Byte0)

	test	$0x02, %cl
	jnz	L(Byte1)

	test	$0x04, %cl
	jnz	L(Byte2)

	test	$0x08, %cl
	jnz	L(Byte3)

	test	$0x10, %cl
	jnz	L(Byte4)

	test	$0x20, %cl
	jnz	L(Byte5)

	test	$0x40, %cl
	jnz	L(Byte6)
#if defined USE_AS_STRNCMP || defined USE_AS_STRNCASECMP_L
	cmp	$7, REM
	jbe	L(eq)
#endif

	movzx	7(%eax), %ecx
	movzx	7(%edx), %eax
#if defined USE_AS_STRCASECMP_L || defined USE_AS_STRNCASECMP_L
# ifdef PIC
	movl	_nl_C_LC_CTYPE_tolower@GOTOFF+128*4(%ebx,%ecx,4), %ecx
	movl	_nl_C_LC_CTYPE_tolower@GOTOFF+128*4(%ebx,%eax,4), %eax
# else
	movl	_nl_C_LC_CTYPE_tolower+128*4(,%ecx,4), %ecx
	movl	_nl_C_LC_CTYPE_tolower+128*4(,%eax,4), %eax
# endif
#endif

	sub	%ecx, %eax
	RETURN

L(Byte0):
#if defined USE_AS_STRNCMP || defined USE_AS_STRNCASECMP_L
	cmp	$0, REM
	jbe	L(eq)
#endif
	movzx	(%eax), %ecx
	movzx	(%edx), %eax

#if defined USE_AS_STRCASECMP_L || defined USE_AS_STRNCASECMP_L
# ifdef PIC
	movl	_nl_C_LC_CTYPE_tolower@GOTOFF+128*4(%ebx,%ecx,4), %ecx
	movl	_nl_C_LC_CTYPE_tolower@GOTOFF+128*4(%ebx,%eax,4), %eax
# else
	movl	_nl_C_LC_CTYPE_tolower+128*4(,%ecx,4), %ecx
	movl	_nl_C_LC_CTYPE_tolower+128*4(,%eax,4), %eax
# endif
#endif

	sub	%ecx, %eax
	RETURN

L(Byte1):
#if defined USE_AS_STRNCMP || defined USE_AS_STRNCASECMP_L
	cmp	$1, REM
	jbe	L(eq)
#endif
	movzx	1(%eax), %ecx
	movzx	1(%edx), %eax

#if defined USE_AS_STRCASECMP_L || defined USE_AS_STRNCASECMP_L
# ifdef PIC
	movl	_nl_C_LC_CTYPE_tolower@GOTOFF+128*4(%ebx,%ecx,4), %ecx
	movl	_nl_C_LC_CTYPE_tolower@GOTOFF+128*4(%ebx,%eax,4), %eax
# else
	movl	_nl_C_LC_CTYPE_tolower+128*4(,%ecx,4), %ecx
	movl	_nl_C_LC_CTYPE_tolower+128*4(,%eax,4), %eax
# endif
#endif

	sub	%ecx, %eax
	RETURN

L(Byte2):
#if defined USE_AS_STRNCMP || defined USE_AS_STRNCASECMP_L
	cmp	$2, REM
	jbe	L(eq)
#endif
	movzx	2(%eax), %ecx
	movzx	2(%edx), %eax

#if defined USE_AS_STRCASECMP_L || defined USE_AS_STRNCASECMP_L
# ifdef PIC
	movl	_nl_C_LC_CTYPE_tolower@GOTOFF+128*4(%ebx,%ecx,4), %ecx
	movl	_nl_C_LC_CTYPE_tolower@GOTOFF+128*4(%ebx,%eax,4), %eax
# else
	movl	_nl_C_LC_CTYPE_tolower+128*4(,%ecx,4), %ecx
	movl	_nl_C_LC_CTYPE_tolower+128*4(,%eax,4), %eax
# endif
#endif

	sub	%ecx, %eax
	RETURN

L(Byte3):
#if defined USE_AS_STRNCMP || defined USE_AS_STRNCASECMP_L
	cmp	$3, REM
	jbe	L(eq)
#endif
	movzx	3(%eax), %ecx
	movzx	3(%edx), %eax

#if defined USE_AS_STRCASECMP_L || defined USE_AS_STRNCASECMP_L
# ifdef PIC
	movl	_nl_C_LC_CTYPE_tolower@GOTOFF+128*4(%ebx,%ecx,4), %ecx
	movl	_nl_C_LC_CTYPE_tolower@GOTOFF+128*4(%ebx,%eax,4), %eax
# else
	movl	_nl_C_LC_CTYPE_tolower+128*4(,%ecx,4), %ecx
	movl	_nl_C_LC_CTYPE_tolower+128*4(,%eax,4), %eax
# endif
#endif

	sub	%ecx, %eax
	RETURN

L(Byte4):
#if defined USE_AS_STRNCMP || defined USE_AS_STRNCASECMP_L
	cmp	$4, REM
	jbe	L(eq)
#endif
	movzx	4(%eax), %ecx
	movzx	4(%edx), %eax

#if defined USE_AS_STRCASECMP_L || defined USE_AS_STRNCASECMP_L
# ifdef PIC
	movl	_nl_C_LC_CTYPE_tolower@GOTOFF+128*4(%ebx,%ecx,4), %ecx
	movl	_nl_C_LC_CTYPE_tolower@GOTOFF+128*4(%ebx,%eax,4), %eax
# else
	movl	_nl_C_LC_CTYPE_tolower+128*4(,%ecx,4), %ecx
	movl	_nl_C_LC_CTYPE_tolower+128*4(,%eax,4), %eax
# endif
#endif

	sub	%ecx, %eax
	RETURN

L(Byte5):
#if defined USE_AS_STRNCMP || defined USE_AS_STRNCASECMP_L
	cmp	$5, REM
	jbe	L(eq)
#endif
	movzx	5(%eax), %ecx
	movzx	5(%edx), %eax

#if defined USE_AS_STRCASECMP_L || defined USE_AS_STRNCASECMP_L
# ifdef PIC
	movl	_nl_C_LC_CTYPE_tolower@GOTOFF+128*4(%ebx,%ecx,4), %ecx
	movl	_nl_C_LC_CTYPE_tolower@GOTOFF+128*4(%ebx,%eax,4), %eax
# else
	movl	_nl_C_LC_CTYPE_tolower+128*4(,%ecx,4), %ecx
	movl	_nl_C_LC_CTYPE_tolower+128*4(,%eax,4), %eax
# endif
#endif

	sub	%ecx, %eax
	RETURN

L(Byte6):
#if defined USE_AS_STRNCMP || defined USE_AS_STRNCASECMP_L
	cmp	$6, REM
	jbe	L(eq)
#endif
	movzx	6(%eax), %ecx
	movzx	6(%edx), %eax

#if defined USE_AS_STRCASECMP_L || defined USE_AS_STRNCASECMP_L
# ifdef PIC
	movl	_nl_C_LC_CTYPE_tolower@GOTOFF+128*4(%ebx,%ecx,4), %ecx
	movl	_nl_C_LC_CTYPE_tolower@GOTOFF+128*4(%ebx,%eax,4), %eax
# else
	movl	_nl_C_LC_CTYPE_tolower+128*4(,%ecx,4), %ecx
	movl	_nl_C_LC_CTYPE_tolower+128*4(,%eax,4), %eax
# endif
#endif

	sub	%ecx, %eax
	RETURN

L(2next_8_bytes):
	add	$8, %eax
	add	$8, %edx
#if defined USE_AS_STRNCMP || defined USE_AS_STRNCASECMP_L
	cmp	$8, REM
	lea	-8(REM), REM
	jbe	L(eq)
#endif

	test	$0x01, %ch
	jnz	L(Byte0)

	test	$0x02, %ch
	jnz	L(Byte1)

	test	$0x04, %ch
	jnz	L(Byte2)

	test	$0x08, %ch
	jnz	L(Byte3)

	test	$0x10, %ch
	jnz	L(Byte4)

	test	$0x20, %ch
	jnz	L(Byte5)

	test	$0x40, %ch
	jnz	L(Byte6)

#if defined USE_AS_STRNCMP || defined USE_AS_STRNCASECMP_L
	cmp	$7, REM
	jbe	L(eq)
#endif
	movzx	7(%eax), %ecx
	movzx	7(%edx), %eax

#if defined USE_AS_STRCASECMP_L || defined USE_AS_STRNCASECMP_L
# ifdef PIC
	movl	_nl_C_LC_CTYPE_tolower@GOTOFF+128*4(%ebx,%ecx,4), %ecx
	movl	_nl_C_LC_CTYPE_tolower@GOTOFF+128*4(%ebx,%eax,4), %eax
# else
	movl	_nl_C_LC_CTYPE_tolower+128*4(,%ecx,4), %ecx
	movl	_nl_C_LC_CTYPE_tolower+128*4(,%eax,4), %eax
# endif
#endif

	sub	%ecx, %eax
	RETURN

#ifdef USE_AS_STRNCMP
L(neq_sncmp):
#endif
L(neq):
	mov	$1, %eax
	ja	L(neq_bigger)
	neg	%eax
L(neq_bigger):
#if defined USE_AS_STRCASECMP_L || defined USE_AS_STRNCASECMP_L
	addl	$4, %esp
	cfi_adjust_cfa_offset (-4)
#endif
#if defined USE_AS_STRNCMP || defined USE_AS_STRNCASECMP_L
	POP	(REM)
#endif
#if defined USE_AS_STRCASECMP_L || defined USE_AS_STRNCASECMP_L
# ifdef PIC
	POP	(%ebx)
# endif
#endif
	ret

#if defined USE_AS_STRNCMP || defined USE_AS_STRNCASECMP_L
	.p2align 4
	cfi_restore_state
L(more8byteseq):

# ifdef USE_AS_STRNCASECMP_L
	addl	$4, %esp
	cfi_adjust_cfa_offset (-4)
# endif
	POP	(%esi)
	POP	(%edi)
# ifdef USE_AS_STRNCMP
	POP	(FLAGS)
# endif
#endif

#ifdef USE_AS_STRNCMP
L(eq_sncmp):
#endif
L(eq):

#if defined USE_AS_STRNCMP || defined USE_AS_STRNCASECMP_L
	POP	(REM)
#endif
#if defined USE_AS_STRCASECMP_L || defined USE_AS_STRNCASECMP_L
# ifdef PIC
	POP	(%ebx)
# endif
#endif
	xorl	%eax, %eax
	ret

#if defined USE_AS_STRNCMP || defined USE_AS_STRNCASECMP_L
	.p2align 4
# if defined USE_AS_STRNCASECMP_L && defined PIC
	CFI_PUSH (%ebx)
# endif
	CFI_PUSH (REM)
L(less16bytes_sncmp):
# ifdef USE_AS_STRNCASECMP_L
	PUSH	(%esi)
# endif
	test	REM, REM
	jz	L(eq_sncmp)

	movzbl	(%eax), %ecx
# ifdef USE_AS_STRNCASECMP_L
	movzbl	(%edx), %esi
#  ifdef PIC
	movl	_nl_C_LC_CTYPE_tolower@GOTOFF+128*4(%ebx,%ecx,4), %ecx
	movl	_nl_C_LC_CTYPE_tolower@GOTOFF+128*4(%ebx,%esi,4), %esi
#  else
	movl	_nl_C_LC_CTYPE_tolower+128*4(,%ecx,4), %ecx
	movl	_nl_C_LC_CTYPE_tolower+128*4(,%esi,4), %esi
#  endif
	cmpl	%ecx, %esi
# else
	cmpb	%cl, (%edx)
# endif
	jne	L(neq_sncmp)
	test	%cl, %cl
	je	L(eq_sncmp)

	cmp	$1, REM
	je	L(eq_sncmp)

	movzbl	1(%eax), %ecx
# ifdef USE_AS_STRNCASECMP_L
	movzbl	1(%edx), %esi
#  ifdef PIC
	movl	_nl_C_LC_CTYPE_tolower@GOTOFF+128*4(%ebx,%ecx,4), %ecx
	movl	_nl_C_LC_CTYPE_tolower@GOTOFF+128*4(%ebx,%esi,4), %esi
#  else
	movl	_nl_C_LC_CTYPE_tolower+128*4(,%ecx,4), %ecx
	movl	_nl_C_LC_CTYPE_tolower+128*4(,%esi,4), %esi
#  endif
	cmpl	%ecx, %esi
# else
	cmpb	%cl, 1(%edx)
# endif
	jne	L(neq_sncmp)
	test	%cl, %cl
	je	L(eq_sncmp)

	cmp	$2, REM
	je	L(eq_sncmp)

	movzbl	2(%eax), %ecx
# ifdef USE_AS_STRNCASECMP_L
	movzbl	2(%edx), %esi
#  ifdef PIC
	movl	_nl_C_LC_CTYPE_tolower@GOTOFF+128*4(%ebx,%ecx,4), %ecx
	movl	_nl_C_LC_CTYPE_tolower@GOTOFF+128*4(%ebx,%esi,4), %esi
#  else
	movl	_nl_C_LC_CTYPE_tolower+128*4(,%ecx,4), %ecx
	movl	_nl_C_LC_CTYPE_tolower+128*4(,%esi,4), %esi
#  endif
	cmpl	%ecx, %esi
# else
	cmpb	%cl, 2(%edx)
# endif
	jne	L(neq_sncmp)
	test	%cl, %cl
	je	L(eq_sncmp)

	cmp	$3, REM
	je	L(eq_sncmp)

	movzbl	3(%eax), %ecx
# ifdef USE_AS_STRNCASECMP_L
	movzbl	3(%edx), %esi
#  ifdef PIC
	movl	_nl_C_LC_CTYPE_tolower@GOTOFF+128*4(%ebx,%ecx,4), %ecx
	movl	_nl_C_LC_CTYPE_tolower@GOTOFF+128*4(%ebx,%esi,4), %esi
#  else
	movl	_nl_C_LC_CTYPE_tolower+128*4(,%ecx,4), %ecx
	movl	_nl_C_LC_CTYPE_tolower+128*4(,%esi,4), %esi
#  endif
	cmpl	%ecx, %esi
# else
	cmpb	%cl, 3(%edx)
# endif
	jne	L(neq_sncmp)
	test	%cl, %cl
	je	L(eq_sncmp)

	cmp	$4, REM
	je	L(eq_sncmp)

	movzbl	4(%eax), %ecx
# ifdef USE_AS_STRNCASECMP_L
	movzbl	4(%edx), %esi
#  ifdef PIC
	movl	_nl_C_LC_CTYPE_tolower@GOTOFF+128*4(%ebx,%ecx,4), %ecx
	movl	_nl_C_LC_CTYPE_tolower@GOTOFF+128*4(%ebx,%esi,4), %esi
#  else
	movl	_nl_C_LC_CTYPE_tolower+128*4(,%ecx,4), %ecx
	movl	_nl_C_LC_CTYPE_tolower+128*4(,%esi,4), %esi
#  endif
	cmpl	%ecx, %esi
# else
	cmpb	%cl, 4(%edx)
# endif
	jne	L(neq_sncmp)
	test	%cl, %cl
	je	L(eq_sncmp)

	cmp	$5, REM
	je	L(eq_sncmp)

	movzbl	5(%eax), %ecx
# ifdef USE_AS_STRNCASECMP_L
	movzbl	5(%edx), %esi
#  ifdef PIC
	movl	_nl_C_LC_CTYPE_tolower@GOTOFF+128*4(%ebx,%ecx,4), %ecx
	movl	_nl_C_LC_CTYPE_tolower@GOTOFF+128*4(%ebx,%esi,4), %esi
#  else
	movl	_nl_C_LC_CTYPE_tolower+128*4(,%ecx,4), %ecx
	movl	_nl_C_LC_CTYPE_tolower+128*4(,%esi,4), %esi
#  endif
	cmpl	%ecx, %esi
# else
	cmpb	%cl, 5(%edx)
# endif
	jne	L(neq_sncmp)
	test	%cl, %cl
	je	L(eq_sncmp)

	cmp	$6, REM
	je	L(eq_sncmp)

	movzbl	6(%eax), %ecx
# ifdef USE_AS_STRNCASECMP_L
	movzbl	6(%edx), %esi
#  ifdef PIC
	movl	_nl_C_LC_CTYPE_tolower@GOTOFF+128*4(%ebx,%ecx,4), %ecx
	movl	_nl_C_LC_CTYPE_tolower@GOTOFF+128*4(%ebx,%esi,4), %esi
#  else
	movl	_nl_C_LC_CTYPE_tolower+128*4(,%ecx,4), %ecx
	movl	_nl_C_LC_CTYPE_tolower+128*4(,%esi,4), %esi
#  endif
	cmpl	%ecx, %esi
# else
	cmpb	%cl, 6(%edx)
# endif
	jne	L(neq_sncmp)
	test	%cl, %cl
	je	L(eq_sncmp)

	cmp	$7, REM
	je	L(eq_sncmp)

	movzbl	7(%eax), %ecx
# ifdef USE_AS_STRNCASECMP_L
	movzbl	7(%edx), %esi
#  ifdef PIC
	movl	_nl_C_LC_CTYPE_tolower@GOTOFF+128*4(%ebx,%ecx,4), %ecx
	movl	_nl_C_LC_CTYPE_tolower@GOTOFF+128*4(%ebx,%esi,4), %esi
#  else
	movl	_nl_C_LC_CTYPE_tolower+128*4(,%ecx,4), %ecx
	movl	_nl_C_LC_CTYPE_tolower+128*4(,%esi,4), %esi
#  endif
	cmpl	%ecx, %esi
# else
	cmpb	%cl, 7(%edx)
# endif
	jne	L(neq_sncmp)
	test	%cl, %cl
	je	L(eq_sncmp)


	cmp	$8, REM
	je	L(eq_sncmp)

	movzbl	8(%eax), %ecx
# ifdef USE_AS_STRNCASECMP_L
	movzbl	8(%edx), %esi
#  ifdef PIC
	movl	_nl_C_LC_CTYPE_tolower@GOTOFF+128*4(%ebx,%ecx,4), %ecx
	movl	_nl_C_LC_CTYPE_tolower@GOTOFF+128*4(%ebx,%esi,4), %esi
#  else
	movl	_nl_C_LC_CTYPE_tolower+128*4(,%ecx,4), %ecx
	movl	_nl_C_LC_CTYPE_tolower+128*4(,%esi,4), %esi
#  endif
	cmpl	%ecx, %esi
# else
	cmpb	%cl, 8(%edx)
# endif
	jne	L(neq_sncmp)
	test	%cl, %cl
	je	L(eq_sncmp)

	cmp	$9, REM
	je	L(eq_sncmp)

	movzbl	9(%eax), %ecx
# ifdef USE_AS_STRNCASECMP_L
	movzbl	9(%edx), %esi
#  ifdef PIC
	movl	_nl_C_LC_CTYPE_tolower@GOTOFF+128*4(%ebx,%ecx,4), %ecx
	movl	_nl_C_LC_CTYPE_tolower@GOTOFF+128*4(%ebx,%esi,4), %esi
#  else
	movl	_nl_C_LC_CTYPE_tolower+128*4(,%ecx,4), %ecx
	movl	_nl_C_LC_CTYPE_tolower+128*4(,%esi,4), %esi
#  endif
	cmpl	%ecx, %esi
# else
	cmpb	%cl, 9(%edx)
# endif
	jne	L(neq_sncmp)
	test	%cl, %cl
	je	L(eq_sncmp)

	cmp	$10, REM
	je	L(eq_sncmp)

	movzbl	10(%eax), %ecx
# ifdef USE_AS_STRNCASECMP_L
	movzbl	10(%edx), %esi
#  ifdef PIC
	movl	_nl_C_LC_CTYPE_tolower@GOTOFF+128*4(%ebx,%ecx,4), %ecx
	movl	_nl_C_LC_CTYPE_tolower@GOTOFF+128*4(%ebx,%esi,4), %esi
#  else
	movl	_nl_C_LC_CTYPE_tolower+128*4(,%ecx,4), %ecx
	movl	_nl_C_LC_CTYPE_tolower+128*4(,%esi,4), %esi
#  endif
	cmpl	%ecx, %esi
# else
	cmpb	%cl, 10(%edx)
# endif
	jne	L(neq_sncmp)
	test	%cl, %cl
	je	L(eq_sncmp)

	cmp	$11, REM
	je	L(eq_sncmp)

	movzbl	11(%eax), %ecx
# ifdef USE_AS_STRNCASECMP_L
	movzbl	11(%edx), %esi
#  ifdef PIC
	movl	_nl_C_LC_CTYPE_tolower@GOTOFF+128*4(%ebx,%ecx,4), %ecx
	movl	_nl_C_LC_CTYPE_tolower@GOTOFF+128*4(%ebx,%esi,4), %esi
#  else
	movl	_nl_C_LC_CTYPE_tolower+128*4(,%ecx,4), %ecx
	movl	_nl_C_LC_CTYPE_tolower+128*4(,%esi,4), %esi
#  endif
	cmpl	%ecx, %esi
# else
	cmpb	%cl, 11(%edx)
# endif
	jne	L(neq_sncmp)
	test	%cl, %cl
	je	L(eq_sncmp)


	cmp	$12, REM
	je	L(eq_sncmp)

	movzbl	12(%eax), %ecx
# ifdef USE_AS_STRNCASECMP_L
	movzbl	12(%edx), %esi
#  ifdef PIC
	movl	_nl_C_LC_CTYPE_tolower@GOTOFF+128*4(%ebx,%ecx,4), %ecx
	movl	_nl_C_LC_CTYPE_tolower@GOTOFF+128*4(%ebx,%esi,4), %esi
#  else
	movl	_nl_C_LC_CTYPE_tolower+128*4(,%ecx,4), %ecx
	movl	_nl_C_LC_CTYPE_tolower+128*4(,%esi,4), %esi
#  endif
	cmpl	%ecx, %esi
# else
	cmpb	%cl, 12(%edx)
# endif
	jne	L(neq_sncmp)
	test	%cl, %cl
	je	L(eq_sncmp)

	cmp	$13, REM
	je	L(eq_sncmp)

	movzbl	13(%eax), %ecx
# ifdef USE_AS_STRNCASECMP_L
	movzbl	13(%edx), %esi
#  ifdef PIC
	movl	_nl_C_LC_CTYPE_tolower@GOTOFF+128*4(%ebx,%ecx,4), %ecx
	movl	_nl_C_LC_CTYPE_tolower@GOTOFF+128*4(%ebx,%esi,4), %esi
#  else
	movl	_nl_C_LC_CTYPE_tolower+128*4(,%ecx,4), %ecx
	movl	_nl_C_LC_CTYPE_tolower+128*4(,%esi,4), %esi
#  endif
	cmpl	%ecx, %esi
# else
	cmpb	%cl, 13(%edx)
# endif
	jne	L(neq_sncmp)
	test	%cl, %cl
	je	L(eq_sncmp)

	cmp	$14, REM
	je	L(eq_sncmp)

	movzbl	14(%eax), %ecx
# ifdef USE_AS_STRNCASECMP_L
	movzbl	14(%edx), %esi
#  ifdef PIC
	movl	_nl_C_LC_CTYPE_tolower@GOTOFF+128*4(%ebx,%ecx,4), %ecx
	movl	_nl_C_LC_CTYPE_tolower@GOTOFF+128*4(%ebx,%esi,4), %esi
#  else
	movl	_nl_C_LC_CTYPE_tolower+128*4(,%ecx,4), %ecx
	movl	_nl_C_LC_CTYPE_tolower+128*4(,%esi,4), %esi
#  endif
	cmpl	%ecx, %esi
# else
	cmpb	%cl, 14(%edx)
# endif
	jne	L(neq_sncmp)
	test	%cl, %cl
	je	L(eq_sncmp)

	cmp	$15, REM
	je	L(eq_sncmp)

	movzbl	15(%eax), %ecx
# ifdef USE_AS_STRNCASECMP_L
	movzbl	15(%edx), %esi
#  ifdef PIC
	movl	_nl_C_LC_CTYPE_tolower@GOTOFF+128*4(%ebx,%ecx,4), %ecx
	movl	_nl_C_LC_CTYPE_tolower@GOTOFF+128*4(%ebx,%esi,4), %esi
#  else
	movl	_nl_C_LC_CTYPE_tolower+128*4(,%ecx,4), %ecx
	movl	_nl_C_LC_CTYPE_tolower+128*4(,%esi,4), %esi
#  endif
	cmpl	%ecx, %esi
# else
	cmpb	%cl, 15(%edx)
# endif
	jne	L(neq_sncmp)

# ifdef USE_AS_STRNCASECMP_L
L(eq_sncmp):
	POP	(%esi)
# endif
	POP	(REM)
# if defined USE_AS_STRNCASECMP_L && defined PIC
	POP	(%ebx)
# endif
	xor	%eax, %eax
	ret

# ifdef USE_AS_STRNCASECMP_L
	.p2align 4
#  ifdef PIC
	CFI_PUSH (%ebx)
#  endif
	CFI_PUSH (REM)
	CFI_PUSH (%esi)
L(neq_sncmp):
	mov	$1, %eax
	mov	$-1, %edx
	cmovna	%edx, %eax
	POP	(%esi)
	POP	(REM)
#  ifdef PIC
	POP	(%ebx)
#  endif
	ret
# endif
#endif

END (STRCMP)

#endif
