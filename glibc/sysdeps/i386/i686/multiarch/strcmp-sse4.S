/* strcmp with SSE4.2
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
#  define STRCMP	__strncmp_sse4_2
# endif
# define STR1		8
# define STR2		STR1+4
# define CNT		STR2+4
# define RETURN		POP (REM); ret; .p2align 4; CFI_PUSH (REM)
# define REM		%ebp
#elif defined USE_AS_STRCASECMP_L
# include "locale-defines.h"
# ifndef STRCMP
#  define STRCMP	__strcasecmp_l_sse4_2
# endif
# ifdef PIC
#  define STR1		12
# else
#  define STR1		8
# endif
# define STR2		STR1+4
# define LOCALE		12	/* Loaded before the adjustment.  */
# ifdef PIC
#  define RETURN	POP (%edi); POP (%ebx); ret; \
			.p2align 4; CFI_PUSH (%ebx); CFI_PUSH (%edi)
# else
#  define RETURN	POP (%edi); ret; .p2align 4; CFI_PUSH (%edi)
# endif
# define NONASCII	__strcasecmp_nonascii
#elif defined USE_AS_STRNCASECMP_L
# include "locale-defines.h"
# ifndef STRCMP
#  define STRCMP	__strncasecmp_l_sse4_2
# endif
# ifdef PIC
#  define STR1		16
# else
#  define STR1		12
# endif
# define STR2		STR1+4
# define CNT		STR2+4
# define LOCALE		16	/* Loaded before the adjustment.  */
# ifdef PIC
#  define RETURN	POP (%edi); POP (REM); POP (%ebx); ret; \
			.p2align 4; \
			CFI_PUSH (%ebx); CFI_PUSH (REM); CFI_PUSH (%edi)
# else
#  define RETURN	POP (%edi); POP (REM); ret; \
			.p2align 4; CFI_PUSH (REM); CFI_PUSH (%edi)
# endif
# define REM		%ebp
# define NONASCII	__strncasecmp_nonascii
#else
# ifndef STRCMP
#  define STRCMP	__strcmp_sse4_2
# endif
# define STR1		4
# define STR2		STR1+4
# define RETURN		ret; .p2align 4
#endif

	.section .text.sse4.2,"ax",@progbits

#ifdef USE_AS_STRCASECMP_L
ENTRY (__strcasecmp_sse4_2)
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
END (__strcasecmp_sse4_2)
#endif

#ifdef USE_AS_STRNCASECMP_L
ENTRY (__strncasecmp_sse4_2)
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
END (__strncasecmp_sse4_2)
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
#if defined USE_AS_STRCASECMP_L || defined USE_AS_STRNCASECMP_L
	PUSH	(%edi)
#endif
	mov	STR1(%esp), %edx
	mov	STR2(%esp), %eax
#if defined USE_AS_STRNCMP || defined USE_AS_STRNCASECMP_L
	movl	CNT(%esp), REM
	test	REM, REM
	je	L(eq)
#endif
	mov	%dx, %cx
	and	$0xfff, %cx
	cmp	$0xff0, %cx
	ja	L(first4bytes)
	movdqu	(%edx), %xmm2
	mov	%eax, %ecx
	and	$0xfff, %ecx
	cmp	$0xff0, %ecx
	ja	L(first4bytes)
#if defined USE_AS_STRCASECMP_L || defined USE_AS_STRNCASECMP_L
# define TOLOWER(reg1, reg2) \
	movdqa	reg1, %xmm3;						      \
	movdqa	UCHIGH_reg, %xmm4;					      \
	movdqa	reg2, %xmm5;						      \
	movdqa	UCHIGH_reg, %xmm6;					      \
	pcmpgtb	UCLOW_reg, %xmm3;					      \
	pcmpgtb	reg1, %xmm4;						      \
	pcmpgtb	UCLOW_reg, %xmm5;					      \
	pcmpgtb	reg2, %xmm6;						      \
	pand	%xmm4, %xmm3;						      \
	pand	%xmm6, %xmm5;						      \
	pand	LCQWORD_reg, %xmm3;					      \
	pand	LCQWORD_reg, %xmm5;					      \
	por	%xmm3, reg1;						      \
	por	%xmm5, reg2

	movdqu	(%eax), %xmm1
	TOLOWER (%xmm2, %xmm1)
	movd	%xmm2, %ecx
	movd	%xmm1, %edi
	movdqa	%xmm2, %xmm3
	movdqa	%xmm1, %xmm4
	cmpl	%edi, %ecx
#else
# define TOLOWER(reg1, reg)

	movd	%xmm2, %ecx
	cmp	(%eax), %ecx
#endif
	jne	L(less4bytes)
#if !defined USE_AS_STRCASECMP_L && !defined USE_AS_STRNCASECMP_L
	movdqu	(%eax), %xmm1
#endif
	pxor	%xmm2, %xmm1
	pxor	%xmm0, %xmm0
	ptest	%xmm1, %xmm0
	jnc	L(less16bytes)
	pcmpeqb	%xmm0, %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(less16bytes)

#if defined USE_AS_STRNCMP || defined USE_AS_STRNCASECMP_L
	sub	$16, REM
	jbe	L(eq)
#endif
	add	$16, %edx
	add	$16, %eax
L(first4bytes):
	movzbl	(%eax), %ecx
#if defined USE_AS_STRCASECMP_L || defined USE_AS_STRNCASECMP_L
	movzbl	(%edx), %edi
# ifdef PIC
	movl	_nl_C_LC_CTYPE_tolower@GOTOFF+128*4(%ebx,%ecx,4), %ecx
	movl	_nl_C_LC_CTYPE_tolower@GOTOFF+128*4(%ebx,%edi,4), %edi
# else
	movl	_nl_C_LC_CTYPE_tolower+128*4(,%ecx,4), %ecx
	movl	_nl_C_LC_CTYPE_tolower+128*4(,%edi,4), %edi
# endif
	cmpl	%ecx, %edi
#else
	cmpb	%cl, (%edx)
#endif
	jne	L(neq)
	cmpl	$0, %ecx
	je	L(eq)

#if defined USE_AS_STRNCMP || defined USE_AS_STRNCASECMP_L
	cmp	$1, REM
	je	L(eq)
#endif

	movzbl	1(%eax), %ecx
#if defined USE_AS_STRCASECMP_L || defined USE_AS_STRNCASECMP_L
	movzbl	1(%edx), %edi
# ifdef PIC
	movl	_nl_C_LC_CTYPE_tolower@GOTOFF+128*4(%ebx,%ecx,4), %ecx
	movl	_nl_C_LC_CTYPE_tolower@GOTOFF+128*4(%ebx,%edi,4), %edi
# else
	movl	_nl_C_LC_CTYPE_tolower+128*4(,%ecx,4), %ecx
	movl	_nl_C_LC_CTYPE_tolower+128*4(,%edi,4), %edi
# endif
	cmpl	%ecx, %edi
#else
	cmpb	%cl, 1(%edx)
#endif
	jne	L(neq)
	cmpl	$0, %ecx
	je	L(eq)

#if defined USE_AS_STRNCMP || defined USE_AS_STRNCASECMP_L
	cmp	$2, REM
	je	L(eq)
#endif
	movzbl	2(%eax), %ecx
#if defined USE_AS_STRCASECMP_L || defined USE_AS_STRNCASECMP_L
	movzbl	2(%edx), %edi
# ifdef PIC
	movl	_nl_C_LC_CTYPE_tolower@GOTOFF+128*4(%ebx,%ecx,4), %ecx
	movl	_nl_C_LC_CTYPE_tolower@GOTOFF+128*4(%ebx,%edi,4), %edi
# else
	movl	_nl_C_LC_CTYPE_tolower+128*4(,%ecx,4), %ecx
	movl	_nl_C_LC_CTYPE_tolower+128*4(,%edi,4), %edi
# endif
	cmpl	%ecx, %edi
#else
	cmpb	%cl, 2(%edx)
#endif
	jne	L(neq)
	cmpl	$0, %ecx
	je	L(eq)

#if defined USE_AS_STRNCMP || defined USE_AS_STRNCASECMP_L
	cmp	$3, REM
	je	L(eq)
#endif
	movzbl	3(%eax), %ecx
#if defined USE_AS_STRCASECMP_L || defined USE_AS_STRNCASECMP_L
	movzbl	3(%edx), %edi
# ifdef PIC
	movl	_nl_C_LC_CTYPE_tolower@GOTOFF+128*4(%ebx,%ecx,4), %ecx
	movl	_nl_C_LC_CTYPE_tolower@GOTOFF+128*4(%ebx,%edi,4), %edi
# else
	movl	_nl_C_LC_CTYPE_tolower+128*4(,%ecx,4), %ecx
	movl	_nl_C_LC_CTYPE_tolower+128*4(,%edi,4), %edi
# endif
	cmpl	%ecx, %edi
#else
	cmpb	%cl, 3(%edx)
#endif
	jne	L(neq)
	cmpl	$0, %ecx
	je	L(eq)

#if defined USE_AS_STRNCMP || defined USE_AS_STRNCASECMP_L
	cmp	$4, REM
	je	L(eq)
#endif
	movzbl	4(%eax), %ecx
#if defined USE_AS_STRCASECMP_L || defined USE_AS_STRNCASECMP_L
	movzbl	4(%edx), %edi
# ifdef PIC
	movl	_nl_C_LC_CTYPE_tolower@GOTOFF+128*4(%ebx,%ecx,4), %ecx
	movl	_nl_C_LC_CTYPE_tolower@GOTOFF+128*4(%ebx,%edi,4), %edi
# else
	movl	_nl_C_LC_CTYPE_tolower+128*4(,%ecx,4), %ecx
	movl	_nl_C_LC_CTYPE_tolower+128*4(,%edi,4), %edi
# endif
	cmpl	%ecx, %edi
#else
	cmpb	%cl, 4(%edx)
#endif
	jne	L(neq)
	cmpl	$0, %ecx
	je	L(eq)

#if defined USE_AS_STRNCMP || defined USE_AS_STRNCASECMP_L
	cmp	$5, REM
	je	L(eq)
#endif
	movzbl	5(%eax), %ecx
#if defined USE_AS_STRCASECMP_L || defined USE_AS_STRNCASECMP_L
	movzbl	5(%edx), %edi
# ifdef PIC
	movl	_nl_C_LC_CTYPE_tolower@GOTOFF+128*4(%ebx,%ecx,4), %ecx
	movl	_nl_C_LC_CTYPE_tolower@GOTOFF+128*4(%ebx,%edi,4), %edi
# else
	movl	_nl_C_LC_CTYPE_tolower+128*4(,%ecx,4), %ecx
	movl	_nl_C_LC_CTYPE_tolower+128*4(,%edi,4), %edi
# endif
	cmpl	%ecx, %edi
#else
	cmpb	%cl, 5(%edx)
#endif
	jne	L(neq)
	cmpl	$0, %ecx
	je	L(eq)

#if defined USE_AS_STRNCMP || defined USE_AS_STRNCASECMP_L
	cmp	$6, REM
	je	L(eq)
#endif
	movzbl	6(%eax), %ecx
#if defined USE_AS_STRCASECMP_L || defined USE_AS_STRNCASECMP_L
	movzbl	6(%edx), %edi
# ifdef PIC
	movl	_nl_C_LC_CTYPE_tolower@GOTOFF+128*4(%ebx,%ecx,4), %ecx
	movl	_nl_C_LC_CTYPE_tolower@GOTOFF+128*4(%ebx,%edi,4), %edi
# else
	movl	_nl_C_LC_CTYPE_tolower+128*4(,%ecx,4), %ecx
	movl	_nl_C_LC_CTYPE_tolower+128*4(,%edi,4), %edi
# endif
	cmpl	%ecx, %edi
#else
	cmpb	%cl, 6(%edx)
#endif
	jne	L(neq)
	cmpl	$0, %ecx
	je	L(eq)

#if defined USE_AS_STRNCMP || defined USE_AS_STRNCASECMP_L
	cmp	$7, REM
	je	L(eq)
#endif
	movzbl	7(%eax), %ecx
#if defined USE_AS_STRCASECMP_L || defined USE_AS_STRNCASECMP_L
	movzbl	7(%edx), %edi
# ifdef PIC
	movl	_nl_C_LC_CTYPE_tolower@GOTOFF+128*4(%ebx,%ecx,4), %ecx
	movl	_nl_C_LC_CTYPE_tolower@GOTOFF+128*4(%ebx,%edi,4), %edi
# else
	movl	_nl_C_LC_CTYPE_tolower+128*4(,%ecx,4), %ecx
	movl	_nl_C_LC_CTYPE_tolower+128*4(,%edi,4), %edi
# endif
	cmpl	%ecx, %edi
#else
	cmpb	%cl, 7(%edx)
#endif
	jne	L(neq)
	cmpl	$0, %ecx
	je	L(eq)

#if defined USE_AS_STRNCMP || defined USE_AS_STRNCASECMP_L
	sub	$8, REM
	je	L(eq)
#endif
	add	$8, %eax
	add	$8, %edx

#if !defined USE_AS_STRCASECMP_L && !defined USE_AS_STRNCASECMP_L
	PUSH	(%edi)
#endif
	PUSH	(%esi)
#if defined USE_AS_STRNCMP || defined USE_AS_STRNCASECMP_L
	cfi_remember_state
#endif
	mov	%edx, %edi
	mov	%eax, %esi
	xorl	%eax, %eax
L(check_offset):
	movl	%edi, %edx
	movl	%esi, %ecx
	andl	$0xfff, %edx
	andl	$0xfff, %ecx
	cmpl	%edx, %ecx
	cmovl	%edx, %ecx
	lea	-0xff0(%ecx), %edx
	sub	%edx, %edi
	sub	%edx, %esi
	testl	%edx, %edx
	jg	L(crosspage)
L(loop):
	movdqu	(%esi,%edx), %xmm2
	movdqu	(%edi,%edx), %xmm1
	TOLOWER (%xmm2, %xmm1)
	pcmpistri	$0x1a, %xmm2, %xmm1
	jbe	L(end)

#if defined USE_AS_STRNCMP || defined USE_AS_STRNCASECMP_L
	sub	$16, REM
	jbe	L(more16byteseq)
#endif

	add	$16, %edx
	jle	L(loop)
L(crosspage):
	movzbl	(%edi,%edx), %eax
	movzbl	(%esi,%edx), %ecx
#if defined USE_AS_STRCASECMP_L || defined USE_AS_STRNCASECMP_L
# ifdef PIC
	movl	_nl_C_LC_CTYPE_tolower@GOTOFF+128*4(%ebx,%eax,4), %eax
	movl	_nl_C_LC_CTYPE_tolower@GOTOFF+128*4(%ebx,%ecx,4), %ecx
# else
	movl	_nl_C_LC_CTYPE_tolower+128*4(,%eax,4), %eax
	movl	_nl_C_LC_CTYPE_tolower+128*4(,%ecx,4), %ecx
# endif
#endif
	subl	%ecx, %eax
	jne	L(ret)
	testl	%ecx, %ecx
	je	L(ret)
#if defined USE_AS_STRNCMP || defined USE_AS_STRNCASECMP_L
	sub	$1, REM
	jbe	L(more16byteseq)
#endif
	inc	%edx
	cmp	$15, %edx
	jle	L(crosspage)
	add	%edx, %edi
	add	%edx, %esi
	jmp	L(check_offset)

	.p2align 4
L(end):
	jnc	L(ret)
#if defined USE_AS_STRNCMP || defined USE_AS_STRNCASECMP_L
	sub	%ecx, REM
	jbe	L(more16byteseq)
#endif
	lea	(%ecx,%edx), %ecx
	movzbl	(%edi,%ecx), %eax
	movzbl	(%esi,%ecx), %ecx
#if defined USE_AS_STRCASECMP_L || defined USE_AS_STRNCASECMP_L
# ifdef PIC
	movl	_nl_C_LC_CTYPE_tolower@GOTOFF+128*4(%ebx,%eax,4), %eax
	movl	_nl_C_LC_CTYPE_tolower@GOTOFF+128*4(%ebx,%ecx,4), %ecx
# else
	movl	_nl_C_LC_CTYPE_tolower+128*4(,%eax,4), %eax
	movl	_nl_C_LC_CTYPE_tolower+128*4(,%ecx,4), %ecx
# endif
#endif
	subl	%ecx, %eax
L(ret):
	POP	(%esi)
	POP	(%edi)
#if defined USE_AS_STRNCMP || defined USE_AS_STRNCASECMP_L
	POP	(REM)
#endif
#if defined USE_AS_STRCASECMP_L || defined USE_AS_STRNCASECMP_L
# ifdef PIC
	POP	(%ebx)
# endif
#endif
	ret

	.p2align 4
#if defined USE_AS_STRNCMP || defined USE_AS_STRNCASECMP_L
	cfi_restore_state
L(more16byteseq):
	POP	(%esi)
# ifdef USE_AS_STRNCMP
	POP	(%edi)
# endif
#endif
L(eq):
	xorl	%eax, %eax
	RETURN

L(neq):
	mov	$1, %eax
	ja	L(neq_bigger)
	neg	%eax
L(neq_bigger):
	RETURN

L(less16bytes):
	add	$0xfefefeff, %ecx
	jnc	L(less4bytes)
#if defined USE_AS_STRCASECMP_L || defined USE_AS_STRNCASECMP_L
	movd	%xmm3, %edi
	xor	%edi, %ecx
#else
	xor	(%edx), %ecx
#endif
	or	$0xfefefeff, %ecx
	add	$1, %ecx
	jnz	L(less4bytes)

#if defined USE_AS_STRNCMP || defined USE_AS_STRNCASECMP_L
	cmp	$4, REM
	jbe	L(eq)
#endif
#if defined USE_AS_STRCASECMP_L || defined USE_AS_STRNCASECMP_L
	psrldq	$4, %xmm3
	psrldq	$4, %xmm4
	movd	%xmm3, %ecx
	movd	%xmm4, %edi
	cmp	%edi, %ecx
	mov	%ecx, %edi
#else
	mov	4(%edx), %ecx
	cmp	4(%eax), %ecx
#endif
	jne	L(more4bytes)
	add	$0xfefefeff, %ecx
	jnc	L(more4bytes)
#if defined USE_AS_STRCASECMP_L || defined USE_AS_STRNCASECMP_L
	xor	%edi, %ecx
#else
	xor	4(%edx), %ecx
#endif
	or	$0xfefefeff, %ecx
	add	$1, %ecx
	jnz	L(more4bytes)

#if defined USE_AS_STRNCMP || defined USE_AS_STRNCASECMP_L
	sub	$8, REM
	jbe	L(eq)
#endif

	add	$8, %edx
	add	$8, %eax
L(less4bytes):

	movzbl	(%eax), %ecx
#if defined USE_AS_STRCASECMP_L || defined USE_AS_STRNCASECMP_L
	movzbl	(%edx), %edi
# ifdef PIC
	movl	_nl_C_LC_CTYPE_tolower@GOTOFF+128*4(%ebx,%ecx,4), %ecx
	movl	_nl_C_LC_CTYPE_tolower@GOTOFF+128*4(%ebx,%edi,4), %edi
# else
	movl	_nl_C_LC_CTYPE_tolower+128*4(,%ecx,4), %ecx
	movl	_nl_C_LC_CTYPE_tolower+128*4(,%edi,4), %edi
# endif
	cmpl	%ecx, %edi
#else
	cmpb	%cl, (%edx)
#endif
	jne	L(neq)
	cmpl	$0, %ecx
	je	L(eq)

#if defined USE_AS_STRNCMP || defined USE_AS_STRNCASECMP_L
	cmp	$1, REM
	je	L(eq)
#endif
	movzbl	1(%eax), %ecx
#if defined USE_AS_STRCASECMP_L || defined USE_AS_STRNCASECMP_L
	movzbl	1(%edx), %edi
# ifdef PIC
	movl	_nl_C_LC_CTYPE_tolower@GOTOFF+128*4(%ebx,%ecx,4), %ecx
	movl	_nl_C_LC_CTYPE_tolower@GOTOFF+128*4(%ebx,%edi,4), %edi
# else
	movl	_nl_C_LC_CTYPE_tolower+128*4(,%ecx,4), %ecx
	movl	_nl_C_LC_CTYPE_tolower+128*4(,%edi,4), %edi
# endif
	cmpl	%ecx, %edi
#else
	cmpb	%cl, 1(%edx)
#endif
	jne	L(neq)
	cmpl	$0, %ecx
	je	L(eq)

#if defined USE_AS_STRNCMP || defined USE_AS_STRNCASECMP_L
	cmp	$2, REM
	je	L(eq)
#endif

	movzbl	2(%eax), %ecx
#if defined USE_AS_STRCASECMP_L || defined USE_AS_STRNCASECMP_L
	movzbl	2(%edx), %edi
# ifdef PIC
	movl	_nl_C_LC_CTYPE_tolower@GOTOFF+128*4(%ebx,%ecx,4), %ecx
	movl	_nl_C_LC_CTYPE_tolower@GOTOFF+128*4(%ebx,%edi,4), %edi
# else
	movl	_nl_C_LC_CTYPE_tolower+128*4(,%ecx,4), %ecx
	movl	_nl_C_LC_CTYPE_tolower+128*4(,%edi,4), %edi
# endif
	cmpl	%ecx, %edi
#else
	cmpb	%cl, 2(%edx)
#endif
	jne	L(neq)
	cmpl	$0, %ecx
	je	L(eq)

#if defined USE_AS_STRNCMP || defined USE_AS_STRNCASECMP_L
	cmp	$3, REM
	je	L(eq)
#endif
	movzbl	3(%eax), %ecx
#if defined USE_AS_STRCASECMP_L || defined USE_AS_STRNCASECMP_L
	movzbl	3(%edx), %edi
# ifdef PIC
	movl	_nl_C_LC_CTYPE_tolower@GOTOFF+128*4(%ebx,%ecx,4), %ecx
	movl	_nl_C_LC_CTYPE_tolower@GOTOFF+128*4(%ebx,%edi,4), %edi
# else
	movl	_nl_C_LC_CTYPE_tolower+128*4(,%ecx,4), %ecx
	movl	_nl_C_LC_CTYPE_tolower+128*4(,%edi,4), %edi
# endif
	cmpl	%ecx, %edi
#else
	cmpb	%cl, 3(%edx)
#endif
	jne	L(neq)
	cmpl	$0, %ecx
	je	L(eq)

L(more4bytes):
#if defined USE_AS_STRNCMP || defined USE_AS_STRNCASECMP_L
	cmp	$4, REM
	je	L(eq)
#endif
	movzbl	4(%eax), %ecx
#if defined USE_AS_STRCASECMP_L || defined USE_AS_STRNCASECMP_L
	movzbl	4(%edx), %edi
# ifdef PIC
	movl	_nl_C_LC_CTYPE_tolower@GOTOFF+128*4(%ebx,%ecx,4), %ecx
	movl	_nl_C_LC_CTYPE_tolower@GOTOFF+128*4(%ebx,%edi,4), %edi
# else
	movl	_nl_C_LC_CTYPE_tolower+128*4(,%ecx,4), %ecx
	movl	_nl_C_LC_CTYPE_tolower+128*4(,%edi,4), %edi
# endif
	cmpl	%ecx, %edi
#else
	cmpb	%cl, 4(%edx)
#endif
	jne	L(neq)
	cmpl	$0, %ecx
	je	L(eq)


#if defined USE_AS_STRNCMP || defined USE_AS_STRNCASECMP_L
	cmp	$5, REM
	je	L(eq)
#endif
	movzbl	5(%eax), %ecx
#if defined USE_AS_STRCASECMP_L || defined USE_AS_STRNCASECMP_L
	movzbl	5(%edx), %edi
# ifdef PIC
	movl	_nl_C_LC_CTYPE_tolower@GOTOFF+128*4(%ebx,%ecx,4), %ecx
	movl	_nl_C_LC_CTYPE_tolower@GOTOFF+128*4(%ebx,%edi,4), %edi
# else
	movl	_nl_C_LC_CTYPE_tolower+128*4(,%ecx,4), %ecx
	movl	_nl_C_LC_CTYPE_tolower+128*4(,%edi,4), %edi
# endif
	cmpl	%ecx, %edi
#else
	cmpb	%cl, 5(%edx)
#endif
	jne	L(neq)
	cmpl	$0, %ecx
	je	L(eq)

#if defined USE_AS_STRNCMP || defined USE_AS_STRNCASECMP_L
	cmp	$6, REM
	je	L(eq)
#endif
	movzbl	6(%eax), %ecx
#if defined USE_AS_STRCASECMP_L || defined USE_AS_STRNCASECMP_L
	movzbl	6(%edx), %edi
# ifdef PIC
	movl	_nl_C_LC_CTYPE_tolower@GOTOFF+128*4(%ebx,%ecx,4), %ecx
	movl	_nl_C_LC_CTYPE_tolower@GOTOFF+128*4(%ebx,%edi,4), %edi
# else
	movl	_nl_C_LC_CTYPE_tolower+128*4(,%ecx,4), %ecx
	movl	_nl_C_LC_CTYPE_tolower+128*4(,%edi,4), %edi
# endif
	cmpl	%ecx, %edi
#else
	cmpb	%cl, 6(%edx)
#endif
	jne	L(neq)
	cmpl	$0, %ecx
	je	L(eq)

#if defined USE_AS_STRNCMP || defined USE_AS_STRNCASECMP_L
	cmp	$7, REM
	je	L(eq)
#endif
	movzbl	7(%eax), %ecx
#if defined USE_AS_STRCASECMP_L || defined USE_AS_STRNCASECMP_L
	movzbl	7(%edx), %edi
# ifdef PIC
	movl	_nl_C_LC_CTYPE_tolower@GOTOFF+128*4(%ebx,%ecx,4), %ecx
	movl	_nl_C_LC_CTYPE_tolower@GOTOFF+128*4(%ebx,%edi,4), %edi
# else
	movl	_nl_C_LC_CTYPE_tolower+128*4(,%ecx,4), %ecx
	movl	_nl_C_LC_CTYPE_tolower+128*4(,%edi,4), %edi
# endif
	cmpl	%ecx, %edi
#else
	cmpb	%cl, 7(%edx)
#endif
	jne	L(neq)
	jmp	L(eq)

END (STRCMP)

#endif
