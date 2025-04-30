/* memcmp with SSE4.1, wmemcmp with SSE4.1
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

# include <sysdep.h>

# ifndef MEMCMP
#  define MEMCMP	__memcmp_sse4_1
# endif

# define JMPTBL(I, B)	(I - B)

# define BRANCH_TO_JMPTBL_ENTRY(TABLE, INDEX, SCALE)		\
  lea		TABLE(%rip), %r11;				\
  movslq	(%r11, INDEX, SCALE), %rcx;			\
  add		%r11, %rcx;					\
  _CET_NOTRACK jmp *%rcx;					\
  ud2

/* Warning!
           wmemcmp has to use SIGNED comparison for elements.
           memcmp has to use UNSIGNED comparison for elemnts.
*/

	.section .text.sse4.1,"ax",@progbits
ENTRY (MEMCMP)
# ifdef USE_AS_WMEMCMP
	shl	$2, %RDX_LP
# elif defined __ILP32__
	/* Clear the upper 32 bits.  */
	mov	%edx, %edx
# endif
	pxor	%xmm0, %xmm0
	cmp	$79, %RDX_LP
	ja	L(79bytesormore)
# ifndef USE_AS_WMEMCMP
	cmp	$1, %RDX_LP
	je	L(firstbyte)
# endif
	add	%rdx, %rsi
	add	%rdx, %rdi
	BRANCH_TO_JMPTBL_ENTRY(L(table_64bytes), %rdx, 4)

# ifndef USE_AS_WMEMCMP
	.p2align 4
L(firstbyte):
	movzbl	(%rdi), %eax
	movzbl	(%rsi), %ecx
	sub	%ecx, %eax
	ret
# endif

	.p2align 4
L(79bytesormore):
	movdqu	(%rsi), %xmm1
	movdqu	(%rdi), %xmm2
	pxor	%xmm1, %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(16bytesin256)
	mov	%rsi, %rcx
	and	$-16, %rsi
	add	$16, %rsi
	sub	%rsi, %rcx

	sub	%rcx, %rdi
	add	%rcx, %rdx
	test	$0xf, %rdi
	jz	L(2aligned)

	cmp	$128, %rdx
	ja	L(128bytesormore)
L(less128bytes):
	sub	$64, %rdx

	movdqu	(%rdi), %xmm2
	pxor	(%rsi), %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(16bytesin256)

	movdqu	16(%rdi), %xmm2
	pxor	16(%rsi), %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(32bytesin256)

	movdqu	32(%rdi), %xmm2
	pxor	32(%rsi), %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(48bytesin256)

	movdqu	48(%rdi), %xmm2
	pxor	48(%rsi), %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(64bytesin256)
	cmp	$32, %rdx
	jb	L(less32bytesin64)

	movdqu	64(%rdi), %xmm2
	pxor	64(%rsi), %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(80bytesin256)

	movdqu	80(%rdi), %xmm2
	pxor	80(%rsi), %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(96bytesin256)
	sub	$32, %rdx
	add	$32, %rdi
	add	$32, %rsi
L(less32bytesin64):
	add	$64, %rdi
	add	$64, %rsi
	add	%rdx, %rsi
	add	%rdx, %rdi
	BRANCH_TO_JMPTBL_ENTRY(L(table_64bytes), %rdx, 4)

L(128bytesormore):
	cmp	$512, %rdx
	ja	L(512bytesormore)
	cmp	$256, %rdx
	ja	L(less512bytes)
L(less256bytes):
	sub	$128, %rdx

	movdqu	(%rdi), %xmm2
	pxor	(%rsi), %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(16bytesin256)

	movdqu	16(%rdi), %xmm2
	pxor	16(%rsi), %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(32bytesin256)

	movdqu	32(%rdi), %xmm2
	pxor	32(%rsi), %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(48bytesin256)

	movdqu	48(%rdi), %xmm2
	pxor	48(%rsi), %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(64bytesin256)

	movdqu	64(%rdi), %xmm2
	pxor	64(%rsi), %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(80bytesin256)

	movdqu	80(%rdi), %xmm2
	pxor	80(%rsi), %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(96bytesin256)

	movdqu	96(%rdi), %xmm2
	pxor	96(%rsi), %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(112bytesin256)

	movdqu	112(%rdi), %xmm2
	pxor	112(%rsi), %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(128bytesin256)

	add	$128, %rsi
	add	$128, %rdi

	cmp	$64, %rdx
	jae	L(less128bytes)

	cmp	$32, %rdx
	jb	L(less32bytesin128)

	movdqu	(%rdi), %xmm2
	pxor	(%rsi), %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(16bytesin256)

	movdqu	16(%rdi), %xmm2
	pxor	16(%rsi), %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(32bytesin256)
	sub	$32, %rdx
	add	$32, %rdi
	add	$32, %rsi
L(less32bytesin128):
	add	%rdx, %rsi
	add	%rdx, %rdi
	BRANCH_TO_JMPTBL_ENTRY(L(table_64bytes), %rdx, 4)

L(less512bytes):
	sub	$256, %rdx
	movdqu	(%rdi), %xmm2
	pxor	(%rsi), %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(16bytesin256)

	movdqu	16(%rdi), %xmm2
	pxor	16(%rsi), %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(32bytesin256)

	movdqu	32(%rdi), %xmm2
	pxor	32(%rsi), %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(48bytesin256)

	movdqu	48(%rdi), %xmm2
	pxor	48(%rsi), %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(64bytesin256)

	movdqu	64(%rdi), %xmm2
	pxor	64(%rsi), %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(80bytesin256)

	movdqu	80(%rdi), %xmm2
	pxor	80(%rsi), %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(96bytesin256)

	movdqu	96(%rdi), %xmm2
	pxor	96(%rsi), %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(112bytesin256)

	movdqu	112(%rdi), %xmm2
	pxor	112(%rsi), %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(128bytesin256)

	movdqu	128(%rdi), %xmm2
	pxor	128(%rsi), %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(144bytesin256)

	movdqu	144(%rdi), %xmm2
	pxor	144(%rsi), %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(160bytesin256)

	movdqu	160(%rdi), %xmm2
	pxor	160(%rsi), %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(176bytesin256)

	movdqu	176(%rdi), %xmm2
	pxor	176(%rsi), %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(192bytesin256)

	movdqu	192(%rdi), %xmm2
	pxor	192(%rsi), %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(208bytesin256)

	movdqu	208(%rdi), %xmm2
	pxor	208(%rsi), %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(224bytesin256)

	movdqu	224(%rdi), %xmm2
	pxor	224(%rsi), %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(240bytesin256)

	movdqu	240(%rdi), %xmm2
	pxor	240(%rsi), %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(256bytesin256)

	add	$256, %rsi
	add	$256, %rdi

	cmp	$128, %rdx
	jae	L(less256bytes)

	cmp	$64, %rdx
	jae	L(less128bytes)

	cmp	$32, %rdx
	jb	L(less32bytesin256)

	movdqu	(%rdi), %xmm2
	pxor	(%rsi), %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(16bytesin256)

	movdqu	16(%rdi), %xmm2
	pxor	16(%rsi), %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(32bytesin256)
	sub	$32, %rdx
	add	$32, %rdi
	add	$32, %rsi
L(less32bytesin256):
	add	%rdx, %rsi
	add	%rdx, %rdi
	BRANCH_TO_JMPTBL_ENTRY(L(table_64bytes), %rdx, 4)

	.p2align 4
L(512bytesormore):
# ifdef DATA_CACHE_SIZE_HALF
	mov	$DATA_CACHE_SIZE_HALF, %R8_LP
# else
	mov	__x86_data_cache_size_half(%rip), %R8_LP
# endif
	mov	%r8, %r9
	shr	$1, %r8
	add	%r9, %r8
	cmp	%r8, %rdx
	ja	L(L2_L3_cache_unaglined)
	sub	$64, %rdx
	.p2align 4
L(64bytesormore_loop):
	movdqu	(%rdi), %xmm2
	pxor	(%rsi), %xmm2
	movdqa	%xmm2, %xmm1

	movdqu	16(%rdi), %xmm3
	pxor	16(%rsi), %xmm3
	por	%xmm3, %xmm1

	movdqu	32(%rdi), %xmm4
	pxor	32(%rsi), %xmm4
	por	%xmm4, %xmm1

	movdqu	48(%rdi), %xmm5
	pxor	48(%rsi), %xmm5
	por	%xmm5, %xmm1

	ptest	%xmm1, %xmm0
	jnc	L(64bytesormore_loop_end)
	add	$64, %rsi
	add	$64, %rdi
	sub	$64, %rdx
	jae	L(64bytesormore_loop)

	add	$64, %rdx
	add	%rdx, %rsi
	add	%rdx, %rdi
	BRANCH_TO_JMPTBL_ENTRY(L(table_64bytes), %rdx, 4)

L(L2_L3_cache_unaglined):
	sub	$64, %rdx
	.p2align 4
L(L2_L3_unaligned_128bytes_loop):
	prefetchnta 0x1c0(%rdi)
	prefetchnta 0x1c0(%rsi)
	movdqu	(%rdi), %xmm2
	pxor	(%rsi), %xmm2
	movdqa	%xmm2, %xmm1

	movdqu	16(%rdi), %xmm3
	pxor	16(%rsi), %xmm3
	por	%xmm3, %xmm1

	movdqu	32(%rdi), %xmm4
	pxor	32(%rsi), %xmm4
	por	%xmm4, %xmm1

	movdqu	48(%rdi), %xmm5
	pxor	48(%rsi), %xmm5
	por	%xmm5, %xmm1

	ptest	%xmm1, %xmm0
	jnc	L(64bytesormore_loop_end)
	add	$64, %rsi
	add	$64, %rdi
	sub	$64, %rdx
	jae	L(L2_L3_unaligned_128bytes_loop)

	add	$64, %rdx
	add	%rdx, %rsi
	add	%rdx, %rdi
	BRANCH_TO_JMPTBL_ENTRY(L(table_64bytes), %rdx, 4)

/*
 * This case is for machines which are sensitive for unaligned instructions.
 */
	.p2align 4
L(2aligned):
	cmp	$128, %rdx
	ja	L(128bytesormorein2aligned)
L(less128bytesin2aligned):
	sub	$64, %rdx

	movdqa	(%rdi), %xmm2
	pxor	(%rsi), %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(16bytesin256)

	movdqa	16(%rdi), %xmm2
	pxor	16(%rsi), %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(32bytesin256)

	movdqa	32(%rdi), %xmm2
	pxor	32(%rsi), %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(48bytesin256)

	movdqa	48(%rdi), %xmm2
	pxor	48(%rsi), %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(64bytesin256)
	cmp	$32, %rdx
	jb	L(less32bytesin64in2alinged)

	movdqa	64(%rdi), %xmm2
	pxor	64(%rsi), %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(80bytesin256)

	movdqa	80(%rdi), %xmm2
	pxor	80(%rsi), %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(96bytesin256)
	sub	$32, %rdx
	add	$32, %rdi
	add	$32, %rsi
L(less32bytesin64in2alinged):
	add	$64, %rdi
	add	$64, %rsi
	add	%rdx, %rsi
	add	%rdx, %rdi
	BRANCH_TO_JMPTBL_ENTRY(L(table_64bytes), %rdx, 4)

	.p2align 4
L(128bytesormorein2aligned):
	cmp	$512, %rdx
	ja	L(512bytesormorein2aligned)
	cmp	$256, %rdx
	ja	L(256bytesormorein2aligned)
L(less256bytesin2alinged):
	sub	$128, %rdx

	movdqa	(%rdi), %xmm2
	pxor	(%rsi), %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(16bytesin256)

	movdqa	16(%rdi), %xmm2
	pxor	16(%rsi), %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(32bytesin256)

	movdqa	32(%rdi), %xmm2
	pxor	32(%rsi), %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(48bytesin256)

	movdqa	48(%rdi), %xmm2
	pxor	48(%rsi), %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(64bytesin256)

	movdqa	64(%rdi), %xmm2
	pxor	64(%rsi), %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(80bytesin256)

	movdqa	80(%rdi), %xmm2
	pxor	80(%rsi), %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(96bytesin256)

	movdqa	96(%rdi), %xmm2
	pxor	96(%rsi), %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(112bytesin256)

	movdqa	112(%rdi), %xmm2
	pxor	112(%rsi), %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(128bytesin256)

	add	$128, %rsi
	add	$128, %rdi

	cmp	$64, %rdx
	jae	L(less128bytesin2aligned)

	cmp	$32, %rdx
	jb	L(less32bytesin128in2aligned)

	movdqu	(%rdi), %xmm2
	pxor	(%rsi), %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(16bytesin256)

	movdqu	16(%rdi), %xmm2
	pxor	16(%rsi), %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(32bytesin256)
	sub	$32, %rdx
	add	$32, %rdi
	add	$32, %rsi
L(less32bytesin128in2aligned):
	add	%rdx, %rsi
	add	%rdx, %rdi
	BRANCH_TO_JMPTBL_ENTRY(L(table_64bytes), %rdx, 4)

	.p2align 4
L(256bytesormorein2aligned):

	sub	$256, %rdx
	movdqa	(%rdi), %xmm2
	pxor	(%rsi), %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(16bytesin256)

	movdqa	16(%rdi), %xmm2
	pxor	16(%rsi), %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(32bytesin256)

	movdqa	32(%rdi), %xmm2
	pxor	32(%rsi), %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(48bytesin256)

	movdqa	48(%rdi), %xmm2
	pxor	48(%rsi), %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(64bytesin256)

	movdqa	64(%rdi), %xmm2
	pxor	64(%rsi), %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(80bytesin256)

	movdqa	80(%rdi), %xmm2
	pxor	80(%rsi), %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(96bytesin256)

	movdqa	96(%rdi), %xmm2
	pxor	96(%rsi), %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(112bytesin256)

	movdqa	112(%rdi), %xmm2
	pxor	112(%rsi), %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(128bytesin256)

	movdqa	128(%rdi), %xmm2
	pxor	128(%rsi), %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(144bytesin256)

	movdqa	144(%rdi), %xmm2
	pxor	144(%rsi), %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(160bytesin256)

	movdqa	160(%rdi), %xmm2
	pxor	160(%rsi), %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(176bytesin256)

	movdqa	176(%rdi), %xmm2
	pxor	176(%rsi), %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(192bytesin256)

	movdqa	192(%rdi), %xmm2
	pxor	192(%rsi), %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(208bytesin256)

	movdqa	208(%rdi), %xmm2
	pxor	208(%rsi), %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(224bytesin256)

	movdqa	224(%rdi), %xmm2
	pxor	224(%rsi), %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(240bytesin256)

	movdqa	240(%rdi), %xmm2
	pxor	240(%rsi), %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(256bytesin256)

	add	$256, %rsi
	add	$256, %rdi

	cmp	$128, %rdx
	jae	L(less256bytesin2alinged)

	cmp	$64, %rdx
	jae	L(less128bytesin2aligned)

	cmp	$32, %rdx
	jb	L(less32bytesin256in2alinged)

	movdqa	(%rdi), %xmm2
	pxor	(%rsi), %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(16bytesin256)

	movdqa	16(%rdi), %xmm2
	pxor	16(%rsi), %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(32bytesin256)
	sub	$32, %rdx
	add	$32, %rdi
	add	$32, %rsi
L(less32bytesin256in2alinged):
	add	%rdx, %rsi
	add	%rdx, %rdi
	BRANCH_TO_JMPTBL_ENTRY(L(table_64bytes), %rdx, 4)

	.p2align 4
L(512bytesormorein2aligned):
# ifdef DATA_CACHE_SIZE_HALF
	mov	$DATA_CACHE_SIZE_HALF, %R8_LP
# else
	mov	__x86_data_cache_size_half(%rip), %R8_LP
# endif
	mov	%r8, %r9
	shr	$1, %r8
	add	%r9, %r8
	cmp	%r8, %rdx
	ja	L(L2_L3_cache_aglined)

	sub	$64, %rdx
	.p2align 4
L(64bytesormore_loopin2aligned):
	movdqa	(%rdi), %xmm2
	pxor	(%rsi), %xmm2
	movdqa	%xmm2, %xmm1

	movdqa	16(%rdi), %xmm3
	pxor	16(%rsi), %xmm3
	por	%xmm3, %xmm1

	movdqa	32(%rdi), %xmm4
	pxor	32(%rsi), %xmm4
	por	%xmm4, %xmm1

	movdqa	48(%rdi), %xmm5
	pxor	48(%rsi), %xmm5
	por	%xmm5, %xmm1

	ptest	%xmm1, %xmm0
	jnc	L(64bytesormore_loop_end)
	add	$64, %rsi
	add	$64, %rdi
	sub	$64, %rdx
	jae	L(64bytesormore_loopin2aligned)

	add	$64, %rdx
	add	%rdx, %rsi
	add	%rdx, %rdi
	BRANCH_TO_JMPTBL_ENTRY(L(table_64bytes), %rdx, 4)
L(L2_L3_cache_aglined):
	sub	$64, %rdx

	.p2align 4
L(L2_L3_aligned_128bytes_loop):
	prefetchnta 0x1c0(%rdi)
	prefetchnta 0x1c0(%rsi)
	movdqa	(%rdi), %xmm2
	pxor	(%rsi), %xmm2
	movdqa	%xmm2, %xmm1

	movdqa	16(%rdi), %xmm3
	pxor	16(%rsi), %xmm3
	por	%xmm3, %xmm1

	movdqa	32(%rdi), %xmm4
	pxor	32(%rsi), %xmm4
	por	%xmm4, %xmm1

	movdqa	48(%rdi), %xmm5
	pxor	48(%rsi), %xmm5
	por	%xmm5, %xmm1

	ptest	%xmm1, %xmm0
	jnc	L(64bytesormore_loop_end)
	add	$64, %rsi
	add	$64, %rdi
	sub	$64, %rdx
	jae	L(L2_L3_aligned_128bytes_loop)

	add	$64, %rdx
	add	%rdx, %rsi
	add	%rdx, %rdi
	BRANCH_TO_JMPTBL_ENTRY(L(table_64bytes), %rdx, 4)


	.p2align 4
L(64bytesormore_loop_end):
	add	$16, %rdi
	add	$16, %rsi
	ptest	%xmm2, %xmm0
	jnc	L(16bytes)

	add	$16, %rdi
	add	$16, %rsi
	ptest	%xmm3, %xmm0
	jnc	L(16bytes)

	add	$16, %rdi
	add	$16, %rsi
	ptest	%xmm4, %xmm0
	jnc	L(16bytes)

	add	$16, %rdi
	add	$16, %rsi
	jmp	L(16bytes)

L(256bytesin256):
	add	$256, %rdi
	add	$256, %rsi
	jmp	L(16bytes)
L(240bytesin256):
	add	$240, %rdi
	add	$240, %rsi
	jmp	L(16bytes)
L(224bytesin256):
	add	$224, %rdi
	add	$224, %rsi
	jmp	L(16bytes)
L(208bytesin256):
	add	$208, %rdi
	add	$208, %rsi
	jmp	L(16bytes)
L(192bytesin256):
	add	$192, %rdi
	add	$192, %rsi
	jmp	L(16bytes)
L(176bytesin256):
	add	$176, %rdi
	add	$176, %rsi
	jmp	L(16bytes)
L(160bytesin256):
	add	$160, %rdi
	add	$160, %rsi
	jmp	L(16bytes)
L(144bytesin256):
	add	$144, %rdi
	add	$144, %rsi
	jmp	L(16bytes)
L(128bytesin256):
	add	$128, %rdi
	add	$128, %rsi
	jmp	L(16bytes)
L(112bytesin256):
	add	$112, %rdi
	add	$112, %rsi
	jmp	L(16bytes)
L(96bytesin256):
	add	$96, %rdi
	add	$96, %rsi
	jmp	L(16bytes)
L(80bytesin256):
	add	$80, %rdi
	add	$80, %rsi
	jmp	L(16bytes)
L(64bytesin256):
	add	$64, %rdi
	add	$64, %rsi
	jmp	L(16bytes)
L(48bytesin256):
	add	$16, %rdi
	add	$16, %rsi
L(32bytesin256):
	add	$16, %rdi
	add	$16, %rsi
L(16bytesin256):
	add	$16, %rdi
	add	$16, %rsi
L(16bytes):
	mov	-16(%rdi), %rax
	mov	-16(%rsi), %rcx
	cmp	%rax, %rcx
	jne	L(diffin8bytes)
L(8bytes):
	mov	-8(%rdi), %rax
	mov	-8(%rsi), %rcx
	cmp	%rax, %rcx
	jne	L(diffin8bytes)
	xor	%eax, %eax
	ret

	.p2align 4
L(12bytes):
	mov	-12(%rdi), %rax
	mov	-12(%rsi), %rcx
	cmp	%rax, %rcx
	jne	L(diffin8bytes)
L(4bytes):
	mov	-4(%rsi), %ecx
# ifndef USE_AS_WMEMCMP
	mov	-4(%rdi), %eax
	cmp	%eax, %ecx
# else
	cmp	-4(%rdi), %ecx
# endif
	jne	L(diffin4bytes)
L(0bytes):
	xor	%eax, %eax
	ret

# ifndef USE_AS_WMEMCMP
/* unreal case for wmemcmp */
	.p2align 4
L(65bytes):
	movdqu	-65(%rdi), %xmm1
	movdqu	-65(%rsi), %xmm2
	mov	$-65, %dl
	pxor	%xmm1, %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(less16bytes)
L(49bytes):
	movdqu	-49(%rdi), %xmm1
	movdqu	-49(%rsi), %xmm2
	mov	$-49, %dl
	pxor	%xmm1, %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(less16bytes)
L(33bytes):
	movdqu	-33(%rdi), %xmm1
	movdqu	-33(%rsi), %xmm2
	mov	$-33, %dl
	pxor	%xmm1, %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(less16bytes)
L(17bytes):
	mov	-17(%rdi), %rax
	mov	-17(%rsi), %rcx
	cmp	%rax, %rcx
	jne	L(diffin8bytes)
L(9bytes):
	mov	-9(%rdi), %rax
	mov	-9(%rsi), %rcx
	cmp	%rax, %rcx
	jne	L(diffin8bytes)
	movzbl	-1(%rdi), %eax
	movzbl	-1(%rsi), %edx
	sub	%edx, %eax
	ret

	.p2align 4
L(13bytes):
	mov	-13(%rdi), %rax
	mov	-13(%rsi), %rcx
	cmp	%rax, %rcx
	jne	L(diffin8bytes)
	mov	-8(%rdi), %rax
	mov	-8(%rsi), %rcx
	cmp	%rax, %rcx
	jne	L(diffin8bytes)
	xor	%eax, %eax
	ret

	.p2align 4
L(5bytes):
	mov	-5(%rdi), %eax
	mov	-5(%rsi), %ecx
	cmp	%eax, %ecx
	jne	L(diffin4bytes)
	movzbl	-1(%rdi), %eax
	movzbl	-1(%rsi), %edx
	sub	%edx, %eax
	ret

	.p2align 4
L(66bytes):
	movdqu	-66(%rdi), %xmm1
	movdqu	-66(%rsi), %xmm2
	mov	$-66, %dl
	pxor	%xmm1, %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(less16bytes)
L(50bytes):
	movdqu	-50(%rdi), %xmm1
	movdqu	-50(%rsi), %xmm2
	mov	$-50, %dl
	pxor	%xmm1, %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(less16bytes)
L(34bytes):
	movdqu	-34(%rdi), %xmm1
	movdqu	-34(%rsi), %xmm2
	mov	$-34, %dl
	pxor	%xmm1, %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(less16bytes)
L(18bytes):
	mov	-18(%rdi), %rax
	mov	-18(%rsi), %rcx
	cmp	%rax, %rcx
	jne	L(diffin8bytes)
L(10bytes):
	mov	-10(%rdi), %rax
	mov	-10(%rsi), %rcx
	cmp	%rax, %rcx
	jne	L(diffin8bytes)
	movzwl	-2(%rdi), %eax
	movzwl	-2(%rsi), %ecx
	cmp	%cl, %al
	jne	L(end)
	and	$0xffff, %eax
	and	$0xffff, %ecx
	sub	%ecx, %eax
	ret

	.p2align 4
L(14bytes):
	mov	-14(%rdi), %rax
	mov	-14(%rsi), %rcx
	cmp	%rax, %rcx
	jne	L(diffin8bytes)
	mov	-8(%rdi), %rax
	mov	-8(%rsi), %rcx
	cmp	%rax, %rcx
	jne	L(diffin8bytes)
	xor	%eax, %eax
	ret

	.p2align 4
L(6bytes):
	mov	-6(%rdi), %eax
	mov	-6(%rsi), %ecx
	cmp	%eax, %ecx
	jne	L(diffin4bytes)
L(2bytes):
	movzwl	-2(%rsi), %ecx
	movzwl	-2(%rdi), %eax
	cmp	%cl, %al
	jne	L(end)
	and	$0xffff, %eax
	and	$0xffff, %ecx
	sub	%ecx, %eax
	ret

	.p2align 4
L(67bytes):
	movdqu	-67(%rdi), %xmm2
	movdqu	-67(%rsi), %xmm1
	mov	$-67, %dl
	pxor	%xmm1, %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(less16bytes)
L(51bytes):
	movdqu	-51(%rdi), %xmm2
	movdqu	-51(%rsi), %xmm1
	mov	$-51, %dl
	pxor	%xmm1, %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(less16bytes)
L(35bytes):
	movdqu	-35(%rsi), %xmm1
	movdqu	-35(%rdi), %xmm2
	mov	$-35, %dl
	pxor	%xmm1, %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(less16bytes)
L(19bytes):
	mov	-19(%rdi), %rax
	mov	-19(%rsi), %rcx
	cmp	%rax, %rcx
	jne	L(diffin8bytes)
L(11bytes):
	mov	-11(%rdi), %rax
	mov	-11(%rsi), %rcx
	cmp	%rax, %rcx
	jne	L(diffin8bytes)
	mov	-4(%rdi), %eax
	mov	-4(%rsi), %ecx
	cmp	%eax, %ecx
	jne	L(diffin4bytes)
	xor	%eax, %eax
	ret

	.p2align 4
L(15bytes):
	mov	-15(%rdi), %rax
	mov	-15(%rsi), %rcx
	cmp	%rax, %rcx
	jne	L(diffin8bytes)
	mov	-8(%rdi), %rax
	mov	-8(%rsi), %rcx
	cmp	%rax, %rcx
	jne	L(diffin8bytes)
	xor	%eax, %eax
	ret

	.p2align 4
L(7bytes):
	mov	-7(%rdi), %eax
	mov	-7(%rsi), %ecx
	cmp	%eax, %ecx
	jne	L(diffin4bytes)
	mov	-4(%rdi), %eax
	mov	-4(%rsi), %ecx
	cmp	%eax, %ecx
	jne	L(diffin4bytes)
	xor	%eax, %eax
	ret

	.p2align 4
L(3bytes):
	movzwl	-3(%rdi), %eax
	movzwl	-3(%rsi), %ecx
	cmp	%eax, %ecx
	jne	L(diffin2bytes)
L(1bytes):
	movzbl	-1(%rdi), %eax
	movzbl	-1(%rsi), %ecx
	sub	%ecx, %eax
	ret
# endif

	.p2align 4
L(68bytes):
	movdqu	-68(%rdi), %xmm2
	movdqu	-68(%rsi), %xmm1
	mov	$-68, %dl
	pxor	%xmm1, %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(less16bytes)
L(52bytes):
	movdqu	-52(%rdi), %xmm2
	movdqu	-52(%rsi), %xmm1
	mov	$-52, %dl
	pxor	%xmm1, %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(less16bytes)
L(36bytes):
	movdqu	-36(%rdi), %xmm2
	movdqu	-36(%rsi), %xmm1
	mov	$-36, %dl
	pxor	%xmm1, %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(less16bytes)
L(20bytes):
	movdqu	-20(%rdi), %xmm2
	movdqu	-20(%rsi), %xmm1
	mov	$-20, %dl
	pxor	%xmm1, %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(less16bytes)
	mov	-4(%rsi), %ecx

# ifndef USE_AS_WMEMCMP
	mov	-4(%rdi), %eax
	cmp	%eax, %ecx
# else
	cmp	-4(%rdi), %ecx
# endif
	jne	L(diffin4bytes)
	xor	%eax, %eax
	ret

# ifndef USE_AS_WMEMCMP
/* unreal cases for wmemcmp */
	.p2align 4
L(69bytes):
	movdqu	-69(%rsi), %xmm1
	movdqu	-69(%rdi), %xmm2
	mov	$-69, %dl
	pxor	%xmm1, %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(less16bytes)
L(53bytes):
	movdqu	-53(%rsi), %xmm1
	movdqu	-53(%rdi), %xmm2
	mov	$-53, %dl
	pxor	%xmm1, %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(less16bytes)
L(37bytes):
	movdqu	-37(%rsi), %xmm1
	movdqu	-37(%rdi), %xmm2
	mov	$-37, %dl
	pxor	%xmm1, %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(less16bytes)
L(21bytes):
	movdqu	-21(%rsi), %xmm1
	movdqu	-21(%rdi), %xmm2
	mov	$-21, %dl
	pxor	%xmm1, %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(less16bytes)
	mov	-8(%rdi), %rax
	mov	-8(%rsi), %rcx
	cmp	%rax, %rcx
	jne	L(diffin8bytes)
	xor	%eax, %eax
	ret

	.p2align 4
L(70bytes):
	movdqu	-70(%rsi), %xmm1
	movdqu	-70(%rdi), %xmm2
	mov	$-70, %dl
	pxor	%xmm1, %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(less16bytes)
L(54bytes):
	movdqu	-54(%rsi), %xmm1
	movdqu	-54(%rdi), %xmm2
	mov	$-54, %dl
	pxor	%xmm1, %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(less16bytes)
L(38bytes):
	movdqu	-38(%rsi), %xmm1
	movdqu	-38(%rdi), %xmm2
	mov	$-38, %dl
	pxor	%xmm1, %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(less16bytes)
L(22bytes):
	movdqu	-22(%rsi), %xmm1
	movdqu	-22(%rdi), %xmm2
	mov	$-22, %dl
	pxor	%xmm1, %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(less16bytes)
	mov	-8(%rdi), %rax
	mov	-8(%rsi), %rcx
	cmp	%rax, %rcx
	jne	L(diffin8bytes)
	xor	%eax, %eax
	ret

	.p2align 4
L(71bytes):
	movdqu	-71(%rsi), %xmm1
	movdqu	-71(%rdi), %xmm2
	mov	$-71, %dl
	pxor	%xmm1, %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(less16bytes)
L(55bytes):
	movdqu	-55(%rdi), %xmm2
	movdqu	-55(%rsi), %xmm1
	mov	$-55, %dl
	pxor	%xmm1, %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(less16bytes)
L(39bytes):
	movdqu	-39(%rdi), %xmm2
	movdqu	-39(%rsi), %xmm1
	mov	$-39, %dl
	pxor	%xmm1, %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(less16bytes)
L(23bytes):
	movdqu	-23(%rdi), %xmm2
	movdqu	-23(%rsi), %xmm1
	mov	$-23, %dl
	pxor	%xmm1, %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(less16bytes)
	mov	-8(%rdi), %rax
	mov	-8(%rsi), %rcx
	cmp	%rax, %rcx
	jne	L(diffin8bytes)
	xor	%eax, %eax
	ret
# endif

	.p2align 4
L(72bytes):
	movdqu	-72(%rsi), %xmm1
	movdqu	-72(%rdi), %xmm2
	mov	$-72, %dl
	pxor	%xmm1, %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(less16bytes)
L(56bytes):
	movdqu	-56(%rdi), %xmm2
	movdqu	-56(%rsi), %xmm1
	mov	$-56, %dl
	pxor	%xmm1, %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(less16bytes)
L(40bytes):
	movdqu	-40(%rdi), %xmm2
	movdqu	-40(%rsi), %xmm1
	mov	$-40, %dl
	pxor	%xmm1, %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(less16bytes)
L(24bytes):
	movdqu	-24(%rdi), %xmm2
	movdqu	-24(%rsi), %xmm1
	mov	$-24, %dl
	pxor	%xmm1, %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(less16bytes)

	mov	-8(%rsi), %rcx
	mov	-8(%rdi), %rax
	cmp	%rax, %rcx
	jne	L(diffin8bytes)
	xor	%eax, %eax
	ret

# ifndef USE_AS_WMEMCMP
/* unreal cases for wmemcmp */
	.p2align 4
L(73bytes):
	movdqu	-73(%rsi), %xmm1
	movdqu	-73(%rdi), %xmm2
	mov	$-73, %dl
	pxor	%xmm1, %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(less16bytes)
L(57bytes):
	movdqu	-57(%rdi), %xmm2
	movdqu	-57(%rsi), %xmm1
	mov	$-57, %dl
	pxor	%xmm1, %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(less16bytes)
L(41bytes):
	movdqu	-41(%rdi), %xmm2
	movdqu	-41(%rsi), %xmm1
	mov	$-41, %dl
	pxor	%xmm1, %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(less16bytes)
L(25bytes):
	movdqu	-25(%rdi), %xmm2
	movdqu	-25(%rsi), %xmm1
	mov	$-25, %dl
	pxor	%xmm1, %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(less16bytes)
	mov	-9(%rdi), %rax
	mov	-9(%rsi), %rcx
	cmp	%rax, %rcx
	jne	L(diffin8bytes)
	movzbl	-1(%rdi), %eax
	movzbl	-1(%rsi), %ecx
	sub	%ecx, %eax
	ret

	.p2align 4
L(74bytes):
	movdqu	-74(%rsi), %xmm1
	movdqu	-74(%rdi), %xmm2
	mov	$-74, %dl
	pxor	%xmm1, %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(less16bytes)
L(58bytes):
	movdqu	-58(%rdi), %xmm2
	movdqu	-58(%rsi), %xmm1
	mov	$-58, %dl
	pxor	%xmm1, %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(less16bytes)
L(42bytes):
	movdqu	-42(%rdi), %xmm2
	movdqu	-42(%rsi), %xmm1
	mov	$-42, %dl
	pxor	%xmm1, %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(less16bytes)
L(26bytes):
	movdqu	-26(%rdi), %xmm2
	movdqu	-26(%rsi), %xmm1
	mov	$-26, %dl
	pxor	%xmm1, %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(less16bytes)
	mov	-10(%rdi), %rax
	mov	-10(%rsi), %rcx
	cmp	%rax, %rcx
	jne	L(diffin8bytes)
	movzwl	-2(%rdi), %eax
	movzwl	-2(%rsi), %ecx
	jmp	L(diffin2bytes)

	.p2align 4
L(75bytes):
	movdqu	-75(%rsi), %xmm1
	movdqu	-75(%rdi), %xmm2
	mov	$-75, %dl
	pxor	%xmm1, %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(less16bytes)
L(59bytes):
	movdqu	-59(%rdi), %xmm2
	movdqu	-59(%rsi), %xmm1
	mov	$-59, %dl
	pxor	%xmm1, %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(less16bytes)
L(43bytes):
	movdqu	-43(%rdi), %xmm2
	movdqu	-43(%rsi), %xmm1
	mov	$-43, %dl
	pxor	%xmm1, %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(less16bytes)
L(27bytes):
	movdqu	-27(%rdi), %xmm2
	movdqu	-27(%rsi), %xmm1
	mov	$-27, %dl
	pxor	%xmm1, %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(less16bytes)
	mov	-11(%rdi), %rax
	mov	-11(%rsi), %rcx
	cmp	%rax, %rcx
	jne	L(diffin8bytes)
	mov	-4(%rdi), %eax
	mov	-4(%rsi), %ecx
	cmp	%eax, %ecx
	jne	L(diffin4bytes)
	xor	%eax, %eax
	ret
# endif
	.p2align 4
L(76bytes):
	movdqu	-76(%rsi), %xmm1
	movdqu	-76(%rdi), %xmm2
	mov	$-76, %dl
	pxor	%xmm1, %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(less16bytes)
L(60bytes):
	movdqu	-60(%rdi), %xmm2
	movdqu	-60(%rsi), %xmm1
	mov	$-60, %dl
	pxor	%xmm1, %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(less16bytes)
L(44bytes):
	movdqu	-44(%rdi), %xmm2
	movdqu	-44(%rsi), %xmm1
	mov	$-44, %dl
	pxor	%xmm1, %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(less16bytes)
L(28bytes):
	movdqu	-28(%rdi), %xmm2
	movdqu	-28(%rsi), %xmm1
	mov	$-28, %dl
	pxor	%xmm1, %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(less16bytes)
	mov	-12(%rdi), %rax
	mov	-12(%rsi), %rcx
	cmp	%rax, %rcx
	jne	L(diffin8bytes)
	mov	-4(%rsi), %ecx
# ifndef USE_AS_WMEMCMP
	mov	-4(%rdi), %eax
	cmp	%eax, %ecx
# else
	cmp	-4(%rdi), %ecx
# endif
	jne	L(diffin4bytes)
	xor	%eax, %eax
	ret

# ifndef USE_AS_WMEMCMP
/* unreal cases for wmemcmp */
	.p2align 4
L(77bytes):
	movdqu	-77(%rsi), %xmm1
	movdqu	-77(%rdi), %xmm2
	mov	$-77, %dl
	pxor	%xmm1, %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(less16bytes)
L(61bytes):
	movdqu	-61(%rdi), %xmm2
	movdqu	-61(%rsi), %xmm1
	mov	$-61, %dl
	pxor	%xmm1, %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(less16bytes)
L(45bytes):
	movdqu	-45(%rdi), %xmm2
	movdqu	-45(%rsi), %xmm1
	mov	$-45, %dl
	pxor	%xmm1, %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(less16bytes)
L(29bytes):
	movdqu	-29(%rdi), %xmm2
	movdqu	-29(%rsi), %xmm1
	mov	$-29, %dl
	pxor	%xmm1, %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(less16bytes)

	mov	-13(%rdi), %rax
	mov	-13(%rsi), %rcx
	cmp	%rax, %rcx
	jne	L(diffin8bytes)

	mov	-8(%rdi), %rax
	mov	-8(%rsi), %rcx
	cmp	%rax, %rcx
	jne	L(diffin8bytes)
	xor	%eax, %eax
	ret

	.p2align 4
L(78bytes):
	movdqu	-78(%rsi), %xmm1
	movdqu	-78(%rdi), %xmm2
	mov	$-78, %dl
	pxor	%xmm1, %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(less16bytes)
L(62bytes):
	movdqu	-62(%rdi), %xmm2
	movdqu	-62(%rsi), %xmm1
	mov	$-62, %dl
	pxor	%xmm1, %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(less16bytes)
L(46bytes):
	movdqu	-46(%rdi), %xmm2
	movdqu	-46(%rsi), %xmm1
	mov	$-46, %dl
	pxor	%xmm1, %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(less16bytes)
L(30bytes):
	movdqu	-30(%rdi), %xmm2
	movdqu	-30(%rsi), %xmm1
	mov	$-30, %dl
	pxor	%xmm1, %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(less16bytes)
	mov	-14(%rdi), %rax
	mov	-14(%rsi), %rcx
	cmp	%rax, %rcx
	jne	L(diffin8bytes)
	mov	-8(%rdi), %rax
	mov	-8(%rsi), %rcx
	cmp	%rax, %rcx
	jne	L(diffin8bytes)
	xor	%eax, %eax
	ret

	.p2align 4
L(79bytes):
	movdqu	-79(%rsi), %xmm1
	movdqu	-79(%rdi), %xmm2
	mov	$-79, %dl
	pxor	%xmm1, %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(less16bytes)
L(63bytes):
	movdqu	-63(%rdi), %xmm2
	movdqu	-63(%rsi), %xmm1
	mov	$-63, %dl
	pxor	%xmm1, %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(less16bytes)
L(47bytes):
	movdqu	-47(%rdi), %xmm2
	movdqu	-47(%rsi), %xmm1
	mov	$-47, %dl
	pxor	%xmm1, %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(less16bytes)
L(31bytes):
	movdqu	-31(%rdi), %xmm2
	movdqu	-31(%rsi), %xmm1
	mov	$-31, %dl
	pxor	%xmm1, %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(less16bytes)
	mov	-15(%rdi), %rax
	mov	-15(%rsi), %rcx
	cmp	%rax, %rcx
	jne	L(diffin8bytes)
	mov	-8(%rdi), %rax
	mov	-8(%rsi), %rcx
	cmp	%rax, %rcx
	jne	L(diffin8bytes)
	xor	%eax, %eax
	ret
# endif
	.p2align 4
L(64bytes):
	movdqu	-64(%rdi), %xmm2
	movdqu	-64(%rsi), %xmm1
	mov	$-64, %dl
	pxor	%xmm1, %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(less16bytes)
L(48bytes):
	movdqu	-48(%rdi), %xmm2
	movdqu	-48(%rsi), %xmm1
	mov	$-48, %dl
	pxor	%xmm1, %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(less16bytes)
L(32bytes):
	movdqu	-32(%rdi), %xmm2
	movdqu	-32(%rsi), %xmm1
	mov	$-32, %dl
	pxor	%xmm1, %xmm2
	ptest	%xmm2, %xmm0
	jnc	L(less16bytes)

	mov	-16(%rdi), %rax
	mov	-16(%rsi), %rcx
	cmp	%rax, %rcx
	jne	L(diffin8bytes)

	mov	-8(%rdi), %rax
	mov	-8(%rsi), %rcx
	cmp	%rax, %rcx
	jne	L(diffin8bytes)
	xor	%eax, %eax
	ret

/*
 * Aligned 8 bytes to avoid 2 branch "taken" in one 16 alinged code block.
 */
	.p2align 3
L(less16bytes):
	movsbq	%dl, %rdx
	mov	(%rsi, %rdx), %rcx
	mov	(%rdi, %rdx), %rax
	cmp	%rax, %rcx
	jne	L(diffin8bytes)
	mov	8(%rsi, %rdx), %rcx
	mov	8(%rdi, %rdx), %rax
L(diffin8bytes):
	cmp	%eax, %ecx
	jne	L(diffin4bytes)
	shr	$32, %rcx
	shr	$32, %rax

# ifdef USE_AS_WMEMCMP
/* for wmemcmp */
	cmp	%eax, %ecx
	jne	L(diffin4bytes)
	xor	%eax, %eax
	ret
# endif

L(diffin4bytes):
# ifndef USE_AS_WMEMCMP
	cmp	%cx, %ax
	jne	L(diffin2bytes)
	shr	$16, %ecx
	shr	$16, %eax
L(diffin2bytes):
	cmp	%cl, %al
	jne	L(end)
	and	$0xffff, %eax
	and	$0xffff, %ecx
	sub	%ecx, %eax
	ret

	.p2align 4
L(end):
	and	$0xff, %eax
	and	$0xff, %ecx
	sub	%ecx, %eax
	ret
# else

/* for wmemcmp */
	mov	$1, %eax
	jl	L(nequal_bigger)
	neg	%eax
	ret

	.p2align 4
L(nequal_bigger):
	ret

L(unreal_case):
	xor	%eax, %eax
	ret
# endif

END (MEMCMP)

	.section .rodata.sse4.1,"a",@progbits
	.p2align 3
# ifndef USE_AS_WMEMCMP
L(table_64bytes):
	.int	JMPTBL (L(0bytes), L(table_64bytes))
	.int	JMPTBL (L(1bytes), L(table_64bytes))
	.int	JMPTBL (L(2bytes), L(table_64bytes))
	.int	JMPTBL (L(3bytes), L(table_64bytes))
	.int	JMPTBL (L(4bytes), L(table_64bytes))
	.int	JMPTBL (L(5bytes), L(table_64bytes))
	.int	JMPTBL (L(6bytes), L(table_64bytes))
	.int	JMPTBL (L(7bytes), L(table_64bytes))
	.int	JMPTBL (L(8bytes), L(table_64bytes))
	.int	JMPTBL (L(9bytes), L(table_64bytes))
	.int	JMPTBL (L(10bytes), L(table_64bytes))
	.int	JMPTBL (L(11bytes), L(table_64bytes))
	.int	JMPTBL (L(12bytes), L(table_64bytes))
	.int	JMPTBL (L(13bytes), L(table_64bytes))
	.int	JMPTBL (L(14bytes), L(table_64bytes))
	.int	JMPTBL (L(15bytes), L(table_64bytes))
	.int	JMPTBL (L(16bytes), L(table_64bytes))
	.int	JMPTBL (L(17bytes), L(table_64bytes))
	.int	JMPTBL (L(18bytes), L(table_64bytes))
	.int	JMPTBL (L(19bytes), L(table_64bytes))
	.int	JMPTBL (L(20bytes), L(table_64bytes))
	.int	JMPTBL (L(21bytes), L(table_64bytes))
	.int	JMPTBL (L(22bytes), L(table_64bytes))
	.int	JMPTBL (L(23bytes), L(table_64bytes))
	.int	JMPTBL (L(24bytes), L(table_64bytes))
	.int	JMPTBL (L(25bytes), L(table_64bytes))
	.int	JMPTBL (L(26bytes), L(table_64bytes))
	.int	JMPTBL (L(27bytes), L(table_64bytes))
	.int	JMPTBL (L(28bytes), L(table_64bytes))
	.int	JMPTBL (L(29bytes), L(table_64bytes))
	.int	JMPTBL (L(30bytes), L(table_64bytes))
	.int	JMPTBL (L(31bytes), L(table_64bytes))
	.int	JMPTBL (L(32bytes), L(table_64bytes))
	.int	JMPTBL (L(33bytes), L(table_64bytes))
	.int	JMPTBL (L(34bytes), L(table_64bytes))
	.int	JMPTBL (L(35bytes), L(table_64bytes))
	.int	JMPTBL (L(36bytes), L(table_64bytes))
	.int	JMPTBL (L(37bytes), L(table_64bytes))
	.int	JMPTBL (L(38bytes), L(table_64bytes))
	.int	JMPTBL (L(39bytes), L(table_64bytes))
	.int	JMPTBL (L(40bytes), L(table_64bytes))
	.int	JMPTBL (L(41bytes), L(table_64bytes))
	.int	JMPTBL (L(42bytes), L(table_64bytes))
	.int	JMPTBL (L(43bytes), L(table_64bytes))
	.int	JMPTBL (L(44bytes), L(table_64bytes))
	.int	JMPTBL (L(45bytes), L(table_64bytes))
	.int	JMPTBL (L(46bytes), L(table_64bytes))
	.int	JMPTBL (L(47bytes), L(table_64bytes))
	.int	JMPTBL (L(48bytes), L(table_64bytes))
	.int	JMPTBL (L(49bytes), L(table_64bytes))
	.int	JMPTBL (L(50bytes), L(table_64bytes))
	.int	JMPTBL (L(51bytes), L(table_64bytes))
	.int	JMPTBL (L(52bytes), L(table_64bytes))
	.int	JMPTBL (L(53bytes), L(table_64bytes))
	.int	JMPTBL (L(54bytes), L(table_64bytes))
	.int	JMPTBL (L(55bytes), L(table_64bytes))
	.int	JMPTBL (L(56bytes), L(table_64bytes))
	.int	JMPTBL (L(57bytes), L(table_64bytes))
	.int	JMPTBL (L(58bytes), L(table_64bytes))
	.int	JMPTBL (L(59bytes), L(table_64bytes))
	.int	JMPTBL (L(60bytes), L(table_64bytes))
	.int	JMPTBL (L(61bytes), L(table_64bytes))
	.int	JMPTBL (L(62bytes), L(table_64bytes))
	.int	JMPTBL (L(63bytes), L(table_64bytes))
	.int	JMPTBL (L(64bytes), L(table_64bytes))
	.int	JMPTBL (L(65bytes), L(table_64bytes))
	.int	JMPTBL (L(66bytes), L(table_64bytes))
	.int	JMPTBL (L(67bytes), L(table_64bytes))
	.int	JMPTBL (L(68bytes), L(table_64bytes))
	.int	JMPTBL (L(69bytes), L(table_64bytes))
	.int	JMPTBL (L(70bytes), L(table_64bytes))
	.int	JMPTBL (L(71bytes), L(table_64bytes))
	.int	JMPTBL (L(72bytes), L(table_64bytes))
	.int	JMPTBL (L(73bytes), L(table_64bytes))
	.int	JMPTBL (L(74bytes), L(table_64bytes))
	.int	JMPTBL (L(75bytes), L(table_64bytes))
	.int	JMPTBL (L(76bytes), L(table_64bytes))
	.int	JMPTBL (L(77bytes), L(table_64bytes))
	.int	JMPTBL (L(78bytes), L(table_64bytes))
	.int	JMPTBL (L(79bytes), L(table_64bytes))
# else
L(table_64bytes):
	.int	JMPTBL (L(0bytes), L(table_64bytes))
	.int	JMPTBL (L(unreal_case), L(table_64bytes))
	.int	JMPTBL (L(unreal_case), L(table_64bytes))
	.int	JMPTBL (L(unreal_case), L(table_64bytes))
	.int	JMPTBL (L(4bytes), L(table_64bytes))
	.int	JMPTBL (L(unreal_case), L(table_64bytes))
	.int	JMPTBL (L(unreal_case), L(table_64bytes))
	.int	JMPTBL (L(unreal_case), L(table_64bytes))
	.int	JMPTBL (L(8bytes), L(table_64bytes))
	.int	JMPTBL (L(unreal_case), L(table_64bytes))
	.int	JMPTBL (L(unreal_case), L(table_64bytes))
	.int	JMPTBL (L(unreal_case), L(table_64bytes))
	.int	JMPTBL (L(12bytes), L(table_64bytes))
	.int	JMPTBL (L(unreal_case), L(table_64bytes))
	.int	JMPTBL (L(unreal_case), L(table_64bytes))
	.int	JMPTBL (L(unreal_case), L(table_64bytes))
	.int	JMPTBL (L(16bytes), L(table_64bytes))
	.int	JMPTBL (L(unreal_case), L(table_64bytes))
	.int	JMPTBL (L(unreal_case), L(table_64bytes))
	.int	JMPTBL (L(unreal_case), L(table_64bytes))
	.int	JMPTBL (L(20bytes), L(table_64bytes))
	.int	JMPTBL (L(unreal_case), L(table_64bytes))
	.int	JMPTBL (L(unreal_case), L(table_64bytes))
	.int	JMPTBL (L(unreal_case), L(table_64bytes))
	.int	JMPTBL (L(24bytes), L(table_64bytes))
	.int	JMPTBL (L(unreal_case), L(table_64bytes))
	.int	JMPTBL (L(unreal_case), L(table_64bytes))
	.int	JMPTBL (L(unreal_case), L(table_64bytes))
	.int	JMPTBL (L(28bytes), L(table_64bytes))
	.int	JMPTBL (L(unreal_case), L(table_64bytes))
	.int	JMPTBL (L(unreal_case), L(table_64bytes))
	.int	JMPTBL (L(unreal_case), L(table_64bytes))
	.int	JMPTBL (L(32bytes), L(table_64bytes))
	.int	JMPTBL (L(unreal_case), L(table_64bytes))
	.int	JMPTBL (L(unreal_case), L(table_64bytes))
	.int	JMPTBL (L(unreal_case), L(table_64bytes))
	.int	JMPTBL (L(36bytes), L(table_64bytes))
	.int	JMPTBL (L(unreal_case), L(table_64bytes))
	.int	JMPTBL (L(unreal_case), L(table_64bytes))
	.int	JMPTBL (L(unreal_case), L(table_64bytes))
	.int	JMPTBL (L(40bytes), L(table_64bytes))
	.int	JMPTBL (L(unreal_case), L(table_64bytes))
	.int	JMPTBL (L(unreal_case), L(table_64bytes))
	.int	JMPTBL (L(unreal_case), L(table_64bytes))
	.int	JMPTBL (L(44bytes), L(table_64bytes))
	.int	JMPTBL (L(unreal_case), L(table_64bytes))
	.int	JMPTBL (L(unreal_case), L(table_64bytes))
	.int	JMPTBL (L(unreal_case), L(table_64bytes))
	.int	JMPTBL (L(48bytes), L(table_64bytes))
	.int	JMPTBL (L(unreal_case), L(table_64bytes))
	.int	JMPTBL (L(unreal_case), L(table_64bytes))
	.int	JMPTBL (L(unreal_case), L(table_64bytes))
	.int	JMPTBL (L(52bytes), L(table_64bytes))
	.int	JMPTBL (L(unreal_case), L(table_64bytes))
	.int	JMPTBL (L(unreal_case), L(table_64bytes))
	.int	JMPTBL (L(unreal_case), L(table_64bytes))
	.int	JMPTBL (L(56bytes), L(table_64bytes))
	.int	JMPTBL (L(unreal_case), L(table_64bytes))
	.int	JMPTBL (L(unreal_case), L(table_64bytes))
	.int	JMPTBL (L(unreal_case), L(table_64bytes))
	.int	JMPTBL (L(60bytes), L(table_64bytes))
	.int	JMPTBL (L(unreal_case), L(table_64bytes))
	.int	JMPTBL (L(unreal_case), L(table_64bytes))
	.int	JMPTBL (L(unreal_case), L(table_64bytes))
	.int	JMPTBL (L(64bytes), L(table_64bytes))
	.int	JMPTBL (L(unreal_case), L(table_64bytes))
	.int	JMPTBL (L(unreal_case), L(table_64bytes))
	.int	JMPTBL (L(unreal_case), L(table_64bytes))
	.int	JMPTBL (L(68bytes), L(table_64bytes))
	.int	JMPTBL (L(unreal_case), L(table_64bytes))
	.int	JMPTBL (L(unreal_case), L(table_64bytes))
	.int	JMPTBL (L(unreal_case), L(table_64bytes))
	.int	JMPTBL (L(72bytes), L(table_64bytes))
	.int	JMPTBL (L(unreal_case), L(table_64bytes))
	.int	JMPTBL (L(unreal_case), L(table_64bytes))
	.int	JMPTBL (L(unreal_case), L(table_64bytes))
	.int	JMPTBL (L(76bytes), L(table_64bytes))
	.int	JMPTBL (L(unreal_case), L(table_64bytes))
	.int	JMPTBL (L(unreal_case), L(table_64bytes))
	.int	JMPTBL (L(unreal_case), L(table_64bytes))
# endif
#endif
