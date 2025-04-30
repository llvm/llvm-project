/* Optimized wcscmp for x86-64 with SSE2.
   Copyright (C) 2011-2021 Free Software Foundation, Inc.
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

/* Note: wcscmp uses signed comparison, not unsighed as in strcmp function. */

	.text
ENTRY (__wcscmp)
/*
	* This implementation uses SSE to compare up to 16 bytes at a time.
*/
	mov	%esi, %eax
	mov	%edi, %edx
	pxor	%xmm0, %xmm0		/* clear %xmm0 for null char checks */
	mov	%al, %ch
	mov	%dl, %cl
	and	$63, %eax		/* rsi alignment in cache line */
	and	$63, %edx		/* rdi alignment in cache line */
	and	$15, %cl
	jz	L(continue_00)
	cmp	$16, %edx
	jb	L(continue_0)
	cmp	$32, %edx
	jb	L(continue_16)
	cmp	$48, %edx
	jb	L(continue_32)

L(continue_48):
	and	$15, %ch
	jz	L(continue_48_00)
	cmp	$16, %eax
	jb	L(continue_0_48)
	cmp	$32, %eax
	jb	L(continue_16_48)
	cmp	$48, %eax
	jb	L(continue_32_48)

	.p2align 4
L(continue_48_48):
	mov	(%rsi), %ecx
	cmp	%ecx, (%rdi)
	jne	L(nequal)
	test	%ecx, %ecx
	jz	L(equal)

	mov	4(%rsi), %ecx
	cmp	%ecx, 4(%rdi)
	jne	L(nequal)
	test	%ecx, %ecx
	jz	L(equal)

	mov	8(%rsi), %ecx
	cmp	%ecx, 8(%rdi)
	jne	L(nequal)
	test	%ecx, %ecx
	jz	L(equal)

	mov	12(%rsi), %ecx
	cmp	%ecx, 12(%rdi)
	jne	L(nequal)
	test	%ecx, %ecx
	jz	L(equal)

	movdqu	16(%rdi), %xmm1
	movdqu	16(%rsi), %xmm2
	pcmpeqd	%xmm1, %xmm0		/* Any null double_word? */
	pcmpeqd	%xmm2, %xmm1		/* compare first 4 double_words for equality */
	psubb	%xmm0, %xmm1		/* packed sub of comparison results*/
	pmovmskb %xmm1, %edx
	sub	$0xffff, %edx		/* if first 4 double_words are same, edx == 0xffff */
	jnz	L(less4_double_words_16)

	movdqu	32(%rdi), %xmm1
	movdqu	32(%rsi), %xmm2
	pcmpeqd	%xmm1, %xmm0		/* Any null double_word? */
	pcmpeqd	%xmm2, %xmm1		/* compare first 4 double_words for equality */
	psubb	%xmm0, %xmm1		/* packed sub of comparison results*/
	pmovmskb %xmm1, %edx
	sub	$0xffff, %edx		/* if first 4 double_words are same, edx == 0xffff */
	jnz	L(less4_double_words_32)

	movdqu	48(%rdi), %xmm1
	movdqu	48(%rsi), %xmm2
	pcmpeqd	%xmm1, %xmm0		/* Any null double_word? */
	pcmpeqd	%xmm2, %xmm1		/* compare first 4 double_words for equality */
	psubb	%xmm0, %xmm1		/* packed sub of comparison results*/
	pmovmskb %xmm1, %edx
	sub	$0xffff, %edx		/* if first 4 double_words are same, edx == 0xffff */
	jnz	L(less4_double_words_48)

	add	$64, %rsi
	add	$64, %rdi
	jmp	L(continue_48_48)

L(continue_0):
	and	$15, %ch
	jz	L(continue_0_00)
	cmp	$16, %eax
	jb	L(continue_0_0)
	cmp	$32, %eax
	jb	L(continue_0_16)
	cmp	$48, %eax
	jb	L(continue_0_32)

	.p2align 4
L(continue_0_48):
	mov	(%rsi), %ecx
	cmp	%ecx, (%rdi)
	jne	L(nequal)
	test	%ecx, %ecx
	jz	L(equal)

	mov	4(%rsi), %ecx
	cmp	%ecx, 4(%rdi)
	jne	L(nequal)
	test	%ecx, %ecx
	jz	L(equal)

	mov	8(%rsi), %ecx
	cmp	%ecx, 8(%rdi)
	jne	L(nequal)
	test	%ecx, %ecx
	jz	L(equal)

	mov	12(%rsi), %ecx
	cmp	%ecx, 12(%rdi)
	jne	L(nequal)
	test	%ecx, %ecx
	jz	L(equal)

	movdqu	16(%rdi), %xmm1
	movdqu	16(%rsi), %xmm2
	pcmpeqd	%xmm1, %xmm0		/* Any null double_word? */
	pcmpeqd	%xmm2, %xmm1		/* compare first 4 double_words for equality */
	psubb	%xmm0, %xmm1		/* packed sub of comparison results*/
	pmovmskb %xmm1, %edx
	sub	$0xffff, %edx		/* if first 4 double_words are same, edx == 0xffff */
	jnz	L(less4_double_words_16)

	movdqu	32(%rdi), %xmm1
	movdqu	32(%rsi), %xmm2
	pcmpeqd	%xmm1, %xmm0		/* Any null double_word? */
	pcmpeqd	%xmm2, %xmm1		/* compare first 4 double_words for equality */
	psubb	%xmm0, %xmm1		/* packed sub of comparison results*/
	pmovmskb %xmm1, %edx
	sub	$0xffff, %edx		/* if first 4 double_words are same, edx == 0xffff */
	jnz	L(less4_double_words_32)

	mov	48(%rsi), %ecx
	cmp	%ecx, 48(%rdi)
	jne	L(nequal)
	test	%ecx, %ecx
	jz	L(equal)

	mov	52(%rsi), %ecx
	cmp	%ecx, 52(%rdi)
	jne	L(nequal)
	test	%ecx, %ecx
	jz	L(equal)

	mov	56(%rsi), %ecx
	cmp	%ecx, 56(%rdi)
	jne	L(nequal)
	test	%ecx, %ecx
	jz	L(equal)

	mov	60(%rsi), %ecx
	cmp	%ecx, 60(%rdi)
	jne	L(nequal)
	test	%ecx, %ecx
	jz	L(equal)

	add	$64, %rsi
	add	$64, %rdi
	jmp	L(continue_0_48)

	.p2align 4
L(continue_00):
	and	$15, %ch
	jz	L(continue_00_00)
	cmp	$16, %eax
	jb	L(continue_00_0)
	cmp	$32, %eax
	jb	L(continue_00_16)
	cmp	$48, %eax
	jb	L(continue_00_32)

	.p2align 4
L(continue_00_48):
	pcmpeqd	(%rdi), %xmm0
	mov	(%rdi), %eax
	pmovmskb %xmm0, %ecx
	test	%ecx, %ecx
	jnz	L(less4_double_words1)

	cmp	(%rsi), %eax
	jne	L(nequal)

	mov	4(%rdi), %eax
	cmp	4(%rsi), %eax
	jne	L(nequal)

	mov	8(%rdi), %eax
	cmp	8(%rsi), %eax
	jne	L(nequal)

	mov	12(%rdi), %eax
	cmp	12(%rsi), %eax
	jne	L(nequal)

	movdqu	16(%rsi), %xmm2
	pcmpeqd	%xmm2, %xmm0		/* Any null double_word? */
	pcmpeqd	16(%rdi), %xmm2		/* compare first 4 double_words for equality */
	psubb	%xmm0, %xmm2		/* packed sub of comparison results*/
	pmovmskb %xmm2, %edx
	sub	$0xffff, %edx		/* if first 4 double_words are same, edx == 0xffff */
	jnz	L(less4_double_words_16)

	movdqu	32(%rsi), %xmm2
	pcmpeqd	%xmm2, %xmm0		/* Any null double_word? */
	pcmpeqd	32(%rdi), %xmm2		/* compare first 4 double_words for equality */
	psubb	%xmm0, %xmm2		/* packed sub of comparison results*/
	pmovmskb %xmm2, %edx
	sub	$0xffff, %edx		/* if first 4 double_words are same, edx == 0xffff */
	jnz	L(less4_double_words_32)

	movdqu	48(%rsi), %xmm2
	pcmpeqd	%xmm2, %xmm0		/* Any null double_word? */
	pcmpeqd	48(%rdi), %xmm2		/* compare first 4 double_words for equality */
	psubb	%xmm0, %xmm2		/* packed sub of comparison results*/
	pmovmskb %xmm2, %edx
	sub	$0xffff, %edx		/* if first 4 double_words are same, edx == 0xffff */
	jnz	L(less4_double_words_48)

	add	$64, %rsi
	add	$64, %rdi
	jmp	L(continue_00_48)

	.p2align 4
L(continue_32):
	and	$15, %ch
	jz	L(continue_32_00)
	cmp	$16, %eax
	jb	L(continue_0_32)
	cmp	$32, %eax
	jb	L(continue_16_32)
	cmp	$48, %eax
	jb	L(continue_32_32)

	.p2align 4
L(continue_32_48):
	mov	(%rsi), %ecx
	cmp	%ecx, (%rdi)
	jne	L(nequal)
	test	%ecx, %ecx
	jz	L(equal)

	mov	4(%rsi), %ecx
	cmp	%ecx, 4(%rdi)
	jne	L(nequal)
	test	%ecx, %ecx
	jz	L(equal)

	mov	8(%rsi), %ecx
	cmp	%ecx, 8(%rdi)
	jne	L(nequal)
	test	%ecx, %ecx
	jz	L(equal)

	mov	12(%rsi), %ecx
	cmp	%ecx, 12(%rdi)
	jne	L(nequal)
	test	%ecx, %ecx
	jz	L(equal)

	mov	16(%rsi), %ecx
	cmp	%ecx, 16(%rdi)
	jne	L(nequal)
	test	%ecx, %ecx
	jz	L(equal)

	mov	20(%rsi), %ecx
	cmp	%ecx, 20(%rdi)
	jne	L(nequal)
	test	%ecx, %ecx
	jz	L(equal)

	mov	24(%rsi), %ecx
	cmp	%ecx, 24(%rdi)
	jne	L(nequal)
	test	%ecx, %ecx
	jz	L(equal)

	mov	28(%rsi), %ecx
	cmp	%ecx, 28(%rdi)
	jne	L(nequal)
	test	%ecx, %ecx
	jz	L(equal)

	movdqu	32(%rdi), %xmm1
	movdqu	32(%rsi), %xmm2
	pcmpeqd	%xmm1, %xmm0		/* Any null double_word? */
	pcmpeqd	%xmm2, %xmm1		/* compare first 4 double_words for equality */
	psubb	%xmm0, %xmm1		/* packed sub of comparison results*/
	pmovmskb %xmm1, %edx
	sub	$0xffff, %edx		/* if first 4 double_words are same, edx == 0xffff */
	jnz	L(less4_double_words_32)

	movdqu	48(%rdi), %xmm1
	movdqu	48(%rsi), %xmm2
	pcmpeqd	%xmm1, %xmm0		/* Any null double_word? */
	pcmpeqd	%xmm2, %xmm1		/* compare first 4 double_words for equality */
	psubb	%xmm0, %xmm1		/* packed sub of comparison results*/
	pmovmskb %xmm1, %edx
	sub	$0xffff, %edx		/* if first 4 double_words are same, edx == 0xffff */
	jnz	L(less4_double_words_48)

	add	$64, %rsi
	add	$64, %rdi
	jmp	L(continue_32_48)

	.p2align 4
L(continue_16):
	and	$15, %ch
	jz	L(continue_16_00)
	cmp	$16, %eax
	jb	L(continue_0_16)
	cmp	$32, %eax
	jb	L(continue_16_16)
	cmp	$48, %eax
	jb	L(continue_16_32)

	.p2align 4
L(continue_16_48):
	mov	(%rsi), %ecx
	cmp	%ecx, (%rdi)
	jne	L(nequal)
	test	%ecx, %ecx
	jz	L(equal)

	mov	4(%rsi), %ecx
	cmp	%ecx, 4(%rdi)
	jne	L(nequal)
	test	%ecx, %ecx
	jz	L(equal)

	mov	8(%rsi), %ecx
	cmp	%ecx, 8(%rdi)
	jne	L(nequal)
	test	%ecx, %ecx
	jz	L(equal)

	mov	12(%rsi), %ecx
	cmp	%ecx, 12(%rdi)
	jne	L(nequal)
	test	%ecx, %ecx
	jz	L(equal)

	movdqu	16(%rdi), %xmm1
	movdqu	16(%rsi), %xmm2
	pcmpeqd	%xmm1, %xmm0		/* Any null double_word? */
	pcmpeqd	%xmm2, %xmm1		/* compare first 4 double_words for equality */
	psubb	%xmm0, %xmm1		/* packed sub of comparison results*/
	pmovmskb %xmm1, %edx
	sub	$0xffff, %edx		/* if first 4 double_words are same, edx == 0xffff */
	jnz	L(less4_double_words_16)

	mov	32(%rsi), %ecx
	cmp	%ecx, 32(%rdi)
	jne	L(nequal)
	test	%ecx, %ecx
	jz	L(equal)

	mov	36(%rsi), %ecx
	cmp	%ecx, 36(%rdi)
	jne	L(nequal)
	test	%ecx, %ecx
	jz	L(equal)

	mov	40(%rsi), %ecx
	cmp	%ecx, 40(%rdi)
	jne	L(nequal)
	test	%ecx, %ecx
	jz	L(equal)

	mov	44(%rsi), %ecx
	cmp	%ecx, 44(%rdi)
	jne	L(nequal)
	test	%ecx, %ecx
	jz	L(equal)

	movdqu	48(%rdi), %xmm1
	movdqu	48(%rsi), %xmm2
	pcmpeqd	%xmm1, %xmm0		/* Any null double_word? */
	pcmpeqd	%xmm2, %xmm1		/* compare first 4 double_words for equality */
	psubb	%xmm0, %xmm1		/* packed sub of comparison results*/
	pmovmskb %xmm1, %edx
	sub	$0xffff, %edx		/* if first 4 double_words are same, edx == 0xffff */
	jnz	L(less4_double_words_48)

	add	$64, %rsi
	add	$64, %rdi
	jmp	L(continue_16_48)

	.p2align 4
L(continue_00_00):
	movdqa	(%rdi), %xmm1
	pcmpeqd	%xmm1, %xmm0		/* Any null double_word? */
	pcmpeqd	(%rsi), %xmm1		/* compare first 4 double_words for equality */
	psubb	%xmm0, %xmm1		/* packed sub of comparison results*/
	pmovmskb %xmm1, %edx
	sub	$0xffff, %edx		/* if first 4 double_words are same, edx == 0xffff */
	jnz	L(less4_double_words)

	movdqa	16(%rdi), %xmm3
	pcmpeqd	%xmm3, %xmm0		/* Any null double_word? */
	pcmpeqd	16(%rsi), %xmm3		/* compare first 4 double_words for equality */
	psubb	%xmm0, %xmm3		/* packed sub of comparison results*/
	pmovmskb %xmm3, %edx
	sub	$0xffff, %edx		/* if first 4 double_words are same, edx == 0xffff */
	jnz	L(less4_double_words_16)

	movdqa	32(%rdi), %xmm5
	pcmpeqd	%xmm5, %xmm0		/* Any null double_word? */
	pcmpeqd	32(%rsi), %xmm5		/* compare first 4 double_words for equality */
	psubb	%xmm0, %xmm5		/* packed sub of comparison results*/
	pmovmskb %xmm5, %edx
	sub	$0xffff, %edx		/* if first 4 double_words are same, edx == 0xffff */
	jnz	L(less4_double_words_32)

	movdqa	48(%rdi), %xmm1
	pcmpeqd	%xmm1, %xmm0		/* Any null double_word? */
	pcmpeqd	48(%rsi), %xmm1		/* compare first 4 double_words for equality */
	psubb	%xmm0, %xmm1		/* packed sub of comparison results*/
	pmovmskb %xmm1, %edx
	sub	$0xffff, %edx		/* if first 4 double_words are same, edx == 0xffff */
	jnz	L(less4_double_words_48)

	add	$64, %rsi
	add	$64, %rdi
	jmp	L(continue_00_00)

	.p2align 4
L(continue_00_32):
	movdqu	(%rsi), %xmm2
	pcmpeqd	%xmm2, %xmm0		/* Any null double_word? */
	pcmpeqd	(%rdi), %xmm2		/* compare first 4 double_words for equality */
	psubb	%xmm0, %xmm2		/* packed sub of comparison results*/
	pmovmskb %xmm2, %edx
	sub	$0xffff, %edx		/* if first 4 double_words are same, edx == 0xffff */
	jnz	L(less4_double_words)

	add	$16, %rsi
	add	$16, %rdi
	jmp	L(continue_00_48)

	.p2align 4
L(continue_00_16):
	movdqu	(%rsi), %xmm2
	pcmpeqd	%xmm2, %xmm0		/* Any null double_word? */
	pcmpeqd	(%rdi), %xmm2		/* compare first 4 double_words for equality */
	psubb	%xmm0, %xmm2		/* packed sub of comparison results*/
	pmovmskb %xmm2, %edx
	sub	$0xffff, %edx		/* if first 4 double_words are same, edx == 0xffff */
	jnz	L(less4_double_words)

	movdqu	16(%rsi), %xmm2
	pcmpeqd	%xmm2, %xmm0		/* Any null double_word? */
	pcmpeqd	16(%rdi), %xmm2		/* compare first 4 double_words for equality */
	psubb	%xmm0, %xmm2		/* packed sub of comparison results*/
	pmovmskb %xmm2, %edx
	sub	$0xffff, %edx		/* if first 4 double_words are same, edx == 0xffff */
	jnz	L(less4_double_words_16)

	add	$32, %rsi
	add	$32, %rdi
	jmp	L(continue_00_48)

	.p2align 4
L(continue_00_0):
	movdqu	(%rsi), %xmm2
	pcmpeqd	%xmm2, %xmm0		/* Any null double_word? */
	pcmpeqd	(%rdi), %xmm2		/* compare first 4 double_words for equality */
	psubb	%xmm0, %xmm2		/* packed sub of comparison results*/
	pmovmskb %xmm2, %edx
	sub	$0xffff, %edx		/* if first 4 double_words are same, edx == 0xffff */
	jnz	L(less4_double_words)

	movdqu	16(%rsi), %xmm2
	pcmpeqd	%xmm2, %xmm0		/* Any null double_word? */
	pcmpeqd	16(%rdi), %xmm2		/* compare first 4 double_words for equality */
	psubb	%xmm0, %xmm2		/* packed sub of comparison results*/
	pmovmskb %xmm2, %edx
	sub	$0xffff, %edx		/* if first 4 double_words are same, edx == 0xffff */
	jnz	L(less4_double_words_16)

	movdqu	32(%rsi), %xmm2
	pcmpeqd	%xmm2, %xmm0		/* Any null double_word? */
	pcmpeqd	32(%rdi), %xmm2		/* compare first 4 double_words for equality */
	psubb	%xmm0, %xmm2		/* packed sub of comparison results*/
	pmovmskb %xmm2, %edx
	sub	$0xffff, %edx		/* if first 4 double_words are same, edx == 0xffff */
	jnz	L(less4_double_words_32)

	add	$48, %rsi
	add	$48, %rdi
	jmp	L(continue_00_48)

	.p2align 4
L(continue_48_00):
	pcmpeqd	(%rsi), %xmm0
	mov	(%rdi), %eax
	pmovmskb %xmm0, %ecx
	test	%ecx, %ecx
	jnz	L(less4_double_words1)

	cmp	(%rsi), %eax
	jne	L(nequal)

	mov	4(%rdi), %eax
	cmp	4(%rsi), %eax
	jne	L(nequal)

	mov	8(%rdi), %eax
	cmp	8(%rsi), %eax
	jne	L(nequal)

	mov	12(%rdi), %eax
	cmp	12(%rsi), %eax
	jne	L(nequal)

	movdqu	16(%rdi), %xmm1
	pcmpeqd	%xmm1, %xmm0		/* Any null double_word? */
	pcmpeqd	16(%rsi), %xmm1		/* compare first 4 double_words for equality */
	psubb	%xmm0, %xmm1		/* packed sub of comparison results*/
	pmovmskb %xmm1, %edx
	sub	$0xffff, %edx		/* if first 4 double_words are same, edx == 0xffff */
	jnz	L(less4_double_words_16)

	movdqu	32(%rdi), %xmm1
	pcmpeqd	%xmm1, %xmm0		/* Any null double_word? */
	pcmpeqd	32(%rsi), %xmm1		/* compare first 4 double_words for equality */
	psubb	%xmm0, %xmm1		/* packed sub of comparison results*/
	pmovmskb %xmm1, %edx
	sub	$0xffff, %edx		/* if first 4 double_words are same, edx == 0xffff */
	jnz	L(less4_double_words_32)

	movdqu	48(%rdi), %xmm1
	pcmpeqd	%xmm1, %xmm0		/* Any null double_word? */
	pcmpeqd	48(%rsi), %xmm1		/* compare first 4 double_words for equality */
	psubb	%xmm0, %xmm1		/* packed sub of comparison results*/
	pmovmskb %xmm1, %edx
	sub	$0xffff, %edx		/* if first 4 double_words are same, edx == 0xffff */
	jnz	L(less4_double_words_48)

	add	$64, %rsi
	add	$64, %rdi
	jmp	L(continue_48_00)

	.p2align 4
L(continue_32_00):
	movdqu	(%rdi), %xmm1
	pcmpeqd	%xmm1, %xmm0		/* Any null double_word? */
	pcmpeqd	(%rsi), %xmm1		/* compare first 4 double_words for equality */
	psubb	%xmm0, %xmm1		/* packed sub of comparison results*/
	pmovmskb %xmm1, %edx
	sub	$0xffff, %edx		/* if first 4 double_words are same, edx == 0xffff */
	jnz	L(less4_double_words)

	add	$16, %rsi
	add	$16, %rdi
	jmp	L(continue_48_00)

	.p2align 4
L(continue_16_00):
	movdqu	(%rdi), %xmm1
	pcmpeqd	%xmm1, %xmm0		/* Any null double_word? */
	pcmpeqd	(%rsi), %xmm1		/* compare first 4 double_words for equality */
	psubb	%xmm0, %xmm1		/* packed sub of comparison results*/
	pmovmskb %xmm1, %edx
	sub	$0xffff, %edx		/* if first 4 double_words are same, edx == 0xffff */
	jnz	L(less4_double_words)

	movdqu	16(%rdi), %xmm1
	pcmpeqd	%xmm1, %xmm0		/* Any null double_word? */
	pcmpeqd	16(%rsi), %xmm1		/* compare first 4 double_words for equality */
	psubb	%xmm0, %xmm1		/* packed sub of comparison results*/
	pmovmskb %xmm1, %edx
	sub	$0xffff, %edx		/* if first 4 double_words are same, edx == 0xffff */
	jnz	L(less4_double_words_16)

	add	$32, %rsi
	add	$32, %rdi
	jmp	L(continue_48_00)

	.p2align 4
L(continue_0_00):
	movdqu	(%rdi), %xmm1
	pcmpeqd	%xmm1, %xmm0		/* Any null double_word? */
	pcmpeqd	(%rsi), %xmm1		/* compare first 4 double_words for equality */
	psubb	%xmm0, %xmm1		/* packed sub of comparison results*/
	pmovmskb %xmm1, %edx
	sub	$0xffff, %edx		/* if first 4 double_words are same, edx == 0xffff */
	jnz	L(less4_double_words)

	movdqu	16(%rdi), %xmm1
	pcmpeqd	%xmm1, %xmm0		/* Any null double_word? */
	pcmpeqd	16(%rsi), %xmm1		/* compare first 4 double_words for equality */
	psubb	%xmm0, %xmm1		/* packed sub of comparison results*/
	pmovmskb %xmm1, %edx
	sub	$0xffff, %edx		/* if first 4 double_words are same, edx == 0xffff */
	jnz	L(less4_double_words_16)

	movdqu	32(%rdi), %xmm1
	pcmpeqd	%xmm1, %xmm0		/* Any null double_word? */
	pcmpeqd	32(%rsi), %xmm1		/* compare first 4 double_words for equality */
	psubb	%xmm0, %xmm1		/* packed sub of comparison results*/
	pmovmskb %xmm1, %edx
	sub	$0xffff, %edx		/* if first 4 double_words are same, edx == 0xffff */
	jnz	L(less4_double_words_32)

	add	$48, %rsi
	add	$48, %rdi
	jmp	L(continue_48_00)

	.p2align 4
L(continue_32_32):
	movdqu	(%rdi), %xmm1
	movdqu	(%rsi), %xmm2
	pcmpeqd	%xmm1, %xmm0		/* Any null double_word? */
	pcmpeqd	%xmm2, %xmm1		/* compare first 4 double_words for equality */
	psubb	%xmm0, %xmm1		/* packed sub of comparison results*/
	pmovmskb %xmm1, %edx
	sub	$0xffff, %edx		/* if first 4 double_words are same, edx == 0xffff */
	jnz	L(less4_double_words)

	add	$16, %rsi
	add	$16, %rdi
	jmp	L(continue_48_48)

	.p2align 4
L(continue_16_16):
	movdqu	(%rdi), %xmm1
	movdqu	(%rsi), %xmm2
	pcmpeqd	%xmm1, %xmm0		/* Any null double_word? */
	pcmpeqd	%xmm2, %xmm1		/* compare first 4 double_words for equality */
	psubb	%xmm0, %xmm1		/* packed sub of comparison results*/
	pmovmskb %xmm1, %edx
	sub	$0xffff, %edx		/* if first 4 double_words are same, edx == 0xffff */
	jnz	L(less4_double_words)

	movdqu	16(%rdi), %xmm3
	movdqu	16(%rsi), %xmm4
	pcmpeqd	%xmm3, %xmm0		/* Any null double_word? */
	pcmpeqd	%xmm4, %xmm3		/* compare first 4 double_words for equality */
	psubb	%xmm0, %xmm3		/* packed sub of comparison results*/
	pmovmskb %xmm3, %edx
	sub	$0xffff, %edx		/* if first 4 double_words are same, edx == 0xffff */
	jnz	L(less4_double_words_16)

	add	$32, %rsi
	add	$32, %rdi
	jmp	L(continue_48_48)

	.p2align 4
L(continue_0_0):
	movdqu	(%rdi), %xmm1
	movdqu	(%rsi), %xmm2
	pcmpeqd	%xmm1, %xmm0		/* Any null double_word? */
	pcmpeqd	%xmm2, %xmm1		/* compare first 4 double_words for equality */
	psubb	%xmm0, %xmm1		/* packed sub of comparison results*/
	pmovmskb %xmm1, %edx
	sub	$0xffff, %edx		/* if first 4 double_words are same, edx == 0xffff */
	jnz	L(less4_double_words)

	movdqu	16(%rdi), %xmm3
	movdqu	16(%rsi), %xmm4
	pcmpeqd	%xmm3, %xmm0		/* Any null double_word? */
	pcmpeqd	%xmm4, %xmm3		/* compare first 4 double_words for equality */
	psubb	%xmm0, %xmm3		/* packed sub of comparison results*/
	pmovmskb %xmm3, %edx
	sub	$0xffff, %edx		/* if first 4 double_words are same, edx == 0xffff */
	jnz	L(less4_double_words_16)

	movdqu	32(%rdi), %xmm1
	movdqu	32(%rsi), %xmm2
	pcmpeqd	%xmm1, %xmm0		/* Any null double_word? */
	pcmpeqd	%xmm2, %xmm1		/* compare first 4 double_words for equality */
	psubb	%xmm0, %xmm1		/* packed sub of comparison results*/
	pmovmskb %xmm1, %edx
	sub	$0xffff, %edx		/* if first 4 double_words are same, edx == 0xffff */
	jnz	L(less4_double_words_32)

	add	$48, %rsi
	add	$48, %rdi
	jmp	L(continue_48_48)

	.p2align 4
L(continue_0_16):
	movdqu	(%rdi), %xmm1
	movdqu	(%rsi), %xmm2
	pcmpeqd	%xmm1, %xmm0		/* Any null double_word? */
	pcmpeqd	%xmm2, %xmm1		/* compare first 4 double_words for equality */
	psubb	%xmm0, %xmm1		/* packed sub of comparison results*/
	pmovmskb %xmm1, %edx
	sub	$0xffff, %edx		/* if first 4 double_words are same, edx == 0xffff */
	jnz	L(less4_double_words)

	movdqu	16(%rdi), %xmm1
	movdqu	16(%rsi), %xmm2
	pcmpeqd	%xmm1, %xmm0		/* Any null double_word? */
	pcmpeqd	%xmm2, %xmm1		/* compare first 4 double_words for equality */
	psubb	%xmm0, %xmm1		/* packed sub of comparison results*/
	pmovmskb %xmm1, %edx
	sub	$0xffff, %edx		/* if first 4 double_words are same, edx == 0xffff */
	jnz	L(less4_double_words_16)

	add	$32, %rsi
	add	$32, %rdi
	jmp	L(continue_32_48)

	.p2align 4
L(continue_0_32):
	movdqu	(%rdi), %xmm1
	movdqu	(%rsi), %xmm2
	pcmpeqd	%xmm1, %xmm0		/* Any null double_word? */
	pcmpeqd	%xmm2, %xmm1		/* compare first 4 double_words for equality */
	psubb	%xmm0, %xmm1		/* packed sub of comparison results*/
	pmovmskb %xmm1, %edx
	sub	$0xffff, %edx		/* if first 4 double_words are same, edx == 0xffff */
	jnz	L(less4_double_words)

	add	$16, %rsi
	add	$16, %rdi
	jmp	L(continue_16_48)

	.p2align 4
L(continue_16_32):
	movdqu	(%rdi), %xmm1
	movdqu	(%rsi), %xmm2
	pcmpeqd	%xmm1, %xmm0		/* Any null double_word? */
	pcmpeqd	%xmm2, %xmm1		/* compare first 4 double_words for equality */
	psubb	%xmm0, %xmm1		/* packed sub of comparison results*/
	pmovmskb %xmm1, %edx
	sub	$0xffff, %edx		/* if first 4 double_words are same, edx == 0xffff */
	jnz	L(less4_double_words)

	add	$16, %rsi
	add	$16, %rdi
	jmp	L(continue_32_48)

	.p2align 4
L(less4_double_words1):
	cmp	(%rsi), %eax
	jne	L(nequal)
	test	%eax, %eax
	jz	L(equal)

	mov	4(%rsi), %ecx
	cmp	%ecx, 4(%rdi)
	jne	L(nequal)
	test	%ecx, %ecx
	jz	L(equal)

	mov	8(%rsi), %ecx
	cmp	%ecx, 8(%rdi)
	jne	L(nequal)
	test	%ecx, %ecx
	jz	L(equal)

	mov	12(%rsi), %ecx
	cmp	%ecx, 12(%rdi)
	jne	L(nequal)
	xor	%eax, %eax
	ret

	.p2align 4
L(less4_double_words):
	xor	%eax, %eax
	test	%dl, %dl
	jz	L(next_two_double_words)
	and	$15, %dl
	jz	L(second_double_word)
	mov	(%rdi), %eax
	cmp	(%rsi), %eax
	jne	L(nequal)
	ret

	.p2align 4
L(second_double_word):
	mov	4(%rdi), %eax
	cmp	4(%rsi), %eax
	jne	L(nequal)
	ret

	.p2align 4
L(next_two_double_words):
	and	$15, %dh
	jz	L(fourth_double_word)
	mov	8(%rdi), %eax
	cmp	8(%rsi), %eax
	jne	L(nequal)
	ret

	.p2align 4
L(fourth_double_word):
	mov	12(%rdi), %eax
	cmp	12(%rsi), %eax
	jne	L(nequal)
	ret

	.p2align 4
L(less4_double_words_16):
	xor	%eax, %eax
	test	%dl, %dl
	jz	L(next_two_double_words_16)
	and	$15, %dl
	jz	L(second_double_word_16)
	mov	16(%rdi), %eax
	cmp	16(%rsi), %eax
	jne	L(nequal)
	ret

	.p2align 4
L(second_double_word_16):
	mov	20(%rdi), %eax
	cmp	20(%rsi), %eax
	jne	L(nequal)
	ret

	.p2align 4
L(next_two_double_words_16):
	and	$15, %dh
	jz	L(fourth_double_word_16)
	mov	24(%rdi), %eax
	cmp	24(%rsi), %eax
	jne	L(nequal)
	ret

	.p2align 4
L(fourth_double_word_16):
	mov	28(%rdi), %eax
	cmp	28(%rsi), %eax
	jne	L(nequal)
	ret

	.p2align 4
L(less4_double_words_32):
	xor	%eax, %eax
	test	%dl, %dl
	jz	L(next_two_double_words_32)
	and	$15, %dl
	jz	L(second_double_word_32)
	mov	32(%rdi), %eax
	cmp	32(%rsi), %eax
	jne	L(nequal)
	ret

	.p2align 4
L(second_double_word_32):
	mov	36(%rdi), %eax
	cmp	36(%rsi), %eax
	jne	L(nequal)
	ret

	.p2align 4
L(next_two_double_words_32):
	and	$15, %dh
	jz	L(fourth_double_word_32)
	mov	40(%rdi), %eax
	cmp	40(%rsi), %eax
	jne	L(nequal)
	ret

	.p2align 4
L(fourth_double_word_32):
	mov	44(%rdi), %eax
	cmp	44(%rsi), %eax
	jne	L(nequal)
	ret

	.p2align 4
L(less4_double_words_48):
	xor	%eax, %eax
	test	%dl, %dl
	jz	L(next_two_double_words_48)
	and	$15, %dl
	jz	L(second_double_word_48)
	mov	48(%rdi), %eax
	cmp	48(%rsi), %eax
	jne	L(nequal)
	ret

	.p2align 4
L(second_double_word_48):
	mov	52(%rdi), %eax
	cmp	52(%rsi), %eax
	jne	L(nequal)
	ret

	.p2align 4
L(next_two_double_words_48):
	and	$15, %dh
	jz	L(fourth_double_word_48)
	mov	56(%rdi), %eax
	cmp	56(%rsi), %eax
	jne	L(nequal)
	ret

	.p2align 4
L(fourth_double_word_48):
	mov	60(%rdi), %eax
	cmp	60(%rsi), %eax
	jne	L(nequal)
	ret

	.p2align 4
L(nequal):
	mov	$1, %eax
	jg	L(nequal_bigger)
	neg	%eax

L(nequal_bigger):
	ret

	.p2align 4
L(equal):
	xor	%rax, %rax
	ret

END (__wcscmp)
#ifndef __wcscmp
libc_hidden_def (__wcscmp)
weak_alias (__wcscmp, wcscmp)
#endif
