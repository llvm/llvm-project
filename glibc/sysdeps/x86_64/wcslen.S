/* Optimized wcslen for x86-64 with SSE2.
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

	.text
ENTRY (__wcslen)
	cmpl	$0, (%rdi)
	jz	L(exit_tail0)
	cmpl	$0, 4(%rdi)
	jz	L(exit_tail1)
	cmpl	$0, 8(%rdi)
	jz	L(exit_tail2)
	cmpl	$0, 12(%rdi)
	jz	L(exit_tail3)
	cmpl	$0, 16(%rdi)
	jz	L(exit_tail4)
	cmpl	$0, 20(%rdi)
	jz	L(exit_tail5)
	cmpl	$0, 24(%rdi)
	jz	L(exit_tail6)
	cmpl	$0, 28(%rdi)
	jz	L(exit_tail7)

	pxor	%xmm0, %xmm0

	lea	32(%rdi), %rax
	lea	16(%rdi), %rcx
	and	$-16, %rax

	pcmpeqd	(%rax), %xmm0
	pmovmskb %xmm0, %edx
	pxor	%xmm1, %xmm1
	test	%edx, %edx
	lea	16(%rax), %rax
	jnz	L(exit)

	pcmpeqd	(%rax), %xmm1
	pmovmskb %xmm1, %edx
	pxor	%xmm2, %xmm2
	test	%edx, %edx
	lea	16(%rax), %rax
	jnz	L(exit)

	pcmpeqd	(%rax), %xmm2
	pmovmskb %xmm2, %edx
	pxor	%xmm3, %xmm3
	test	%edx, %edx
	lea	16(%rax), %rax
	jnz	L(exit)

	pcmpeqd	(%rax), %xmm3
	pmovmskb %xmm3, %edx
	test	%edx, %edx
	lea	16(%rax), %rax
	jnz	L(exit)

	pcmpeqd	(%rax), %xmm0
	pmovmskb %xmm0, %edx
	test	%edx, %edx
	lea	16(%rax), %rax
	jnz	L(exit)

	pcmpeqd	(%rax), %xmm1
	pmovmskb %xmm1, %edx
	test	%edx, %edx
	lea	16(%rax), %rax
	jnz	L(exit)

	pcmpeqd	(%rax), %xmm2
	pmovmskb %xmm2, %edx
	test	%edx, %edx
	lea	16(%rax), %rax
	jnz	L(exit)

	pcmpeqd	(%rax), %xmm3
	pmovmskb %xmm3, %edx
	test	%edx, %edx
	lea	16(%rax), %rax
	jnz	L(exit)

	pcmpeqd	(%rax), %xmm0
	pmovmskb %xmm0, %edx
	test	%edx, %edx
	lea	16(%rax), %rax
	jnz	L(exit)

	pcmpeqd	(%rax), %xmm1
	pmovmskb %xmm1, %edx
	test	%edx, %edx
	lea	16(%rax), %rax
	jnz	L(exit)

	pcmpeqd	(%rax), %xmm2
	pmovmskb %xmm2, %edx
	test	%edx, %edx
	lea	16(%rax), %rax
	jnz	L(exit)

	pcmpeqd	(%rax), %xmm3
	pmovmskb %xmm3, %edx
	test	%edx, %edx
	lea	16(%rax), %rax
	jnz	L(exit)

	and	$-0x40, %rax

	.p2align 4
L(aligned_64_loop):
	movaps	(%rax), %xmm0
	movaps	16(%rax), %xmm1
	movaps	32(%rax), %xmm2
	movaps	48(%rax), %xmm6

	pminub	%xmm1, %xmm0
	pminub	%xmm6, %xmm2
	pminub	%xmm0, %xmm2
	pcmpeqd	%xmm3, %xmm2
	pmovmskb %xmm2, %edx
	test	%edx, %edx
	lea	64(%rax), %rax
	jz	L(aligned_64_loop)

	pcmpeqd	-64(%rax), %xmm3
	pmovmskb %xmm3, %edx
	test	%edx, %edx
	lea	48(%rcx), %rcx
	jnz	L(exit)

	pcmpeqd	%xmm1, %xmm3
	pmovmskb %xmm3, %edx
	test	%edx, %edx
	lea	-16(%rcx), %rcx
	jnz	L(exit)

	pcmpeqd	-32(%rax), %xmm3
	pmovmskb %xmm3, %edx
	test	%edx, %edx
	lea	-16(%rcx), %rcx
	jnz	L(exit)

	pcmpeqd	%xmm6, %xmm3
	pmovmskb %xmm3, %edx
	test	%edx, %edx
	lea	-16(%rcx), %rcx
	jnz	L(exit)

	jmp	L(aligned_64_loop)

	.p2align 4
L(exit):
	sub	%rcx, %rax
	shr	$2, %rax
	test	%dl, %dl
	jz	L(exit_high)

	mov	%dl, %cl
	and	$15, %cl
	jz	L(exit_1)
	ret

	.p2align 4
L(exit_high):
	mov	%dh, %ch
	and	$15, %ch
	jz	L(exit_3)
	add	$2, %rax
	ret

	.p2align 4
L(exit_1):
	add	$1, %rax
	ret

	.p2align 4
L(exit_3):
	add	$3, %rax
	ret

	.p2align 4
L(exit_tail0):
	xor	%rax, %rax
	ret

	.p2align 4
L(exit_tail1):
	mov	$1, %rax
	ret

	.p2align 4
L(exit_tail2):
	mov	$2, %rax
	ret

	.p2align 4
L(exit_tail3):
	mov	$3, %rax
	ret

	.p2align 4
L(exit_tail4):
	mov	$4, %rax
	ret

	.p2align 4
L(exit_tail5):
	mov	$5, %rax
	ret

	.p2align 4
L(exit_tail6):
	mov	$6, %rax
	ret

	.p2align 4
L(exit_tail7):
	mov	$7, %rax
	ret

END (__wcslen)

weak_alias(__wcslen, wcslen)
