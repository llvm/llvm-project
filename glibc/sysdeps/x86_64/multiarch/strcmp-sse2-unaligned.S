/* strcmp with unaligned loads
   Copyright (C) 2013-2021 Free Software Foundation, Inc.
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

#include "sysdep.h"

ENTRY ( __strcmp_sse2_unaligned)
	movl	%edi, %eax
	xorl	%edx, %edx
	pxor	%xmm7, %xmm7
	orl	%esi, %eax
	andl	$4095, %eax
	cmpl	$4032, %eax
	jg	L(cross_page)
	movdqu	(%rdi), %xmm1
	movdqu	(%rsi), %xmm0
	pcmpeqb	%xmm1, %xmm0
	pminub	%xmm1, %xmm0
	pxor	%xmm1, %xmm1
	pcmpeqb	%xmm1, %xmm0
	pmovmskb	%xmm0, %eax
	testq	%rax, %rax
	je	L(next_48_bytes)
L(return):
	bsfq	%rax, %rdx
	movzbl	(%rdi, %rdx), %eax
	movzbl	(%rsi, %rdx), %edx
	subl	%edx, %eax
	ret

	.p2align 4
L(next_48_bytes):
	movdqu	16(%rdi), %xmm6
	movdqu	16(%rsi), %xmm3
	movdqu	32(%rdi), %xmm5
	pcmpeqb	%xmm6, %xmm3
	movdqu	32(%rsi), %xmm2
	pminub	%xmm6, %xmm3
	pcmpeqb	%xmm1, %xmm3
	movdqu	48(%rdi), %xmm4
	pcmpeqb	%xmm5, %xmm2
	pmovmskb	%xmm3, %edx
	movdqu	48(%rsi), %xmm0
	pminub	%xmm5, %xmm2
	pcmpeqb	%xmm1, %xmm2
	pcmpeqb	%xmm4, %xmm0
	pmovmskb	%xmm2, %eax
	salq	$16, %rdx
	pminub	%xmm4, %xmm0
	pcmpeqb	%xmm1, %xmm0
	salq	$32, %rax
	orq	%rdx, %rax
	pmovmskb	%xmm0, %ecx
	movq	%rcx, %rdx
	salq	$48, %rdx
	orq	%rdx, %rax
	jne	L(return)
L(main_loop_header):
	leaq	64(%rdi), %rdx
	movl	$4096, %ecx
	pxor	%xmm9, %xmm9
	andq	$-64, %rdx
	subq	%rdi, %rdx
	leaq	(%rdi, %rdx), %rax
	addq	%rsi, %rdx
	movq	%rdx, %rsi
	andl	$4095, %esi
	subq	%rsi, %rcx
	shrq	$6, %rcx
	movq	%rcx, %rsi
	jmp	L(loop_start)

	.p2align 4
L(loop):
	addq	$64, %rax
	addq	$64, %rdx
L(loop_start):
	testq	%rsi, %rsi
	leaq	-1(%rsi), %rsi
	je	L(loop_cross_page)
L(back_to_loop):
	movdqu	(%rdx), %xmm0
	movdqu	16(%rdx), %xmm1
	movdqa	(%rax), %xmm2
	movdqa	16(%rax), %xmm3
	pcmpeqb	%xmm2, %xmm0
	movdqu	32(%rdx), %xmm5
	pcmpeqb	%xmm3, %xmm1
	pminub	%xmm2, %xmm0
	movdqu	48(%rdx), %xmm6
	pminub	%xmm3, %xmm1
	movdqa	32(%rax), %xmm2
	pminub	%xmm1, %xmm0
	movdqa	48(%rax), %xmm3
	pcmpeqb	%xmm2, %xmm5
	pcmpeqb	%xmm3, %xmm6
	pminub	%xmm2, %xmm5
	pminub	%xmm3, %xmm6
	pminub	%xmm5, %xmm0
	pminub	%xmm6, %xmm0
	pcmpeqb	%xmm7, %xmm0
	pmovmskb	%xmm0, %ecx
	testl	%ecx, %ecx
	je	L(loop)
	pcmpeqb	%xmm7, %xmm5
	movdqu	(%rdx), %xmm0
	pcmpeqb	%xmm7, %xmm1
	movdqa	(%rax), %xmm2
	pcmpeqb	%xmm2, %xmm0
	pminub	%xmm2, %xmm0
	pcmpeqb	%xmm7, %xmm6
	pcmpeqb	%xmm7, %xmm0
	pmovmskb	%xmm1, %ecx
	pmovmskb	%xmm5, %r8d
	pmovmskb	%xmm0, %edi
	salq	$16, %rcx
	salq	$32, %r8
	pmovmskb	%xmm6, %esi
	orq	%r8, %rcx
	orq	%rdi, %rcx
	salq	$48, %rsi
	orq	%rsi, %rcx
	bsfq	%rcx, %rcx
	movzbl	(%rax, %rcx), %eax
	movzbl	(%rdx, %rcx), %edx
	subl	%edx, %eax
	ret

	.p2align 4
L(loop_cross_page):
	xor	%r10, %r10
	movq	%rdx, %r9
	and	$63, %r9
	subq	%r9, %r10

	movdqa	(%rdx, %r10), %xmm0
	movdqa	16(%rdx, %r10), %xmm1
	movdqu	(%rax, %r10), %xmm2
	movdqu	16(%rax, %r10), %xmm3
	pcmpeqb	%xmm2, %xmm0
	movdqa	32(%rdx, %r10), %xmm5
	pcmpeqb	%xmm3, %xmm1
	pminub	%xmm2, %xmm0
	movdqa	48(%rdx, %r10), %xmm6
	pminub	%xmm3, %xmm1
	movdqu	32(%rax, %r10), %xmm2
	movdqu	48(%rax, %r10), %xmm3
	pcmpeqb	%xmm2, %xmm5
	pcmpeqb	%xmm3, %xmm6
	pminub	%xmm2, %xmm5
	pminub	%xmm3, %xmm6

	pcmpeqb	%xmm7, %xmm0
	pcmpeqb	%xmm7, %xmm1
	pcmpeqb	%xmm7, %xmm5
	pcmpeqb	%xmm7, %xmm6

	pmovmskb	%xmm1, %ecx
	pmovmskb	%xmm5, %r8d
	pmovmskb	%xmm0, %edi
	salq	$16, %rcx
	salq	$32, %r8
	pmovmskb	%xmm6, %esi
	orq	%r8, %rdi
	orq	%rcx, %rdi
	salq	$48, %rsi
	orq	%rsi, %rdi
	movq	%r9, %rcx
	movq	$63, %rsi
	shrq	%cl, %rdi
	test	%rdi, %rdi
	je	L(back_to_loop)
	bsfq	%rdi, %rcx
	movzbl	(%rax, %rcx), %eax
	movzbl	(%rdx, %rcx), %edx
	subl	%edx, %eax
	ret

	.p2align 4
L(cross_page_loop):
	cmpb	%cl, %al
	jne	L(different)
	addq	$1, %rdx
	cmpq	$64, %rdx
	je	L(main_loop_header)
L(cross_page):
	movzbl	(%rdi, %rdx), %eax
	movzbl	(%rsi, %rdx), %ecx
	testb	%al, %al
	jne	L(cross_page_loop)
	xorl	%eax, %eax
L(different):
	subl	%ecx, %eax
	ret
END (__strcmp_sse2_unaligned)

#endif
