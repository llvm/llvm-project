/* strchr (str, ch) -- Return pointer to first occurrence of CH in STR.
   For AMD x86-64.
   Copyright (C) 2009-2021 Free Software Foundation, Inc.
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
ENTRY (strchr)
	movd	%esi, %xmm1
	movl	%edi, %eax
	andl	$4095, %eax
	punpcklbw %xmm1, %xmm1
	cmpl	$4032, %eax
	punpcklwd %xmm1, %xmm1
	pshufd	$0, %xmm1, %xmm1
	jg	L(cross_page)
	movdqu	(%rdi), %xmm0
	pxor	%xmm3, %xmm3
	movdqa	%xmm0, %xmm4
	pcmpeqb	%xmm1, %xmm0
	pcmpeqb	%xmm3, %xmm4
	por	%xmm4, %xmm0
	pmovmskb %xmm0, %eax
	test	%eax, %eax
	je	L(next_48_bytes)
	bsf	%eax, %eax
#ifdef AS_STRCHRNUL
	leaq	(%rdi,%rax), %rax
#else
	movl	$0, %edx
	leaq	(%rdi,%rax), %rax
	cmpb	%sil, (%rax)
	cmovne	%rdx, %rax
#endif
	ret

	.p2align 3
	L(next_48_bytes):
	movdqu	16(%rdi), %xmm0
	movdqa	%xmm0, %xmm4
	pcmpeqb	%xmm1, %xmm0
	pcmpeqb	%xmm3, %xmm4
	por	%xmm4, %xmm0
	pmovmskb %xmm0, %ecx
	movdqu	32(%rdi), %xmm0
	movdqa	%xmm0, %xmm4
	pcmpeqb	%xmm1, %xmm0
	salq	$16, %rcx
	pcmpeqb	%xmm3, %xmm4
	por	%xmm4, %xmm0
	pmovmskb %xmm0, %eax
	movdqu	48(%rdi), %xmm0
	pcmpeqb	%xmm0, %xmm3
	salq	$32, %rax
	pcmpeqb	%xmm1, %xmm0
	orq	%rcx, %rax
	por	%xmm3, %xmm0
	pmovmskb %xmm0, %ecx
	salq	$48, %rcx
	orq	%rcx, %rax
	testq	%rax, %rax
	jne	L(return)
L(loop_start):
	/* We use this alignment to force loop be aligned to 8 but not
	   16 bytes.  This gives better sheduling on AMD processors.  */
	.p2align 4
	pxor	%xmm6, %xmm6
	andq	$-64, %rdi
	.p2align 3
L(loop64):
	addq	$64, %rdi
	movdqa	(%rdi), %xmm5
	movdqa	16(%rdi), %xmm2
	movdqa	32(%rdi), %xmm3
	pxor	%xmm1, %xmm5
	movdqa	48(%rdi), %xmm4
	pxor	%xmm1, %xmm2
	pxor	%xmm1, %xmm3
	pminub	(%rdi), %xmm5
	pxor	%xmm1, %xmm4
	pminub	16(%rdi), %xmm2
	pminub	32(%rdi), %xmm3
	pminub	%xmm2, %xmm5
	pminub	48(%rdi), %xmm4
	pminub	%xmm3, %xmm5
	pminub	%xmm4, %xmm5
	pcmpeqb %xmm6, %xmm5
	pmovmskb %xmm5, %eax

	testl	%eax, %eax
	je	L(loop64)

	movdqa	(%rdi), %xmm5
	movdqa	%xmm5, %xmm0
	pcmpeqb	%xmm1, %xmm5
	pcmpeqb	%xmm6, %xmm0
	por	%xmm0, %xmm5
	pcmpeqb %xmm6, %xmm2
	pcmpeqb %xmm6, %xmm3
	pcmpeqb %xmm6, %xmm4

	pmovmskb %xmm5, %ecx
	pmovmskb %xmm2, %eax
	salq	$16, %rax
	pmovmskb %xmm3, %r8d
	pmovmskb %xmm4, %edx
	salq	$32, %r8
	orq	%r8, %rax
	orq	%rcx, %rax
	salq	$48, %rdx
	orq	%rdx, %rax
	.p2align 3
L(return):
	bsfq	%rax, %rax
#ifdef AS_STRCHRNUL
	leaq	(%rdi,%rax), %rax
#else
	movl	$0, %edx
	leaq	(%rdi,%rax), %rax
	cmpb	%sil, (%rax)
	cmovne	%rdx, %rax
#endif
	ret
	.p2align 4

L(cross_page):
	movq	%rdi, %rdx
	pxor	%xmm2, %xmm2
	andq	$-64, %rdx
	movdqa	%xmm1, %xmm0
	movdqa	(%rdx), %xmm3
	movdqa	%xmm3, %xmm4
	pcmpeqb	%xmm1, %xmm3
	pcmpeqb	%xmm2, %xmm4
	por	%xmm4, %xmm3
	pmovmskb %xmm3, %r8d
	movdqa	16(%rdx), %xmm3
	movdqa	%xmm3, %xmm4
	pcmpeqb	%xmm1, %xmm3
	pcmpeqb	%xmm2, %xmm4
	por	%xmm4, %xmm3
	pmovmskb %xmm3, %eax
	movdqa	32(%rdx), %xmm3
	movdqa	%xmm3, %xmm4
	pcmpeqb	%xmm1, %xmm3
	salq	$16, %rax
	pcmpeqb	%xmm2, %xmm4
	por	%xmm4, %xmm3
	pmovmskb %xmm3, %r9d
	movdqa	48(%rdx), %xmm3
	pcmpeqb	%xmm3, %xmm2
	salq	$32, %r9
	pcmpeqb	%xmm3, %xmm0
	orq	%r9, %rax
	orq	%r8, %rax
	por	%xmm2, %xmm0
	pmovmskb %xmm0, %ecx
	salq	$48, %rcx
	orq	%rcx, %rax
	movl	%edi, %ecx
	subb	%dl, %cl
	shrq	%cl, %rax
	testq	%rax, %rax
	jne	L(return)
	jmp	L(loop_start)

END (strchr)

#ifndef AS_STRCHRNUL
weak_alias (strchr, index)
libc_hidden_builtin_def (strchr)
#endif
