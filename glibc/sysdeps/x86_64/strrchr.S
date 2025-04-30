/* strrchr (str, ch) -- Return pointer to last occurrence of CH in STR.
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


#include <sysdep.h>

	.text
ENTRY (strrchr)
	movd	%esi, %xmm1
	movq	%rdi, %rax
	andl	$4095, %eax
	punpcklbw	%xmm1, %xmm1
	cmpq	$4032, %rax
	punpcklwd	%xmm1, %xmm1
	pshufd	$0, %xmm1, %xmm1
	ja	L(cross_page)
	movdqu	(%rdi), %xmm0
	pxor	%xmm2, %xmm2
	movdqa	%xmm0, %xmm3
	pcmpeqb	%xmm1, %xmm0
	pcmpeqb	%xmm2, %xmm3
	pmovmskb	%xmm0, %ecx
	pmovmskb	%xmm3, %edx
	testq	%rdx, %rdx
	je	L(next_48_bytes)
	leaq	-1(%rdx), %rax
	xorq	%rdx, %rax
	andq	%rcx, %rax
	je	L(exit)
	bsrq	%rax, %rax
	addq	%rdi, %rax
	ret

	.p2align 4
L(next_48_bytes):
	movdqu	16(%rdi), %xmm4
	movdqa	%xmm4, %xmm5
	movdqu	32(%rdi), %xmm3
	pcmpeqb	%xmm1, %xmm4
	pcmpeqb	%xmm2, %xmm5
	movdqu	48(%rdi), %xmm0
	pmovmskb	%xmm5, %edx
	movdqa	%xmm3, %xmm5
	pcmpeqb	%xmm1, %xmm3
	pcmpeqb	%xmm2, %xmm5
	pcmpeqb	%xmm0, %xmm2
	salq	$16, %rdx
	pmovmskb	%xmm3, %r8d
	pmovmskb	%xmm5, %eax
	pmovmskb	%xmm2, %esi
	salq	$32, %r8
	salq	$32, %rax
	pcmpeqb	%xmm1, %xmm0
	orq	%rdx, %rax
	movq	%rsi, %rdx
	pmovmskb	%xmm4, %esi
	salq	$48, %rdx
	salq	$16, %rsi
	orq	%r8, %rsi
	orq	%rcx, %rsi
	pmovmskb	%xmm0, %ecx
	salq	$48, %rcx
	orq	%rcx, %rsi
	orq	%rdx, %rax
	je	L(loop_header2)
	leaq	-1(%rax), %rcx
	xorq	%rax, %rcx
	andq	%rcx, %rsi
	je	L(exit)
	bsrq	%rsi, %rsi
	leaq	(%rdi,%rsi), %rax
	ret

	.p2align 4
L(loop_header2):
	testq	%rsi, %rsi
	movq	%rdi, %rcx
	je	L(no_c_found)
L(loop_header):
	addq	$64, %rdi
	pxor	%xmm7, %xmm7
	andq	$-64, %rdi
	jmp	L(loop_entry)

	.p2align 4
L(loop64):
	testq	%rdx, %rdx
	cmovne	%rdx, %rsi
	cmovne	%rdi, %rcx
	addq	$64, %rdi
L(loop_entry):
	movdqa	32(%rdi), %xmm3
	pxor	%xmm6, %xmm6
	movdqa	48(%rdi), %xmm2
	movdqa	%xmm3, %xmm0
	movdqa	16(%rdi), %xmm4
	pminub	%xmm2, %xmm0
	movdqa	(%rdi), %xmm5
	pminub	%xmm4, %xmm0
	pminub	%xmm5, %xmm0
	pcmpeqb	%xmm7, %xmm0
	pmovmskb	%xmm0, %eax
	movdqa	%xmm5, %xmm0
	pcmpeqb	%xmm1, %xmm0
	pmovmskb	%xmm0, %r9d
	movdqa	%xmm4, %xmm0
	pcmpeqb	%xmm1, %xmm0
	pmovmskb	%xmm0, %edx
	movdqa	%xmm3, %xmm0
	pcmpeqb	%xmm1, %xmm0
	salq	$16, %rdx
	pmovmskb	%xmm0, %r10d
	movdqa	%xmm2, %xmm0
	pcmpeqb	%xmm1, %xmm0
	salq	$32, %r10
	orq	%r10, %rdx
	pmovmskb	%xmm0, %r8d
	orq	%r9, %rdx
	salq	$48, %r8
	orq	%r8, %rdx
	testl	%eax, %eax
	je	L(loop64)
	pcmpeqb	%xmm6, %xmm4
	pcmpeqb	%xmm6, %xmm3
	pcmpeqb	%xmm6, %xmm5
	pmovmskb	%xmm4, %eax
	pmovmskb	%xmm3, %r10d
	pcmpeqb	%xmm6, %xmm2
	pmovmskb	%xmm5, %r9d
	salq	$32, %r10
	salq	$16, %rax
	pmovmskb	%xmm2, %r8d
	orq	%r10, %rax
	orq	%r9, %rax
	salq	$48, %r8
	orq	%r8, %rax
	leaq	-1(%rax), %r8
	xorq	%rax, %r8
	andq	%r8, %rdx
	cmovne	%rdi, %rcx
	cmovne	%rdx, %rsi
	bsrq	%rsi, %rsi
	leaq	(%rcx,%rsi), %rax
	ret

	.p2align 4
L(no_c_found):
	movl	$1, %esi
	xorl	%ecx, %ecx
	jmp	L(loop_header)

	.p2align 4
L(exit):
	xorl	%eax, %eax
	ret

	.p2align 4
L(cross_page):
	movq	%rdi, %rax
	pxor	%xmm0, %xmm0
	andq	$-64, %rax
	movdqu	(%rax), %xmm5
	movdqa	%xmm5, %xmm6
	movdqu	16(%rax), %xmm4
	pcmpeqb	%xmm1, %xmm5
	pcmpeqb	%xmm0, %xmm6
	movdqu	32(%rax), %xmm3
	pmovmskb	%xmm6, %esi
	movdqa	%xmm4, %xmm6
	movdqu	48(%rax), %xmm2
	pcmpeqb	%xmm1, %xmm4
	pcmpeqb	%xmm0, %xmm6
	pmovmskb	%xmm6, %edx
	movdqa	%xmm3, %xmm6
	pcmpeqb	%xmm1, %xmm3
	pcmpeqb	%xmm0, %xmm6
	pcmpeqb	%xmm2, %xmm0
	salq	$16, %rdx
	pmovmskb	%xmm3, %r9d
	pmovmskb	%xmm6, %r8d
	pmovmskb	%xmm0, %ecx
	salq	$32, %r9
	salq	$32, %r8
	pcmpeqb	%xmm1, %xmm2
	orq	%r8, %rdx
	salq	$48, %rcx
	pmovmskb	%xmm5, %r8d
	orq	%rsi, %rdx
	pmovmskb	%xmm4, %esi
	orq	%rcx, %rdx
	pmovmskb	%xmm2, %ecx
	salq	$16, %rsi
	salq	$48, %rcx
	orq	%r9, %rsi
	orq	%r8, %rsi
	orq	%rcx, %rsi
	movl	%edi, %ecx
	subl	%eax, %ecx
	shrq	%cl, %rdx
	shrq	%cl, %rsi
	testq	%rdx, %rdx
	je	L(loop_header2)
	leaq	-1(%rdx), %rax
	xorq	%rdx, %rax
	andq	%rax, %rsi
	je	L(exit)
	bsrq	%rsi, %rax
	addq	%rdi, %rax
	ret
END (strrchr)

weak_alias (strrchr, rindex)
libc_hidden_builtin_def (strrchr)
