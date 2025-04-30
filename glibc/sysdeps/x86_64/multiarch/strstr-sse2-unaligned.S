/* strstr with unaligned loads
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

ENTRY(__strstr_sse2_unaligned)
	movzbl	(%rsi), %eax
	testb	%al, %al
	je	L(empty)
	movzbl	1(%rsi), %edx
	testb	%dl, %dl
	je	L(strchr)
	movd	%eax, %xmm1
	movd	%edx, %xmm2
	movq	%rdi, %rax
	andl	$4095, %eax
	punpcklbw	%xmm1, %xmm1
	cmpq	$4031, %rax
	punpcklbw	%xmm2, %xmm2
	punpcklwd	%xmm1, %xmm1
	punpcklwd	%xmm2, %xmm2
	pshufd	$0, %xmm1, %xmm1
	pshufd	$0, %xmm2, %xmm2
	ja	L(cross_page)
	movdqu	(%rdi), %xmm3
	pxor	%xmm5, %xmm5
	movdqu	1(%rdi), %xmm4
	movdqa	%xmm3, %xmm6
	pcmpeqb	%xmm1, %xmm3
	pcmpeqb	%xmm2, %xmm4
	movdqu	16(%rdi), %xmm0
	pcmpeqb	%xmm5, %xmm6
	pminub	%xmm4, %xmm3
	movdqa	%xmm3, %xmm4
	movdqu	17(%rdi), %xmm3
	pcmpeqb	%xmm0, %xmm5
	pcmpeqb	%xmm2, %xmm3
	por	%xmm6, %xmm4
	pcmpeqb	%xmm1, %xmm0
	pminub	%xmm3, %xmm0
	por	%xmm5, %xmm0
	pmovmskb	%xmm4, %r8d
	pmovmskb	%xmm0, %eax
	salq	$16, %rax
	orq	%rax, %r8
	je	L(next_32_bytes)
L(next_pair_index):
	bsf	%r8, %rax
	addq	%rdi, %rax
	cmpb	$0, (%rax)
	je	L(zero1)
	movzbl	2(%rsi), %edx
	testb	%dl, %dl
	je	L(found1)
	cmpb	2(%rax), %dl
	jne	L(next_pair)
	xorl	%edx, %edx
	jmp	L(pair_loop_start)

	.p2align 4
L(strchr):
	movzbl	%al, %esi
	jmp	__strchr_sse2

	.p2align 4
L(pair_loop):
	addq	$1, %rdx
	cmpb	2(%rax,%rdx), %cl
	jne	L(next_pair)
L(pair_loop_start):
	movzbl	3(%rsi,%rdx), %ecx
	testb	%cl, %cl
	jne	L(pair_loop)
L(found1):
	ret
L(zero1):
	xorl	%eax, %eax
	ret

	.p2align 4
L(next_pair):
	leaq	-1(%r8), %rax
	andq	%rax, %r8
	jne	L(next_pair_index)

	.p2align 4
L(next_32_bytes):
	movdqu	32(%rdi), %xmm3
	pxor	%xmm5, %xmm5
	movdqu	33(%rdi), %xmm4
	movdqa	%xmm3, %xmm6
	pcmpeqb	%xmm1, %xmm3
	pcmpeqb	%xmm2, %xmm4
	movdqu	48(%rdi), %xmm0
	pcmpeqb	%xmm5, %xmm6
	pminub	%xmm4, %xmm3
	movdqa	%xmm3, %xmm4
	movdqu	49(%rdi), %xmm3
	pcmpeqb	%xmm0, %xmm5
	pcmpeqb	%xmm2, %xmm3
	por	%xmm6, %xmm4
	pcmpeqb	%xmm1, %xmm0
	pminub	%xmm3, %xmm0
	por	%xmm5, %xmm0
	pmovmskb	%xmm4, %eax
	salq	$32, %rax
	pmovmskb	%xmm0, %r8d
	salq	$48, %r8
	orq	%rax, %r8
	je	L(loop_header)
L(next_pair2_index):
	bsfq	%r8, %rax
	addq	%rdi, %rax
	cmpb	$0, (%rax)
	je	L(zero2)
	movzbl	2(%rsi), %edx
	testb	%dl, %dl
	je	L(found2)
	cmpb	2(%rax), %dl
	jne	L(next_pair2)
	xorl	%edx, %edx
	jmp	L(pair_loop2_start)

	.p2align 4
L(pair_loop2):
	addq	$1, %rdx
	cmpb	2(%rax,%rdx), %cl
	jne	L(next_pair2)
L(pair_loop2_start):
	movzbl	3(%rsi,%rdx), %ecx
	testb	%cl, %cl
	jne	L(pair_loop2)
L(found2):
	ret
	L(zero2):
	xorl	%eax, %eax
	ret
L(empty):
	mov %rdi, %rax
	ret

	.p2align 4
L(next_pair2):
	leaq	-1(%r8), %rax
	andq	%rax, %r8
	jne	L(next_pair2_index)
L(loop_header):
	movq	$-512, %r11
	movq	%rdi, %r9

	pxor	%xmm7, %xmm7
	andq	$-64, %rdi

	.p2align 4
L(loop):
	movdqa	64(%rdi), %xmm3
	movdqu	63(%rdi), %xmm6
	movdqa	%xmm3, %xmm0
	pxor	%xmm2, %xmm3
	pxor	%xmm1, %xmm6
	movdqa	80(%rdi), %xmm10
	por	%xmm3, %xmm6
	pminub	%xmm10, %xmm0
	movdqu	79(%rdi), %xmm3
	pxor	%xmm2, %xmm10
	pxor	%xmm1, %xmm3
	movdqa	96(%rdi), %xmm9
	por	%xmm10, %xmm3
	pminub	%xmm9, %xmm0
	pxor	%xmm2, %xmm9
	movdqa	112(%rdi), %xmm8
	addq	$64, %rdi
	pminub	%xmm6, %xmm3
	movdqu	31(%rdi), %xmm4
	pminub	%xmm8, %xmm0
	pxor	%xmm2, %xmm8
	pxor	%xmm1, %xmm4
	por	%xmm9, %xmm4
	pminub	%xmm4, %xmm3
	movdqu	47(%rdi), %xmm5
	pxor	%xmm1, %xmm5
	por	%xmm8, %xmm5
	pminub	%xmm5, %xmm3
	pminub	%xmm3, %xmm0
	pcmpeqb	%xmm7, %xmm0
	pmovmskb	%xmm0, %eax
	testl	%eax, %eax
	je	L(loop)
	pminub (%rdi), %xmm6
	pminub 32(%rdi),%xmm4
	pminub 48(%rdi),%xmm5
	pcmpeqb %xmm7, %xmm6
	pcmpeqb %xmm7, %xmm5
	pmovmskb	%xmm6, %edx
	movdqa	16(%rdi), %xmm8
	pcmpeqb %xmm7, %xmm4
	movdqu  15(%rdi), %xmm0
	pmovmskb	%xmm5, %r8d
	movdqa  %xmm8, %xmm3
	pmovmskb	%xmm4, %ecx
	pcmpeqb %xmm1,%xmm0
	pcmpeqb %xmm2,%xmm3
	salq	$32, %rcx
	pcmpeqb %xmm7,%xmm8
	salq	$48, %r8
	pminub  %xmm0,%xmm3
	orq	%rcx, %rdx
	por	%xmm3,%xmm8
	orq	%rdx, %r8
	pmovmskb	%xmm8, %eax
	salq	$16, %rax
	orq	%rax, %r8
	je	L(loop)
L(next_pair_index3):
	bsfq	%r8, %rcx
	addq	%rdi, %rcx
	cmpb	$0, (%rcx)
	je	L(zero)
	xorl	%eax, %eax
	movzbl	2(%rsi), %edx
	testb	%dl, %dl
	je	L(success3)
	cmpb	1(%rcx), %dl
	jne	L(next_pair3)
	jmp	L(pair_loop_start3)

	.p2align 4
L(pair_loop3):
	addq	$1, %rax
	cmpb	1(%rcx,%rax), %dl
	jne	L(next_pair3)
L(pair_loop_start3):
	movzbl	3(%rsi,%rax), %edx
	testb	%dl, %dl
	jne	L(pair_loop3)
L(success3):
	lea	-1(%rcx), %rax
	ret

	.p2align 4
L(next_pair3):
	addq	%rax, %r11
	movq	%rdi,  %rax
	subq	%r9, %rax
	cmpq	%r11, %rax
	jl	L(switch_strstr)
	leaq	-1(%r8), %rax
	andq	%rax, %r8
	jne	L(next_pair_index3)
	jmp	L(loop)

	.p2align 4
L(switch_strstr):
	movq	%rdi, %rdi
	jmp	__strstr_sse2

	.p2align 4
L(cross_page):

	movq	%rdi, %rax
	pxor	%xmm0, %xmm0
	andq	$-64, %rax
	movdqa	(%rax), %xmm3
	movdqu	-1(%rax), %xmm4
	movdqa	%xmm3, %xmm8
	movdqa	16(%rax), %xmm5
	pcmpeqb	%xmm1, %xmm4
	pcmpeqb	%xmm0, %xmm8
	pcmpeqb	%xmm2, %xmm3
	movdqa	%xmm5, %xmm7
	pminub	%xmm4, %xmm3
	movdqu	15(%rax), %xmm4
	pcmpeqb	%xmm0, %xmm7
	por	%xmm3, %xmm8
	movdqa	%xmm5, %xmm3
	movdqa	32(%rax), %xmm5
	pcmpeqb	%xmm1, %xmm4
	pcmpeqb	%xmm2, %xmm3
	movdqa	%xmm5, %xmm6
	pmovmskb	%xmm8, %ecx
	pminub	%xmm4, %xmm3
	movdqu	31(%rax), %xmm4
	por	%xmm3, %xmm7
	movdqa	%xmm5, %xmm3
	pcmpeqb	%xmm0, %xmm6
	movdqa	48(%rax), %xmm5
	pcmpeqb	%xmm1, %xmm4
	pmovmskb	%xmm7, %r8d
	pcmpeqb	%xmm2, %xmm3
	pcmpeqb	%xmm5, %xmm0
	pminub	%xmm4, %xmm3
	movdqu	47(%rax), %xmm4
	por	%xmm3, %xmm6
	movdqa	%xmm5, %xmm3
	salq	$16, %r8
	pcmpeqb	%xmm1, %xmm4
	pcmpeqb	%xmm2, %xmm3
	pmovmskb	%xmm6, %r10d
	pminub	%xmm4, %xmm3
	por	%xmm3, %xmm0
	salq	$32, %r10
	orq	%r10, %r8
	orq	%rcx, %r8
	movl	%edi, %ecx
	pmovmskb	%xmm0, %edx
	subl	%eax, %ecx
	salq	$48, %rdx
	orq	%rdx, %r8
	shrq	%cl, %r8
	je	L(loop_header)
L(next_pair_index4):
	bsfq	%r8, %rax
	addq	%rdi, %rax
	cmpb	$0, (%rax)
	je	L(zero)

	cmpq	%rax,%rdi
	je	L(next_pair4)

	movzbl	2(%rsi), %edx
	testb	%dl, %dl
	je	L(found3)
	cmpb	1(%rax), %dl
	jne	L(next_pair4)
	xorl	%edx, %edx
	jmp	L(pair_loop_start4)

	.p2align 4
L(pair_loop4):
	addq	$1, %rdx
	cmpb	1(%rax,%rdx), %cl
	jne	L(next_pair4)
L(pair_loop_start4):
	movzbl	3(%rsi,%rdx), %ecx
	testb	%cl, %cl
	jne	L(pair_loop4)
L(found3):
	subq $1, %rax
	ret

	.p2align 4
L(next_pair4):
	leaq	-1(%r8), %rax
	andq	%rax, %r8
	jne	L(next_pair_index4)
	jmp	L(loop_header)

	.p2align 4
L(found):
	rep
	ret

	.p2align 4
L(zero):
	xorl	%eax, %eax
	ret


END(__strstr_sse2_unaligned)
