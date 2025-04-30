/* memcmp with SSE2
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

	.text
ENTRY (memcmp)
#ifdef __ILP32__
	/* Clear the upper 32 bits.  */
	movl	%edx, %edx
#endif
	test	%RDX_LP, %RDX_LP
	jz	L(finz)
	cmpq	$1, %rdx
	jbe	L(finr1b)
	subq	%rdi, %rsi
	movq	%rdx, %r10
	cmpq	$32, %r10
	jae	L(gt32)
	/* Handle small chunks and last block of less than 32 bytes.  */
L(small):
	testq	$1, %r10
	jz	L(s2b)
	movzbl	(%rdi),	%eax
	movzbl	(%rdi, %rsi), %edx
	subq    $1, %r10
	je	L(finz1)
	addq	$1, %rdi
	subl	%edx, %eax
	jnz	L(exit)
L(s2b):
	testq	$2, %r10
	jz	L(s4b)
	movzwl	(%rdi),	%eax
	movzwl	(%rdi, %rsi), %edx
	subq    $2, %r10
	je	L(fin2_7)
	addq	$2, %rdi
	cmpl	%edx, %eax
	jnz	L(fin2_7)
L(s4b):
	testq	$4, %r10
	jz	L(s8b)
	movl	(%rdi),	%eax
	movl	(%rdi, %rsi), %edx
	subq    $4, %r10
	je	L(fin2_7)
	addq	$4, %rdi
	cmpl	%edx, %eax
	jnz	L(fin2_7)
L(s8b):
	testq	$8, %r10
	jz	L(s16b)
	movq	(%rdi),	%rax
	movq	(%rdi, %rsi), %rdx
	subq    $8, %r10
	je	L(fin2_7)
	addq	$8, %rdi
	cmpq	%rdx, %rax
	jnz	L(fin2_7)
L(s16b):
	movdqu    (%rdi), %xmm1
	movdqu    (%rdi, %rsi), %xmm0
	pcmpeqb   %xmm0, %xmm1
	pmovmskb  %xmm1, %edx
	xorl	  %eax, %eax
	subl      $0xffff, %edx
	jz	  L(finz)
	bsfl      %edx, %ecx
	leaq	 (%rdi, %rcx), %rcx
	movzbl	 (%rcx), %eax
	movzbl	 (%rsi, %rcx), %edx
	jmp	 L(finz1)

	.p2align 4,, 4
L(finr1b):
	movzbl	(%rdi), %eax
	movzbl  (%rsi), %edx
L(finz1):
	subl	%edx, %eax
L(exit):
	ret

	.p2align 4,, 4
L(fin2_7):
	cmpq	%rdx, %rax
	jz	L(finz)
	movq	%rax, %r11
	subq	%rdx, %r11
	bsfq	%r11, %rcx
	sarq	$3, %rcx
	salq	$3, %rcx
	sarq	%cl, %rax
	movzbl  %al, %eax
	sarq	%cl, %rdx
	movzbl  %dl, %edx
	subl	%edx, %eax
	ret

	.p2align 4,, 4
L(finz):
	xorl	%eax, %eax
	ret

	/* For blocks bigger than 32 bytes
	   1. Advance one of the addr pointer to be 16B aligned.
	   2. Treat the case of both addr pointers aligned to 16B
	      separately to avoid movdqu.
	   3. Handle any blocks of greater than 64 consecutive bytes with
	      unrolling to reduce branches.
	   4. At least one addr pointer is 16B aligned, use memory version
	      of pcmbeqb.
	*/
	.p2align 4,, 4
L(gt32):
	movq	%rdx, %r11
	addq	%rdi, %r11
	movq	%rdi, %r8

	andq	$15, %r8
	jz	L(16am)
	/* Both pointers may be misaligned.  */
	movdqu	(%rdi),	%xmm1
	movdqu	(%rdi, %rsi), %xmm0
	pcmpeqb   %xmm0, %xmm1
	pmovmskb  %xmm1, %edx
	subl      $0xffff, %edx
	jnz       L(neq)
	neg	 %r8
	leaq    16(%rdi, %r8), %rdi
L(16am):
	/* Handle two 16B aligned pointers separately.  */
	testq   $15, %rsi
	jz      L(ATR)
	testq	$16, %rdi
	jz	L(A32)
	movdqu	(%rdi, %rsi), %xmm0
	pcmpeqb   (%rdi), %xmm0
	pmovmskb  %xmm0, %edx
	subl      $0xffff, %edx
	jnz       L(neq)
	addq	$16, %rdi
L(A32):
	movq	%r11, %r10
	andq	$-32, %r10
	cmpq	%r10, %rdi
        jae	L(mt16)
	/* Pre-unroll to be ready for unrolled 64B loop.  */
	testq	$32, %rdi
	jz	L(A64)
	movdqu    (%rdi,%rsi), %xmm0
	pcmpeqb   (%rdi), %xmm0
	pmovmskb  %xmm0, %edx
	subl      $0xffff, %edx
	jnz       L(neq)
	addq       $16, %rdi

	movdqu    (%rdi,%rsi), %xmm0
	pcmpeqb  (%rdi), %xmm0
	pmovmskb  %xmm0, %edx
	subl      $0xffff, %edx
	jnz       L(neq)
	addq       $16, %rdi

L(A64):
	movq	%r11, %r10
	andq	$-64, %r10
	cmpq	%r10, %rdi
        jae	L(mt32)

L(A64main):
	movdqu    (%rdi,%rsi), %xmm0
	pcmpeqb   (%rdi), %xmm0
	pmovmskb  %xmm0, %edx
	subl      $0xffff, %edx
	jnz       L(neq)
	addq       $16, %rdi

	movdqu    (%rdi,%rsi), %xmm0
	pcmpeqb   (%rdi), %xmm0
	pmovmskb  %xmm0, %edx
	subl      $0xffff, %edx
	jnz       L(neq)
	addq       $16, %rdi

	movdqu    (%rdi,%rsi), %xmm0
	pcmpeqb   (%rdi), %xmm0
	pmovmskb  %xmm0, %edx
	subl      $0xffff, %edx
	jnz       L(neq)
	addq       $16, %rdi

	movdqu    (%rdi,%rsi), %xmm0
	pcmpeqb  (%rdi), %xmm0
	pmovmskb  %xmm0, %edx
	subl      $0xffff, %edx
	jnz       L(neq)
	addq       $16, %rdi

	cmpq       %rdi, %r10
	jne       L(A64main)

L(mt32):
	movq	%r11, %r10
	andq	$-32, %r10
	cmpq	%r10, %rdi
        jae	L(mt16)

L(A32main):
	movdqu    (%rdi,%rsi), %xmm0
	pcmpeqb   (%rdi), %xmm0
	pmovmskb  %xmm0, %edx
	subl      $0xffff, %edx
	jnz       L(neq)
	addq       $16, %rdi

	movdqu    (%rdi,%rsi), %xmm0
	pcmpeqb  (%rdi), %xmm0
	pmovmskb  %xmm0, %edx
	subl      $0xffff, %edx
	jnz       L(neq)
	addq       $16, %rdi

	cmpq       %rdi, %r10
	jne       L(A32main)
L(mt16):
	subq       %rdi, %r11
	je	  L(finz)
	movq	  %r11, %r10
	jmp	  L(small)

	.p2align 4,, 4
L(neq):
	bsfl      %edx, %ecx
	movzbl	 (%rdi, %rcx), %eax
	addq	 %rdi, %rsi
	movzbl	 (%rsi,%rcx), %edx
	jmp	 L(finz1)

	.p2align 4,, 4
L(ATR):
	movq	%r11, %r10
	andq	$-32, %r10
	cmpq	%r10, %rdi
        jae	L(mt16)
	testq	$16, %rdi
	jz	L(ATR32)

	movdqa    (%rdi,%rsi), %xmm0
	pcmpeqb   (%rdi), %xmm0
	pmovmskb  %xmm0, %edx
	subl      $0xffff, %edx
	jnz       L(neq)
	addq       $16, %rdi
	cmpq       %rdi, %r10
	je       L(mt16)

L(ATR32):
	movq	%r11, %r10
	andq	$-64, %r10
	testq	$32, %rdi
	jz	L(ATR64)

	movdqa    (%rdi,%rsi), %xmm0
	pcmpeqb   (%rdi), %xmm0
	pmovmskb  %xmm0, %edx
	subl      $0xffff, %edx
	jnz       L(neq)
	addq       $16, %rdi

	movdqa    (%rdi,%rsi), %xmm0
	pcmpeqb   (%rdi), %xmm0
	pmovmskb  %xmm0, %edx
	subl      $0xffff, %edx
	jnz       L(neq)
	addq       $16, %rdi

L(ATR64):
	cmpq       %rdi, %r10
	je	   L(mt32)

L(ATR64main):
	movdqa    (%rdi,%rsi), %xmm0
	pcmpeqb   (%rdi), %xmm0
	pmovmskb  %xmm0, %edx
	subl      $0xffff, %edx
	jnz       L(neq)
	addq       $16, %rdi

	movdqa    (%rdi,%rsi), %xmm0
	pcmpeqb   (%rdi), %xmm0
	pmovmskb  %xmm0, %edx
	subl      $0xffff, %edx
	jnz       L(neq)
	addq       $16, %rdi

	movdqa    (%rdi,%rsi), %xmm0
	pcmpeqb   (%rdi), %xmm0
	pmovmskb  %xmm0, %edx
	subl      $0xffff, %edx
	jnz       L(neq)
	addq       $16, %rdi

	movdqa    (%rdi,%rsi), %xmm0
	pcmpeqb   (%rdi), %xmm0
	pmovmskb  %xmm0, %edx
	subl      $0xffff, %edx
	jnz       L(neq)
	addq       $16, %rdi
	cmpq       %rdi, %r10
	jne       L(ATR64main)

	movq	%r11, %r10
	andq	$-32, %r10
	cmpq	%r10, %rdi
        jae	L(mt16)

L(ATR32res):
	movdqa    (%rdi,%rsi), %xmm0
	pcmpeqb   (%rdi), %xmm0
	pmovmskb  %xmm0, %edx
	subl      $0xffff, %edx
	jnz       L(neq)
	addq       $16, %rdi

	movdqa    (%rdi,%rsi), %xmm0
	pcmpeqb   (%rdi), %xmm0
	pmovmskb  %xmm0, %edx
	subl      $0xffff, %edx
	jnz       L(neq)
	addq       $16, %rdi

	cmpq	  %r10, %rdi
	jne       L(ATR32res)

	subq       %rdi, %r11
	je	  L(finz)
	movq	  %r11, %r10
	jmp	  L(small)
	/* Align to 16byte to improve instruction fetch.  */
	.p2align 4,, 4
END(memcmp)

#undef bcmp
weak_alias (memcmp, bcmp)
libc_hidden_builtin_def (memcmp)
