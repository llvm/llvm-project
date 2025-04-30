/* fast SSE2 memchr with 64 byte loop and pmaxub instruction using

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
ENTRY (__rawmemchr)
	movd	%rsi, %xmm1
	mov	%rdi, %rcx

	punpcklbw %xmm1, %xmm1
	punpcklbw %xmm1, %xmm1

	and	$63, %rcx
	pshufd	$0, %xmm1, %xmm1

	cmp	$48, %rcx
	ja	L(crosscache)

	movdqu	(%rdi), %xmm0
	pcmpeqb	%xmm1, %xmm0
/* Check if there is a match.  */
	pmovmskb %xmm0, %eax
	test	%eax, %eax

	jnz	L(matches)
	add	$16, %rdi
	and	$-16, %rdi
	jmp	L(loop_prolog)

	.p2align 4
L(crosscache):
	and	$15, %rcx
	and	$-16, %rdi
	movdqa	(%rdi), %xmm0

	pcmpeqb	%xmm1, %xmm0
/* Check if there is a match.  */
	pmovmskb %xmm0, %eax
/* Remove the leading bytes.  */
	sar	%cl, %eax
	test	%eax, %eax
	je	L(unaligned_no_match)
/* Check which byte is a match.  */
	bsf	%eax, %eax

	add	%rdi, %rax
	add	%rcx, %rax
	ret

	.p2align 4
L(unaligned_no_match):
	add	$16, %rdi

	.p2align 4
L(loop_prolog):
	movdqa	(%rdi), %xmm0
	pcmpeqb	%xmm1, %xmm0
	pmovmskb %xmm0, %eax
	test	%eax, %eax
	jnz	L(matches)

	movdqa	16(%rdi), %xmm2
	pcmpeqb	%xmm1, %xmm2
	pmovmskb %xmm2, %eax
	test	%eax, %eax
	jnz	L(matches16)

	movdqa	32(%rdi), %xmm3
	pcmpeqb	%xmm1, %xmm3
	pmovmskb %xmm3, %eax
	test	%eax, %eax
	jnz	L(matches32)

	movdqa	48(%rdi), %xmm4
	pcmpeqb	%xmm1, %xmm4
	add	$64, %rdi
	pmovmskb %xmm4, %eax
	test	%eax, %eax
	jnz	L(matches0)

	test	$0x3f, %rdi
	jz	L(align64_loop)

	movdqa	(%rdi), %xmm0
	pcmpeqb	%xmm1, %xmm0
	pmovmskb %xmm0, %eax
	test	%eax, %eax
	jnz	L(matches)

	movdqa	16(%rdi), %xmm2
	pcmpeqb	%xmm1, %xmm2
	pmovmskb %xmm2, %eax
	test	%eax, %eax
	jnz	L(matches16)

	movdqa	32(%rdi), %xmm3
	pcmpeqb	%xmm1, %xmm3
	pmovmskb %xmm3, %eax
	test	%eax, %eax
	jnz	L(matches32)

	movdqa	48(%rdi), %xmm3
	pcmpeqb	%xmm1, %xmm3
	pmovmskb %xmm3, %eax

	add	$64, %rdi
	test	%eax, %eax
	jnz	L(matches0)

	and	$-64, %rdi

	.p2align 4
L(align64_loop):
	movdqa	(%rdi), %xmm0
	movdqa	16(%rdi), %xmm2
	movdqa	32(%rdi), %xmm3
	movdqa	48(%rdi), %xmm4

	pcmpeqb	%xmm1, %xmm0
	pcmpeqb	%xmm1, %xmm2
	pcmpeqb	%xmm1, %xmm3
	pcmpeqb	%xmm1, %xmm4

	pmaxub	%xmm0, %xmm3
	pmaxub	%xmm2, %xmm4
	pmaxub	%xmm3, %xmm4
	pmovmskb %xmm4, %eax

	add	$64, %rdi

	test	%eax, %eax
	jz	L(align64_loop)

	sub	$64, %rdi

	pmovmskb %xmm0, %eax
	test	%eax, %eax
	jnz	L(matches)

	pmovmskb %xmm2, %eax
	test	%eax, %eax
	jnz	L(matches16)

	movdqa	32(%rdi), %xmm3
	pcmpeqb	%xmm1, %xmm3

	pcmpeqb	48(%rdi), %xmm1
	pmovmskb %xmm3, %eax
	test	%eax, %eax
	jnz	L(matches32)

	pmovmskb %xmm1, %eax
	bsf	%eax, %eax
	lea	48(%rdi, %rax), %rax
	ret

	.p2align 4
L(matches0):
	bsf	%eax, %eax
	lea	-16(%rax, %rdi), %rax
	ret

	.p2align 4
L(matches):
	bsf	%eax, %eax
	add	%rdi, %rax
	ret

	.p2align 4
L(matches16):
	bsf	%eax, %eax
	lea	16(%rax, %rdi), %rax
	ret

	.p2align 4
L(matches32):
	bsf	%eax, %eax
	lea	32(%rax, %rdi), %rax
	ret

END (__rawmemchr)

weak_alias (__rawmemchr, rawmemchr)
libc_hidden_builtin_def (__rawmemchr)
