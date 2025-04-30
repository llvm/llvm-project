/* wcschr with SSSE3
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
ENTRY (__wcschr)

	movd	%rsi, %xmm1
	pxor	%xmm2, %xmm2
	mov	%rdi, %rcx
	punpckldq %xmm1, %xmm1
	punpckldq %xmm1, %xmm1

	and	$63, %rcx
	cmp	$48, %rcx
	ja	L(cross_cache)

	movdqu	(%rdi), %xmm0
	pcmpeqd	%xmm0, %xmm2
	add	$16, %rdi
	pcmpeqd	%xmm1, %xmm0
	pmovmskb %xmm2, %rdx
	pmovmskb %xmm0, %rax
	or	%rax, %rdx
	jnz	L(matches)

	and	$-16, %rdi

	movdqa	(%rdi), %xmm0
	pcmpeqd	%xmm0, %xmm2
	add	$16, %rdi
	pcmpeqd	%xmm1, %xmm0
	pmovmskb %xmm2, %rdx
	pmovmskb %xmm0, %rax
	or	%rax, %rdx
	jnz	L(matches)

	jmp	L(loop)

L(cross_cache):
	and	$15, %rcx
	and	$-16, %rdi
	movdqa	(%rdi), %xmm0
	pcmpeqd	%xmm0, %xmm2
	pcmpeqd	%xmm1, %xmm0
	pmovmskb %xmm2, %rdx
	pmovmskb %xmm0, %rax

	sar	%cl, %rdx
	sar	%cl, %rax
	test	%rax, %rax
	je	L(unaligned_no_match)

	bsf	%rax, %rax
	test	%rdx, %rdx
	je	L(unaligned_match)
	bsf	%rdx, %rdx
	cmp	%rdx, %rax
	ja	L(return_null)

L(unaligned_match):
	add	%rdi, %rax
	add	%rcx, %rax
	ret

	.p2align 4
L(unaligned_no_match):
	test	%rdx, %rdx
	jne	L(return_null)
	pxor	%xmm2, %xmm2

	add	$16, %rdi

	.p2align 4
/* Loop start on aligned string.  */
L(loop):
	movdqa	(%rdi), %xmm0
	pcmpeqd	%xmm0, %xmm2
	add	$16, %rdi
	pcmpeqd	%xmm1, %xmm0
	pmovmskb %xmm2, %rdx
	pmovmskb %xmm0, %rax
	or	%rax, %rdx
	jnz	L(matches)

	movdqa	(%rdi), %xmm0
	pcmpeqd	%xmm0, %xmm2
	add	$16, %rdi
	pcmpeqd	%xmm1, %xmm0
	pmovmskb %xmm2, %rdx
	pmovmskb %xmm0, %rax
	or	%rax, %rdx
	jnz	L(matches)

	movdqa	(%rdi), %xmm0
	pcmpeqd	%xmm0, %xmm2
	add	$16, %rdi
	pcmpeqd	%xmm1, %xmm0
	pmovmskb %xmm2, %rdx
	pmovmskb %xmm0, %rax
	or	%rax, %rdx
	jnz	L(matches)

	movdqa	(%rdi), %xmm0
	pcmpeqd	%xmm0, %xmm2
	add	$16, %rdi
	pcmpeqd	%xmm1, %xmm0
	pmovmskb %xmm2, %rdx
	pmovmskb %xmm0, %rax
	or	%rax, %rdx
	jnz	L(matches)
	jmp	L(loop)

	.p2align 4
L(matches):
	pmovmskb %xmm2, %rdx
	test	%rax, %rax
	jz	L(return_null)
	bsf	%rax, %rax
	test	%rdx, %rdx
	je	L(match)
	bsf	%rdx, %rcx
	cmp	%rcx, %rax
	ja	L(return_null)
L(match):
	sub	$16, %rdi
	add	%rdi, %rax
	ret

	.p2align 4
L(return_null):
	xor	%rax, %rax
	ret

END (__wcschr)

libc_hidden_def(__wcschr)
weak_alias (__wcschr, wcschr)
libc_hidden_weak (wcschr)
