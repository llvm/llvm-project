/* wcsrchr with SSSE3
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
ENTRY (wcsrchr)

	movd	%rsi, %xmm1
	mov	%rdi, %rcx
	punpckldq %xmm1, %xmm1
	pxor	%xmm2, %xmm2
	punpckldq %xmm1, %xmm1
	and	$63, %rcx
	cmp	$48, %rcx
	ja	L(crosscache)

	movdqu	(%rdi), %xmm0
	pcmpeqd	%xmm0, %xmm2
	pcmpeqd	%xmm1, %xmm0
	pmovmskb %xmm2, %rcx
	pmovmskb %xmm0, %rax
	add	$16, %rdi

	test	%rax, %rax
	jnz	L(unaligned_match1)

	test	%rcx, %rcx
	jnz	L(return_null)

	and	$-16, %rdi
	xor	%r8, %r8
	jmp	L(loop)

	.p2align 4
L(unaligned_match1):
	test	%rcx, %rcx
	jnz	L(prolog_find_zero_1)

	mov	%rax, %r8
	mov	%rdi, %rsi
	and	$-16, %rdi
	jmp	L(loop)

	.p2align 4
L(crosscache):
	and	$15, %rcx
	and	$-16, %rdi
	pxor	%xmm3, %xmm3
	movdqa	(%rdi), %xmm0
	pcmpeqd	%xmm0, %xmm3
	pcmpeqd	%xmm1, %xmm0
	pmovmskb %xmm3, %rdx
	pmovmskb %xmm0, %rax
	shr	%cl, %rdx
	shr	%cl, %rax
	add	$16, %rdi

	test	%rax, %rax
	jnz	L(unaligned_match)

	test	%rdx, %rdx
	jnz	L(return_null)

	xor	%r8, %r8
	jmp	L(loop)

	.p2align 4
L(unaligned_match):
	test	%rdx, %rdx
	jnz	L(prolog_find_zero)

	mov	%rax, %r8
	lea	(%rdi, %rcx), %rsi

/* Loop start on aligned string.  */
	.p2align 4
L(loop):
	movdqa	(%rdi), %xmm0
	pcmpeqd	%xmm0, %xmm2
	add	$16, %rdi
	pcmpeqd	%xmm1, %xmm0
	pmovmskb %xmm2, %rcx
	pmovmskb %xmm0, %rax
	or	%rax, %rcx
	jnz	L(matches)

	movdqa	(%rdi), %xmm3
	pcmpeqd	%xmm3, %xmm2
	add	$16, %rdi
	pcmpeqd	%xmm1, %xmm3
	pmovmskb %xmm2, %rcx
	pmovmskb %xmm3, %rax
	or	%rax, %rcx
	jnz	L(matches)

	movdqa	(%rdi), %xmm4
	pcmpeqd	%xmm4, %xmm2
	add	$16, %rdi
	pcmpeqd	%xmm1, %xmm4
	pmovmskb %xmm2, %rcx
	pmovmskb %xmm4, %rax
	or	%rax, %rcx
	jnz	L(matches)

	movdqa	(%rdi), %xmm5
	pcmpeqd	%xmm5, %xmm2
	add	$16, %rdi
	pcmpeqd	%xmm1, %xmm5
	pmovmskb %xmm2, %rcx
	pmovmskb %xmm5, %rax
	or	%rax, %rcx
	jz	L(loop)

	.p2align 4
L(matches):
	test	%rax, %rax
	jnz	L(match)
L(return_value):
	test	%r8, %r8
	jz	L(return_null)
	mov	%r8, %rax
	mov	%rsi, %rdi

	test	$15 << 4, %ah
	jnz	L(match_fourth_wchar)
	test	%ah, %ah
	jnz	L(match_third_wchar)
	test	$15 << 4, %al
	jnz	L(match_second_wchar)
	lea	-16(%rdi), %rax
	ret

	.p2align 4
L(match):
	pmovmskb %xmm2, %rcx
	test	%rcx, %rcx
	jnz	L(find_zero)
	mov	%rax, %r8
	mov	%rdi, %rsi
	jmp	L(loop)

	.p2align 4
L(find_zero):
	test	$15, %cl
	jnz	L(find_zero_in_first_wchar)
	test	%cl, %cl
	jnz	L(find_zero_in_second_wchar)
	test	$15, %ch
	jnz	L(find_zero_in_third_wchar)

	and	$1 << 13 - 1, %rax
	jz	L(return_value)

	test	$15 << 4, %ah
	jnz	L(match_fourth_wchar)
	test	%ah, %ah
	jnz	L(match_third_wchar)
	test	$15 << 4, %al
	jnz	L(match_second_wchar)
	lea	-16(%rdi), %rax
	ret

	.p2align 4
L(find_zero_in_first_wchar):
	test	$1, %rax
	jz	L(return_value)
	lea	-16(%rdi), %rax
	ret

	.p2align 4
L(find_zero_in_second_wchar):
	and	$1 << 5 - 1, %rax
	jz	L(return_value)

	test	$15 << 4, %al
	jnz	L(match_second_wchar)
	lea	-16(%rdi), %rax
	ret

	.p2align 4
L(find_zero_in_third_wchar):
	and	$1 << 9 - 1, %rax
	jz	L(return_value)

	test	%ah, %ah
	jnz	L(match_third_wchar)
	test	$15 << 4, %al
	jnz	L(match_second_wchar)
	lea	-16(%rdi), %rax
	ret

	.p2align 4
L(prolog_find_zero):
	add	%rcx, %rdi
	mov     %rdx, %rcx
L(prolog_find_zero_1):
	test	$15, %cl
	jnz	L(prolog_find_zero_in_first_wchar)
	test	%cl, %cl
	jnz	L(prolog_find_zero_in_second_wchar)
	test	$15, %ch
	jnz	L(prolog_find_zero_in_third_wchar)

	and	$1 << 13 - 1, %rax
	jz	L(return_null)

	test	$15 << 4, %ah
	jnz	L(match_fourth_wchar)
	test	%ah, %ah
	jnz	L(match_third_wchar)
	test	$15 << 4, %al
	jnz	L(match_second_wchar)
	lea	-16(%rdi), %rax
	ret

	.p2align 4
L(prolog_find_zero_in_first_wchar):
	test	$1, %rax
	jz	L(return_null)
	lea	-16(%rdi), %rax
	ret

	.p2align 4
L(prolog_find_zero_in_second_wchar):
	and	$1 << 5 - 1, %rax
	jz	L(return_null)

	test	$15 << 4, %al
	jnz	L(match_second_wchar)
	lea	-16(%rdi), %rax
	ret

	.p2align 4
L(prolog_find_zero_in_third_wchar):
	and	$1 << 9 - 1, %rax
	jz	L(return_null)

	test	%ah, %ah
	jnz	L(match_third_wchar)
	test	$15 << 4, %al
	jnz	L(match_second_wchar)
	lea	-16(%rdi), %rax
	ret

	.p2align 4
L(match_second_wchar):
	lea	-12(%rdi), %rax
	ret

	.p2align 4
L(match_third_wchar):
	lea	-8(%rdi), %rax
	ret

	.p2align 4
L(match_fourth_wchar):
	lea	-4(%rdi), %rax
	ret

	.p2align 4
L(return_null):
	xor	%rax, %rax
	ret

END (wcsrchr)
