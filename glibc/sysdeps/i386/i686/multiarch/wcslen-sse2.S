/* wcslen with SSE2
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

#if IS_IN (libc)
# include <sysdep.h>
# define STR	4

	.text
ENTRY (__wcslen_sse2)
	mov	STR(%esp), %edx

	cmp	$0, (%edx)
	jz	L(exit_tail0)
	cmp	$0, 4(%edx)
	jz	L(exit_tail1)
	cmp	$0, 8(%edx)
	jz	L(exit_tail2)
	cmp	$0, 12(%edx)
	jz	L(exit_tail3)
	cmp	$0, 16(%edx)
	jz	L(exit_tail4)
	cmp	$0, 20(%edx)
	jz	L(exit_tail5)
	cmp	$0, 24(%edx)
	jz	L(exit_tail6)
	cmp	$0, 28(%edx)
	jz	L(exit_tail7)

	pxor	%xmm0, %xmm0

	lea	32(%edx), %eax
	lea	16(%edx), %ecx
	and	$-16, %eax

	pcmpeqd	(%eax), %xmm0
	pmovmskb %xmm0, %edx
	pxor	%xmm1, %xmm1
	test	%edx, %edx
	lea	16(%eax), %eax
	jnz	L(exit)

	pcmpeqd	(%eax), %xmm1
	pmovmskb %xmm1, %edx
	pxor	%xmm2, %xmm2
	test	%edx, %edx
	lea	16(%eax), %eax
	jnz	L(exit)

	pcmpeqd	(%eax), %xmm2
	pmovmskb %xmm2, %edx
	pxor	%xmm3, %xmm3
	test	%edx, %edx
	lea	16(%eax), %eax
	jnz	L(exit)

	pcmpeqd	(%eax), %xmm3
	pmovmskb %xmm3, %edx
	test	%edx, %edx
	lea	16(%eax), %eax
	jnz	L(exit)

	and	$-0x40, %eax

	.p2align 4
L(aligned_64_loop):
	movaps	(%eax), %xmm0
	movaps	16(%eax), %xmm1
	movaps	32(%eax), %xmm2
	movaps	48(%eax), %xmm6

	pminub	%xmm1, %xmm0
	pminub	%xmm6, %xmm2
	pminub	%xmm0, %xmm2
	pcmpeqd	%xmm3, %xmm2
	pmovmskb %xmm2, %edx
	test	%edx, %edx
	lea	64(%eax), %eax
	jz	L(aligned_64_loop)

	pcmpeqd	-64(%eax), %xmm3
	pmovmskb %xmm3, %edx
	test	%edx, %edx
	lea	48(%ecx), %ecx
	jnz	L(exit)

	pcmpeqd	%xmm1, %xmm3
	pmovmskb %xmm3, %edx
	test	%edx, %edx
	lea	-16(%ecx), %ecx
	jnz	L(exit)

	pcmpeqd	-32(%eax), %xmm3
	pmovmskb %xmm3, %edx
	test	%edx, %edx
	lea	-16(%ecx), %ecx
	jnz	L(exit)

	pcmpeqd	%xmm6, %xmm3
	pmovmskb %xmm3, %edx
	test	%edx, %edx
	lea	-16(%ecx), %ecx
	jnz	L(exit)

	jmp	L(aligned_64_loop)

	.p2align 4
L(exit):
	sub	%ecx, %eax
	shr	$2, %eax
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
	add	$2, %eax
	ret

	.p2align 4
L(exit_1):
	add	$1, %eax
	ret

	.p2align 4
L(exit_3):
	add	$3, %eax
	ret

	.p2align 4
L(exit_tail0):
	xor	%eax, %eax
	ret

	.p2align 4
L(exit_tail1):
	mov	$1, %eax
	ret

	.p2align 4
L(exit_tail2):
	mov	$2, %eax
	ret

	.p2align 4
L(exit_tail3):
	mov	$3, %eax
	ret

	.p2align 4
L(exit_tail4):
	mov	$4, %eax
	ret

	.p2align 4
L(exit_tail5):
	mov	$5, %eax
	ret

	.p2align 4
L(exit_tail6):
	mov	$6, %eax
	ret

	.p2align 4
L(exit_tail7):
	mov	$7, %eax
	ret

END (__wcslen_sse2)
#endif
