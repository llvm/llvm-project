/* strchr with SSE2 without bsf
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
# include "asm-syntax.h"

	atom_text_section
ENTRY (__strchr_sse2_no_bsf)
	movd	%esi, %xmm1
	movq	%rdi, %rcx
	punpcklbw %xmm1, %xmm1
	andq	$~15, %rdi
	pxor	%xmm2, %xmm2
	punpcklbw %xmm1, %xmm1
	orl	$0xffffffff, %esi
	movdqa	(%rdi), %xmm0
	pshufd	$0, %xmm1, %xmm1
	subq	%rdi, %rcx
	movdqa	%xmm0, %xmm3
	leaq	16(%rdi), %rdi
	pcmpeqb	%xmm1, %xmm0
	pcmpeqb	%xmm2, %xmm3
	shl	%cl, %esi
	pmovmskb %xmm0, %eax
	pmovmskb %xmm3, %edx
	andl	%esi, %eax
	andl	%esi, %edx
	test	%eax, %eax
	jnz	L(matches)
	test	%edx, %edx
	jnz	L(return_null)

L(loop):
	movdqa	(%rdi), %xmm0
	leaq	16(%rdi), %rdi
	movdqa	%xmm0, %xmm3
	pcmpeqb	%xmm1, %xmm0
	pcmpeqb	%xmm2, %xmm3
	pmovmskb %xmm0, %eax
	pmovmskb %xmm3, %edx
	or	%eax, %edx
	jz	L(loop)

	pmovmskb %xmm3, %edx
	test	%eax, %eax
	jnz	L(matches)

/* Return NULL.  */
	.p2align 4
L(return_null):
	xor	%rax, %rax
	ret

L(matches):
	/* There is a match.  First find where NULL is.  */
	leaq	-16(%rdi), %rdi
	test	%edx, %edx
	jz	L(match_case1)

	.p2align 4
L(match_case2):
	test	%al, %al
	jz	L(match_high_case2)

	mov	%al, %cl
	and	$15, %cl
	jnz	L(match_case2_4)

	mov	%dl, %ch
	and	$15, %ch
	jnz	L(return_null)

	test	$0x10, %al
	jnz	L(Exit5)
	test	$0x10, %dl
	jnz	L(return_null)
	test	$0x20, %al
	jnz	L(Exit6)
	test	$0x20, %dl
	jnz	L(return_null)
	test	$0x40, %al
	jnz	L(Exit7)
	test	$0x40, %dl
	jnz	L(return_null)
	lea	7(%rdi), %rax
	ret

	.p2align 4
L(match_case2_4):
	test	$0x01, %al
	jnz	L(Exit1)
	test	$0x01, %dl
	jnz	L(return_null)
	test	$0x02, %al
	jnz	L(Exit2)
	test	$0x02, %dl
	jnz	L(return_null)
	test	$0x04, %al
	jnz	L(Exit3)
	test	$0x04, %dl
	jnz	L(return_null)
	lea	3(%rdi), %rax
	ret

	.p2align 4
L(match_high_case2):
	test	%dl, %dl
	jnz	L(return_null)

	mov	%ah, %cl
	and	$15, %cl
	jnz	L(match_case2_12)

	mov	%dh, %ch
	and	$15, %ch
	jnz	L(return_null)

	test	$0x10, %ah
	jnz	L(Exit13)
	test	$0x10, %dh
	jnz	L(return_null)
	test	$0x20, %ah
	jnz	L(Exit14)
	test	$0x20, %dh
	jnz	L(return_null)
	test	$0x40, %ah
	jnz	L(Exit15)
	test	$0x40, %dh
	jnz	L(return_null)
	lea	15(%rdi), %rax
	ret

	.p2align 4
L(match_case2_12):
	test	$0x01, %ah
	jnz	L(Exit9)
	test	$0x01, %dh
	jnz	L(return_null)
	test	$0x02, %ah
	jnz	L(Exit10)
	test	$0x02, %dh
	jnz	L(return_null)
	test	$0x04, %ah
	jnz	L(Exit11)
	test	$0x04, %dh
	jnz	L(return_null)
	lea	11(%rdi), %rax
	ret

	.p2align 4
L(match_case1):
	test	%al, %al
	jz	L(match_high_case1)

	test	$0x01, %al
	jnz	L(Exit1)
	test	$0x02, %al
	jnz	L(Exit2)
	test	$0x04, %al
	jnz	L(Exit3)
	test	$0x08, %al
	jnz	L(Exit4)
	test	$0x10, %al
	jnz	L(Exit5)
	test	$0x20, %al
	jnz	L(Exit6)
	test	$0x40, %al
	jnz	L(Exit7)
	lea	7(%rdi), %rax
	ret

	.p2align 4
L(match_high_case1):
	test	$0x01, %ah
	jnz	L(Exit9)
	test	$0x02, %ah
	jnz	L(Exit10)
	test	$0x04, %ah
	jnz	L(Exit11)
	test	$0x08, %ah
	jnz	L(Exit12)
	test	$0x10, %ah
	jnz	L(Exit13)
	test	$0x20, %ah
	jnz	L(Exit14)
	test	$0x40, %ah
	jnz	L(Exit15)
	lea	15(%rdi), %rax
	ret

	.p2align 4
L(Exit1):
	lea	(%rdi), %rax
	ret

	.p2align 4
L(Exit2):
	lea	1(%rdi), %rax
	ret

	.p2align 4
L(Exit3):
	lea	2(%rdi), %rax
	ret

	.p2align 4
L(Exit4):
	lea	3(%rdi), %rax
	ret

	.p2align 4
L(Exit5):
	lea	4(%rdi), %rax
	ret

	.p2align 4
L(Exit6):
	lea	5(%rdi), %rax
	ret

	.p2align 4
L(Exit7):
	lea	6(%rdi), %rax
	ret

	.p2align 4
L(Exit9):
	lea	8(%rdi), %rax
	ret

	.p2align 4
L(Exit10):
	lea	9(%rdi), %rax
	ret

	.p2align 4
L(Exit11):
	lea	10(%rdi), %rax
	ret

	.p2align 4
L(Exit12):
	lea	11(%rdi), %rax
	ret

	.p2align 4
L(Exit13):
	lea	12(%rdi), %rax
	ret

	.p2align 4
L(Exit14):
	lea	13(%rdi), %rax
	ret

	.p2align 4
L(Exit15):
	lea	14(%rdi), %rax
	ret

END (__strchr_sse2_no_bsf)
#endif
