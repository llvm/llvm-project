/* fast SSE2 memrchr with 64 byte loop and pmaxub instruction using

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
ENTRY (__memrchr)
	movd	%esi, %xmm1

	sub	$16, %RDX_LP
	jbe	L(length_less16)

	punpcklbw	%xmm1, %xmm1
	punpcklbw	%xmm1, %xmm1

	add	%RDX_LP, %RDI_LP
	pshufd	$0, %xmm1, %xmm1

	movdqu	(%rdi), %xmm0
	pcmpeqb	%xmm1, %xmm0

/* Check if there is a match.  */
	pmovmskb	%xmm0, %eax
	test	%eax, %eax
	jnz	L(matches0)

	sub	$64, %rdi
	mov	%edi, %ecx
	and	$15, %ecx
	jz	L(loop_prolog)

	add	$16, %rdi
	add	$16, %rdx
	and	$-16, %rdi
	sub	%rcx, %rdx

	.p2align 4
L(loop_prolog):
	sub	$64, %rdx
	jbe	L(exit_loop)

	movdqa	48(%rdi), %xmm0
	pcmpeqb	%xmm1, %xmm0
	pmovmskb	%xmm0, %eax
	test	%eax, %eax
	jnz	L(matches48)

	movdqa	32(%rdi), %xmm2
	pcmpeqb	%xmm1, %xmm2
	pmovmskb	%xmm2, %eax
	test	%eax, %eax
	jnz	L(matches32)

	movdqa	16(%rdi), %xmm3
	pcmpeqb	%xmm1, %xmm3
	pmovmskb	%xmm3, %eax
	test	%eax, %eax
	jnz	L(matches16)

	movdqa	(%rdi), %xmm4
	pcmpeqb	%xmm1, %xmm4
	pmovmskb	%xmm4, %eax
	test	%eax, %eax
	jnz	L(matches0)

	sub	$64, %rdi
	sub	$64, %rdx
	jbe	L(exit_loop)

	movdqa	48(%rdi), %xmm0
	pcmpeqb	%xmm1, %xmm0
	pmovmskb	%xmm0, %eax
	test	%eax, %eax
	jnz	L(matches48)

	movdqa	32(%rdi), %xmm2
	pcmpeqb	%xmm1, %xmm2
	pmovmskb	%xmm2, %eax
	test	%eax, %eax
	jnz	L(matches32)

	movdqa	16(%rdi), %xmm3
	pcmpeqb	%xmm1, %xmm3
	pmovmskb	%xmm3, %eax
	test	%eax, %eax
	jnz	L(matches16)

	movdqa	(%rdi), %xmm3
	pcmpeqb	%xmm1, %xmm3
	pmovmskb	%xmm3, %eax
	test	%eax, %eax
	jnz	L(matches0)

	mov	%edi, %ecx
	and	$63, %ecx
	jz	L(align64_loop)

	add	$64, %rdi
	add	$64, %rdx
	and	$-64, %rdi
	sub	%rcx, %rdx

	.p2align 4
L(align64_loop):
	sub	$64, %rdi
	sub	$64, %rdx
	jbe	L(exit_loop)

	movdqa	(%rdi), %xmm0
	movdqa	16(%rdi), %xmm2
	movdqa	32(%rdi), %xmm3
	movdqa	48(%rdi), %xmm4

	pcmpeqb	%xmm1, %xmm0
	pcmpeqb	%xmm1, %xmm2
	pcmpeqb	%xmm1, %xmm3
	pcmpeqb	%xmm1, %xmm4

	pmaxub	%xmm3, %xmm0
	pmaxub	%xmm4, %xmm2
	pmaxub	%xmm0, %xmm2
	pmovmskb	%xmm2, %eax

	test	%eax, %eax
	jz	L(align64_loop)

	pmovmskb	%xmm4, %eax
	test	%eax, %eax
	jnz	L(matches48)

	pmovmskb	%xmm3, %eax
	test	%eax, %eax
	jnz	L(matches32)

	movdqa	16(%rdi), %xmm2

	pcmpeqb	%xmm1, %xmm2
	pcmpeqb	(%rdi), %xmm1

	pmovmskb	%xmm2, %eax
	test	%eax, %eax
	jnz	L(matches16)

	pmovmskb	%xmm1, %eax
	bsr	%eax, %eax

	add	%rdi, %rax
	ret

	.p2align 4
L(exit_loop):
	add	$64, %edx
	cmp	$32, %edx
	jbe	L(exit_loop_32)

	movdqa	48(%rdi), %xmm0
	pcmpeqb	%xmm1, %xmm0
	pmovmskb	%xmm0, %eax
	test	%eax, %eax
	jnz	L(matches48)

	movdqa	32(%rdi), %xmm2
	pcmpeqb	%xmm1, %xmm2
	pmovmskb	%xmm2, %eax
	test	%eax, %eax
	jnz	L(matches32)

	movdqa	16(%rdi), %xmm3
	pcmpeqb	%xmm1, %xmm3
	pmovmskb	%xmm3, %eax
	test	%eax, %eax
	jnz	L(matches16_1)
	cmp	$48, %edx
	jbe	L(return_null)

	pcmpeqb	(%rdi), %xmm1
	pmovmskb	%xmm1, %eax
	test	%eax, %eax
	jnz	L(matches0_1)
	xor	%eax, %eax
	ret

	.p2align 4
L(exit_loop_32):
	movdqa	48(%rdi), %xmm0
	pcmpeqb	%xmm1, %xmm0
	pmovmskb	%xmm0, %eax
	test	%eax, %eax
	jnz	L(matches48_1)
	cmp	$16, %edx
	jbe	L(return_null)

	pcmpeqb	32(%rdi), %xmm1
	pmovmskb	%xmm1, %eax
	test	%eax, %eax
	jnz	L(matches32_1)
	xor	%eax, %eax
	ret

	.p2align 4
L(matches0):
	bsr	%eax, %eax
	add	%rdi, %rax
	ret

	.p2align 4
L(matches16):
	bsr	%eax, %eax
	lea	16(%rax, %rdi), %rax
	ret

	.p2align 4
L(matches32):
	bsr	%eax, %eax
	lea	32(%rax, %rdi), %rax
	ret

	.p2align 4
L(matches48):
	bsr	%eax, %eax
	lea	48(%rax, %rdi), %rax
	ret

	.p2align 4
L(matches0_1):
	bsr	%eax, %eax
	sub	$64, %rdx
	add	%rax, %rdx
	jl	L(return_null)
	add	%rdi, %rax
	ret

	.p2align 4
L(matches16_1):
	bsr	%eax, %eax
	sub	$48, %rdx
	add	%rax, %rdx
	jl	L(return_null)
	lea	16(%rdi, %rax), %rax
	ret

	.p2align 4
L(matches32_1):
	bsr	%eax, %eax
	sub	$32, %rdx
	add	%rax, %rdx
	jl	L(return_null)
	lea	32(%rdi, %rax), %rax
	ret

	.p2align 4
L(matches48_1):
	bsr	%eax, %eax
	sub	$16, %rdx
	add	%rax, %rdx
	jl	L(return_null)
	lea	48(%rdi, %rax), %rax
	ret

	.p2align 4
L(return_null):
	xor	%eax, %eax
	ret

	.p2align 4
L(length_less16_offset0):
	test	%edx, %edx
	jz	L(return_null)

	mov	%dl, %cl
	pcmpeqb	(%rdi), %xmm1

	mov	$1, %edx
	sal	%cl, %edx
	sub	$1, %edx

	pmovmskb	%xmm1, %eax

	and	%edx, %eax
	test	%eax, %eax
	jz	L(return_null)

	bsr	%eax, %eax
	add	%rdi, %rax
	ret

	.p2align 4
L(length_less16):
	punpcklbw	%xmm1, %xmm1
	punpcklbw	%xmm1, %xmm1

	add	$16, %edx

	pshufd	$0, %xmm1, %xmm1

	mov	%edi, %ecx
	and	$15, %ecx
	jz	L(length_less16_offset0)

	mov	%cl, %dh
	mov	%ecx, %esi
	add	%dl, %dh
	and	$-16, %rdi

	sub	$16, %dh
	ja	L(length_less16_part2)

	pcmpeqb	(%rdi), %xmm1
	pmovmskb	%xmm1, %eax

	sar	%cl, %eax
	mov	%dl, %cl

	mov	$1, %edx
	sal	%cl, %edx
	sub	$1, %edx

	and	%edx, %eax
	test	%eax, %eax
	jz	L(return_null)

	bsr	%eax, %eax
	add	%rdi, %rax
	add	%rsi, %rax
	ret

	.p2align 4
L(length_less16_part2):
	movdqa	16(%rdi), %xmm2
	pcmpeqb	%xmm1, %xmm2
	pmovmskb	%xmm2, %eax

	mov	%dh, %cl
	mov	$1, %edx
	sal	%cl, %edx
	sub	$1, %edx

	and	%edx, %eax

	test	%eax, %eax
	jnz	L(length_less16_part2_return)

	pcmpeqb	(%rdi), %xmm1
	pmovmskb	%xmm1, %eax

	mov	%esi, %ecx
	sar	%cl, %eax
	test	%eax, %eax
	jz	L(return_null)

	bsr	%eax, %eax
	add	%rdi, %rax
	add	%rsi, %rax
	ret

	.p2align 4
L(length_less16_part2_return):
	bsr	%eax, %eax
	lea	16(%rax, %rdi), %rax
	ret

END (__memrchr)
weak_alias (__memrchr, memrchr)
