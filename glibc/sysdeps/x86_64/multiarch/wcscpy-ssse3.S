/* wcscpy with SSSE3
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

	.section .text.ssse3,"ax",@progbits
ENTRY (__wcscpy_ssse3)

	mov	%rsi, %rcx
	mov	%rdi, %rdx

	cmpl	$0, (%rcx)
	jz	L(Exit4)
	cmpl	$0, 4(%rcx)
	jz	L(Exit8)
	cmpl	$0, 8(%rcx)
	jz	L(Exit12)
	cmpl	$0, 12(%rcx)
	jz	L(Exit16)

	lea	16(%rcx), %rsi
	and	$-16, %rsi

	pxor	%xmm0, %xmm0
	mov	(%rcx), %r9
	mov	%r9, (%rdx)

	pcmpeqd	(%rsi), %xmm0
	mov	8(%rcx), %r9
	mov	%r9, 8(%rdx)

	pmovmskb %xmm0, %rax
	sub	%rcx, %rsi

	test	%rax, %rax
	jnz	L(CopyFrom1To16Bytes)

	mov	%rdx, %rax
	lea	16(%rdx), %rdx
	and	$-16, %rdx
	sub	%rdx, %rax
	sub	%rax, %rcx
	mov	%rcx, %rax
	and	$0xf, %rax
	mov	$0, %rsi

/* case: rcx_offset == rdx_offset */

	jz	L(Align16Both)

	cmp	$4, %rax
	je	L(Shl4)
	cmp	$8, %rax
	je	L(Shl8)
	jmp	L(Shl12)

L(Align16Both):
	movaps	(%rcx), %xmm1
	movaps	16(%rcx), %xmm2
	movaps	%xmm1, (%rdx)
	pcmpeqd	%xmm2, %xmm0
	pmovmskb %xmm0, %rax
	lea	16(%rsi), %rsi

	test	%rax, %rax
	jnz	L(CopyFrom1To16Bytes)

	movaps	16(%rcx, %rsi), %xmm3
	movaps	%xmm2, (%rdx, %rsi)
	pcmpeqd	%xmm3, %xmm0
	pmovmskb %xmm0, %rax
	lea	16(%rsi), %rsi

	test	%rax, %rax
	jnz	L(CopyFrom1To16Bytes)

	movaps	16(%rcx, %rsi), %xmm4
	movaps	%xmm3, (%rdx, %rsi)
	pcmpeqd	%xmm4, %xmm0
	pmovmskb %xmm0, %rax
	lea	16(%rsi), %rsi

	test	%rax, %rax
	jnz	L(CopyFrom1To16Bytes)

	movaps	16(%rcx, %rsi), %xmm1
	movaps	%xmm4, (%rdx, %rsi)
	pcmpeqd	%xmm1, %xmm0
	pmovmskb %xmm0, %rax
	lea	16(%rsi), %rsi

	test	%rax, %rax
	jnz	L(CopyFrom1To16Bytes)

	movaps	16(%rcx, %rsi), %xmm2
	movaps	%xmm1, (%rdx, %rsi)
	pcmpeqd	%xmm2, %xmm0
	pmovmskb %xmm0, %rax
	lea	16(%rsi), %rsi

	test	%rax, %rax
	jnz	L(CopyFrom1To16Bytes)

	movaps	16(%rcx, %rsi), %xmm3
	movaps	%xmm2, (%rdx, %rsi)
	pcmpeqd	%xmm3, %xmm0
	pmovmskb %xmm0, %rax
	lea	16(%rsi), %rsi

	test	%rax, %rax
	jnz	L(CopyFrom1To16Bytes)

	movaps	%xmm3, (%rdx, %rsi)
	mov	%rcx, %rax
	lea	16(%rcx, %rsi), %rcx
	and	$-0x40, %rcx
	sub	%rcx, %rax
	sub	%rax, %rdx

	mov	$-0x40, %rsi

	.p2align 4
L(Aligned64Loop):
	movaps	(%rcx), %xmm2
	movaps	%xmm2, %xmm4
	movaps	16(%rcx), %xmm5
	movaps	32(%rcx), %xmm3
	movaps	%xmm3, %xmm6
	movaps	48(%rcx), %xmm7
	pminub	%xmm5, %xmm2
	pminub	%xmm7, %xmm3
	pminub	%xmm2, %xmm3
	pcmpeqd	%xmm0, %xmm3
	pmovmskb %xmm3, %rax
	lea	64(%rdx), %rdx
	lea	64(%rcx), %rcx
	test	%rax, %rax
	jnz	L(Aligned64Leave)
	movaps	%xmm4, -64(%rdx)
	movaps	%xmm5, -48(%rdx)
	movaps	%xmm6, -32(%rdx)
	movaps	%xmm7, -16(%rdx)
	jmp	L(Aligned64Loop)

L(Aligned64Leave):
	pcmpeqd	%xmm4, %xmm0
	pmovmskb %xmm0, %rax
	test	%rax, %rax
	jnz	L(CopyFrom1To16Bytes)

	pcmpeqd	%xmm5, %xmm0

	pmovmskb %xmm0, %rax
	movaps	%xmm4, -64(%rdx)
	test	%rax, %rax
	lea	16(%rsi), %rsi
	jnz	L(CopyFrom1To16Bytes)

	pcmpeqd	%xmm6, %xmm0

	pmovmskb %xmm0, %rax
	movaps	%xmm5, -48(%rdx)
	test	%rax, %rax
	lea	16(%rsi), %rsi
	jnz	L(CopyFrom1To16Bytes)

	movaps	%xmm6, -32(%rdx)
	pcmpeqd	%xmm7, %xmm0

	pmovmskb %xmm0, %rax
	lea	16(%rsi), %rsi
	test	%rax, %rax
	jnz	L(CopyFrom1To16Bytes)

	mov	$-0x40, %rsi
	movaps	%xmm7, -16(%rdx)
	jmp	L(Aligned64Loop)

	.p2align 4
L(Shl4):
	movaps	-4(%rcx), %xmm1
	movaps	12(%rcx), %xmm2
L(Shl4Start):
	pcmpeqd	%xmm2, %xmm0
	pmovmskb %xmm0, %rax
	movaps	%xmm2, %xmm3

	test	%rax, %rax
	jnz	L(Shl4LoopExit)

	palignr	$4, %xmm1, %xmm2
	movaps	%xmm2, (%rdx)
	movaps	28(%rcx), %xmm2

	pcmpeqd	%xmm2, %xmm0
	lea	16(%rdx), %rdx
	pmovmskb %xmm0, %rax
	lea	16(%rcx), %rcx
	movaps	%xmm2, %xmm1

	test	%rax, %rax
	jnz	L(Shl4LoopExit)

	palignr	$4, %xmm3, %xmm2
	movaps	%xmm2, (%rdx)
	movaps	28(%rcx), %xmm2

	pcmpeqd	%xmm2, %xmm0
	lea	16(%rdx), %rdx
	pmovmskb %xmm0, %rax
	lea	16(%rcx), %rcx
	movaps	%xmm2, %xmm3

	test	%rax, %rax
	jnz	L(Shl4LoopExit)

	palignr	$4, %xmm1, %xmm2
	movaps	%xmm2, (%rdx)
	movaps	28(%rcx), %xmm2

	pcmpeqd	%xmm2, %xmm0
	lea	16(%rdx), %rdx
	pmovmskb %xmm0, %rax
	lea	16(%rcx), %rcx

	test	%rax, %rax
	jnz	L(Shl4LoopExit)

	palignr	$4, %xmm3, %xmm2
	movaps	%xmm2, (%rdx)
	lea	28(%rcx), %rcx
	lea	16(%rdx), %rdx

	mov	%rcx, %rax
	and	$-0x40, %rcx
	sub	%rcx, %rax
	lea	-12(%rcx), %rcx
	sub	%rax, %rdx

	movaps	-4(%rcx), %xmm1

	.p2align 4
L(Shl4LoopStart):
	movaps	12(%rcx), %xmm2
	movaps	28(%rcx), %xmm3
	movaps	%xmm3, %xmm6
	movaps	44(%rcx), %xmm4
	movaps	%xmm4, %xmm7
	movaps	60(%rcx), %xmm5
	pminub	%xmm2, %xmm6
	pminub	%xmm5, %xmm7
	pminub	%xmm6, %xmm7
	pcmpeqd	%xmm0, %xmm7
	pmovmskb %xmm7, %rax
	movaps	%xmm5, %xmm7
	palignr	$4, %xmm4, %xmm5
	test	%rax, %rax
	palignr	$4, %xmm3, %xmm4
	jnz	L(Shl4Start)

	palignr	$4, %xmm2, %xmm3
	lea	64(%rcx), %rcx
	palignr	$4, %xmm1, %xmm2
	movaps	%xmm7, %xmm1
	movaps	%xmm5, 48(%rdx)
	movaps	%xmm4, 32(%rdx)
	movaps	%xmm3, 16(%rdx)
	movaps	%xmm2, (%rdx)
	lea	64(%rdx), %rdx
	jmp	L(Shl4LoopStart)

L(Shl4LoopExit):
	movdqu	-4(%rcx), %xmm1
	mov	$12, %rsi
	movdqu	%xmm1, -4(%rdx)
	jmp	L(CopyFrom1To16Bytes)

	.p2align 4
L(Shl8):
	movaps	-8(%rcx), %xmm1
	movaps	8(%rcx), %xmm2
L(Shl8Start):
	pcmpeqd	%xmm2, %xmm0
	pmovmskb %xmm0, %rax
	movaps	%xmm2, %xmm3

	test	%rax, %rax
	jnz	L(Shl8LoopExit)

	palignr	$8, %xmm1, %xmm2
	movaps	%xmm2, (%rdx)
	movaps	24(%rcx), %xmm2

	pcmpeqd	%xmm2, %xmm0
	lea	16(%rdx), %rdx
	pmovmskb %xmm0, %rax
	lea	16(%rcx), %rcx
	movaps	%xmm2, %xmm1

	test	%rax, %rax
	jnz	L(Shl8LoopExit)

	palignr	$8, %xmm3, %xmm2
	movaps	%xmm2, (%rdx)
	movaps	24(%rcx), %xmm2

	pcmpeqd	%xmm2, %xmm0
	lea	16(%rdx), %rdx
	pmovmskb %xmm0, %rax
	lea	16(%rcx), %rcx
	movaps	%xmm2, %xmm3

	test	%rax, %rax
	jnz	L(Shl8LoopExit)

	palignr	$8, %xmm1, %xmm2
	movaps	%xmm2, (%rdx)
	movaps	24(%rcx), %xmm2

	pcmpeqd	%xmm2, %xmm0
	lea	16(%rdx), %rdx
	pmovmskb %xmm0, %rax
	lea	16(%rcx), %rcx

	test	%rax, %rax
	jnz	L(Shl8LoopExit)

	palignr	$8, %xmm3, %xmm2
	movaps	%xmm2, (%rdx)
	lea	24(%rcx), %rcx
	lea	16(%rdx), %rdx

	mov	%rcx, %rax
	and	$-0x40, %rcx
	sub	%rcx, %rax
	lea	-8(%rcx), %rcx
	sub	%rax, %rdx

	movaps	-8(%rcx), %xmm1

	.p2align 4
L(Shl8LoopStart):
	movaps	8(%rcx), %xmm2
	movaps	24(%rcx), %xmm3
	movaps	%xmm3, %xmm6
	movaps	40(%rcx), %xmm4
	movaps	%xmm4, %xmm7
	movaps	56(%rcx), %xmm5
	pminub	%xmm2, %xmm6
	pminub	%xmm5, %xmm7
	pminub	%xmm6, %xmm7
	pcmpeqd	%xmm0, %xmm7
	pmovmskb %xmm7, %rax
	movaps	%xmm5, %xmm7
	palignr	$8, %xmm4, %xmm5
	test	%rax, %rax
	palignr	$8, %xmm3, %xmm4
	jnz	L(Shl8Start)

	palignr	$8, %xmm2, %xmm3
	lea	64(%rcx), %rcx
	palignr	$8, %xmm1, %xmm2
	movaps	%xmm7, %xmm1
	movaps	%xmm5, 48(%rdx)
	movaps	%xmm4, 32(%rdx)
	movaps	%xmm3, 16(%rdx)
	movaps	%xmm2, (%rdx)
	lea	64(%rdx), %rdx
	jmp	L(Shl8LoopStart)

L(Shl8LoopExit):
	mov	(%rcx), %r9
	mov	$8, %rsi
	mov	%r9, (%rdx)
	jmp	L(CopyFrom1To16Bytes)

	.p2align 4
L(Shl12):
	movaps	-12(%rcx), %xmm1
	movaps	4(%rcx), %xmm2
L(Shl12Start):
	pcmpeqd	%xmm2, %xmm0
	pmovmskb %xmm0, %rax
	movaps	%xmm2, %xmm3

	test	%rax, %rax
	jnz	L(Shl12LoopExit)

	palignr	$12, %xmm1, %xmm2
	movaps	%xmm2, (%rdx)
	movaps	20(%rcx), %xmm2

	pcmpeqd	%xmm2, %xmm0
	lea	16(%rdx), %rdx
	pmovmskb %xmm0, %rax
	lea	16(%rcx), %rcx
	movaps	%xmm2, %xmm1

	test	%rax, %rax
	jnz	L(Shl12LoopExit)

	palignr	$12, %xmm3, %xmm2
	movaps	%xmm2, (%rdx)
	movaps	20(%rcx), %xmm2

	pcmpeqd	%xmm2, %xmm0
	lea	16(%rdx), %rdx
	pmovmskb %xmm0, %rax
	lea	16(%rcx), %rcx
	movaps	%xmm2, %xmm3

	test	%rax, %rax
	jnz	L(Shl12LoopExit)

	palignr	$12, %xmm1, %xmm2
	movaps	%xmm2, (%rdx)
	movaps	20(%rcx), %xmm2

	pcmpeqd	%xmm2, %xmm0
	lea	16(%rdx), %rdx
	pmovmskb %xmm0, %rax
	lea	16(%rcx), %rcx

	test	%rax, %rax
	jnz	L(Shl12LoopExit)

	palignr	$12, %xmm3, %xmm2
	movaps	%xmm2, (%rdx)
	lea	20(%rcx), %rcx
	lea	16(%rdx), %rdx

	mov	%rcx, %rax
	and	$-0x40, %rcx
	sub	%rcx, %rax
	lea	-4(%rcx), %rcx
	sub	%rax, %rdx

	movaps	-12(%rcx), %xmm1

	.p2align 4
L(Shl12LoopStart):
	movaps	4(%rcx), %xmm2
	movaps	20(%rcx), %xmm3
	movaps	%xmm3, %xmm6
	movaps	36(%rcx), %xmm4
	movaps	%xmm4, %xmm7
	movaps	52(%rcx), %xmm5
	pminub	%xmm2, %xmm6
	pminub	%xmm5, %xmm7
	pminub	%xmm6, %xmm7
	pcmpeqd	%xmm0, %xmm7
	pmovmskb %xmm7, %rax
	movaps	%xmm5, %xmm7
	palignr	$12, %xmm4, %xmm5
	test	%rax, %rax
	palignr	$12, %xmm3, %xmm4
	jnz	L(Shl12Start)
	palignr	$12, %xmm2, %xmm3
	lea	64(%rcx), %rcx
	palignr	$12, %xmm1, %xmm2
	movaps	%xmm7, %xmm1
	movaps	%xmm5, 48(%rdx)
	movaps	%xmm4, 32(%rdx)
	movaps	%xmm3, 16(%rdx)
	movaps	%xmm2, (%rdx)
	lea	64(%rdx), %rdx
	jmp	L(Shl12LoopStart)

L(Shl12LoopExit):
	mov	(%rcx), %r9d
	mov	$4, %rsi
	mov	%r9d, (%rdx)
	jmp	L(CopyFrom1To16Bytes)

	.p2align 4
L(CopyFrom1To16Bytes):
	add	%rsi, %rdx
	add	%rsi, %rcx

	test	%al, %al
	jz	L(ExitHigh)
	test	$0x01, %al
	jnz	L(Exit4)

	mov	(%rcx), %rax
	mov	%rax, (%rdx)
	mov	%rdi, %rax
	ret

	.p2align 4
L(ExitHigh):
	test	$0x01, %ah
	jnz	L(Exit12)

	mov	(%rcx), %rax
	mov	%rax, (%rdx)
	mov	8(%rcx), %rax
	mov	%rax, 8(%rdx)
	mov	%rdi, %rax
	ret

	.p2align 4
L(Exit4):
	movl	(%rcx), %eax
	movl	%eax, (%rdx)
	mov	%rdi, %rax
	ret

	.p2align 4
L(Exit8):
	mov	(%rcx), %rax
	mov	%rax, (%rdx)
	mov	%rdi, %rax
	ret

	.p2align 4
L(Exit12):
	mov	(%rcx), %rax
	mov	%rax, (%rdx)
	mov	8(%rcx), %eax
	mov	%eax, 8(%rdx)
	mov	%rdi, %rax
	ret

	.p2align 4
L(Exit16):
	mov	(%rcx), %rax
	mov	%rax, (%rdx)
	mov	8(%rcx), %rax
	mov	%rax, 8(%rdx)
	mov	%rdi, %rax
	ret

END(__wcscpy_ssse3)
#endif
