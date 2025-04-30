/* strcat with SSE2
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

# ifndef STRCAT
#  define STRCAT  __strcat_sse2_unaligned
# endif

# define USE_AS_STRCAT

.text
ENTRY (STRCAT)
	mov	%rdi, %r9
# ifdef USE_AS_STRNCAT
	mov	%rdx, %r8
# endif

/* Inline corresponding strlen file, temporary until new strcpy
   implementation gets merged.  */

	xor	%rax, %rax
	mov	%edi, %ecx
	and	$0x3f, %ecx
	pxor	%xmm0, %xmm0
	cmp	$0x30, %ecx
	ja	L(next)
	movdqu	(%rdi), %xmm1
	pcmpeqb	%xmm1, %xmm0
	pmovmskb %xmm0, %edx
	test	%edx, %edx
	jnz	L(exit_less16)
	mov	%rdi, %rax
	and	$-16, %rax
	jmp	L(align16_start)
L(next):
	mov	%rdi, %rax
	and	$-16, %rax
	pcmpeqb	(%rax), %xmm0
	mov	$-1, %r10d
	sub	%rax, %rcx
	shl	%cl, %r10d
	pmovmskb %xmm0, %edx
	and	%r10d, %edx
	jnz	L(exit)

L(align16_start):
	pxor	%xmm0, %xmm0
	pxor	%xmm1, %xmm1
	pxor	%xmm2, %xmm2
	pxor	%xmm3, %xmm3
	pcmpeqb	16(%rax), %xmm0
	pmovmskb %xmm0, %edx
	test	%edx, %edx
	jnz	L(exit16)

	pcmpeqb	32(%rax), %xmm1
	pmovmskb %xmm1, %edx
	test	%edx, %edx
	jnz	L(exit32)

	pcmpeqb	48(%rax), %xmm2
	pmovmskb %xmm2, %edx
	test	%edx, %edx
	jnz	L(exit48)

	pcmpeqb	64(%rax), %xmm3
	pmovmskb %xmm3, %edx
	test	%edx, %edx
	jnz	L(exit64)

	pcmpeqb	80(%rax), %xmm0
	add	$64, %rax
	pmovmskb %xmm0, %edx
	test	%edx, %edx
	jnz	L(exit16)

	pcmpeqb	32(%rax), %xmm1
	pmovmskb %xmm1, %edx
	test	%edx, %edx
	jnz	L(exit32)

	pcmpeqb	48(%rax), %xmm2
	pmovmskb %xmm2, %edx
	test	%edx, %edx
	jnz	L(exit48)

	pcmpeqb	64(%rax), %xmm3
	pmovmskb %xmm3, %edx
	test	%edx, %edx
	jnz	L(exit64)

	pcmpeqb	80(%rax), %xmm0
	add	$64, %rax
	pmovmskb %xmm0, %edx
	test	%edx, %edx
	jnz	L(exit16)

	pcmpeqb	32(%rax), %xmm1
	pmovmskb %xmm1, %edx
	test	%edx, %edx
	jnz	L(exit32)

	pcmpeqb	48(%rax), %xmm2
	pmovmskb %xmm2, %edx
	test	%edx, %edx
	jnz	L(exit48)

	pcmpeqb	64(%rax), %xmm3
	pmovmskb %xmm3, %edx
	test	%edx, %edx
	jnz	L(exit64)

	pcmpeqb	80(%rax), %xmm0
	add	$64, %rax
	pmovmskb %xmm0, %edx
	test	%edx, %edx
	jnz	L(exit16)

	pcmpeqb	32(%rax), %xmm1
	pmovmskb %xmm1, %edx
	test	%edx, %edx
	jnz	L(exit32)

	pcmpeqb	48(%rax), %xmm2
	pmovmskb %xmm2, %edx
	test	%edx, %edx
	jnz	L(exit48)

	pcmpeqb	64(%rax), %xmm3
	pmovmskb %xmm3, %edx
	test	%edx, %edx
	jnz	L(exit64)

	test	$0x3f, %rax
	jz	L(align64_loop)

	pcmpeqb	80(%rax), %xmm0
	add	$80, %rax
	pmovmskb %xmm0, %edx
	test	%edx, %edx
	jnz	L(exit)

	test	$0x3f, %rax
	jz	L(align64_loop)

	pcmpeqb	16(%rax), %xmm1
	add	$16, %rax
	pmovmskb %xmm1, %edx
	test	%edx, %edx
	jnz	L(exit)

	test	$0x3f, %rax
	jz	L(align64_loop)

	pcmpeqb	16(%rax), %xmm2
	add	$16, %rax
	pmovmskb %xmm2, %edx
	test	%edx, %edx
	jnz	L(exit)

	test	$0x3f, %rax
	jz	L(align64_loop)

	pcmpeqb	16(%rax), %xmm3
	add	$16, %rax
	pmovmskb %xmm3, %edx
	test	%edx, %edx
	jnz	L(exit)

	add	$16, %rax
	.p2align 4
	L(align64_loop):
	movaps	(%rax),	%xmm4
	pminub	16(%rax),	%xmm4
	movaps	32(%rax),	%xmm5
	pminub	48(%rax),	%xmm5
	add	$64,	%rax
	pminub	%xmm4,	%xmm5
	pcmpeqb	%xmm0,	%xmm5
	pmovmskb %xmm5,	%edx
	test	%edx,	%edx
	jz	L(align64_loop)

	pcmpeqb	-64(%rax), %xmm0
	sub	$80,	%rax
	pmovmskb %xmm0, %edx
	test	%edx, %edx
	jnz	L(exit16)

	pcmpeqb	32(%rax), %xmm1
	pmovmskb %xmm1, %edx
	test	%edx, %edx
	jnz	L(exit32)

	pcmpeqb	48(%rax), %xmm2
	pmovmskb %xmm2, %edx
	test	%edx, %edx
	jnz	L(exit48)

	pcmpeqb	64(%rax), %xmm3
	pmovmskb %xmm3, %edx
	sub	%rdi, %rax
	bsf	%rdx, %rdx
	add	%rdx, %rax
	add	$64, %rax
	jmp	L(StartStrcpyPart)

	.p2align 4
L(exit):
	sub	%rdi, %rax
L(exit_less16):
	bsf	%rdx, %rdx
	add	%rdx, %rax
	jmp	L(StartStrcpyPart)

	.p2align 4
L(exit16):
	sub	%rdi, %rax
	bsf	%rdx, %rdx
	add	%rdx, %rax
	add	$16, %rax
	jmp	L(StartStrcpyPart)

	.p2align 4
L(exit32):
	sub	%rdi, %rax
	bsf	%rdx, %rdx
	add	%rdx, %rax
	add	$32, %rax
	jmp	L(StartStrcpyPart)

	.p2align 4
L(exit48):
	sub	%rdi, %rax
	bsf	%rdx, %rdx
	add	%rdx, %rax
	add	$48, %rax
	jmp	L(StartStrcpyPart)

	.p2align 4
L(exit64):
	sub	%rdi, %rax
	bsf	%rdx, %rdx
	add	%rdx, %rax
	add	$64, %rax

	.p2align 4
L(StartStrcpyPart):
	lea	(%r9, %rax), %rdi
	mov	%rsi, %rcx
	mov	%r9, %rax      /* save result */

# ifdef USE_AS_STRNCAT
	test	%r8, %r8
	jz	L(ExitZero)
#  define USE_AS_STRNCPY
# endif

# include "strcpy-sse2-unaligned.S"
#endif
