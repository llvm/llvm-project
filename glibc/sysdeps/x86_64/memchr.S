/* Copyright (C) 2011-2021 Free Software Foundation, Inc.
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

#ifdef USE_AS_WMEMCHR
# define MEMCHR		wmemchr
# define PCMPEQ		pcmpeqd
# define CHAR_PER_VEC	4
#else
# define MEMCHR		memchr
# define PCMPEQ		pcmpeqb
# define CHAR_PER_VEC	16
#endif

/* fast SSE2 version with using pmaxub and 64 byte loop */

	.text
ENTRY(MEMCHR)
	movd	%esi, %xmm1
	mov	%edi, %ecx

#ifdef __ILP32__
	/* Clear the upper 32 bits.  */
	movl	%edx, %edx
#endif
#ifdef USE_AS_WMEMCHR
	test	%RDX_LP, %RDX_LP
	jz	L(return_null)
#else
	punpcklbw %xmm1, %xmm1
	test	%RDX_LP, %RDX_LP
	jz	L(return_null)
	punpcklbw %xmm1, %xmm1
#endif

	and	$63, %ecx
	pshufd	$0, %xmm1, %xmm1

	cmp	$48, %ecx
	ja	L(crosscache)

	movdqu	(%rdi), %xmm0
	PCMPEQ	%xmm1, %xmm0
	pmovmskb %xmm0, %eax
	test	%eax, %eax

	jnz	L(matches_1)
	sub	$CHAR_PER_VEC, %rdx
	jbe	L(return_null)
	add	$16, %rdi
	and	$15, %ecx
	and	$-16, %rdi
#ifdef USE_AS_WMEMCHR
	shr	$2, %ecx
#endif
	add	%rcx, %rdx
	sub	$(CHAR_PER_VEC * 4), %rdx
	jbe	L(exit_loop)
	jmp	L(loop_prolog)

	.p2align 4
L(crosscache):
	and	$15, %ecx
	and	$-16, %rdi
	movdqa	(%rdi), %xmm0

	PCMPEQ	%xmm1, %xmm0
	/* Check if there is a match.  */
	pmovmskb %xmm0, %eax
	/* Remove the leading bytes.  */
	sar	%cl, %eax
	test	%eax, %eax
	je	L(unaligned_no_match)
	/* Check which byte is a match.  */
	bsf	%eax, %eax
#ifdef USE_AS_WMEMCHR
	mov	%eax, %esi
	shr	$2, %esi
	sub	%rsi, %rdx
#else
	sub	%rax, %rdx
#endif
	jbe	L(return_null)
	add	%rdi, %rax
	add	%rcx, %rax
	ret

	.p2align 4
L(unaligned_no_match):
	/* "rcx" is less than 16.  Calculate "rdx + rcx - 16" by using
	   "rdx - (16 - rcx)" instead of "(rdx + rcx) - 16" to void
	   possible addition overflow.  */
	neg	%rcx
	add	$16, %rcx
#ifdef USE_AS_WMEMCHR
	shr	$2, %ecx
#endif
	sub	%rcx, %rdx
	jbe	L(return_null)
	add	$16, %rdi
	sub	$(CHAR_PER_VEC * 4), %rdx
	jbe	L(exit_loop)

	.p2align 4
L(loop_prolog):
	movdqa	(%rdi), %xmm0
	PCMPEQ	%xmm1, %xmm0
	pmovmskb %xmm0, %eax
	test	%eax, %eax
	jnz	L(matches)

	movdqa	16(%rdi), %xmm2
	PCMPEQ	%xmm1, %xmm2
	pmovmskb %xmm2, %eax
	test	%eax, %eax
	jnz	L(matches16)

	movdqa	32(%rdi), %xmm3
	PCMPEQ	%xmm1, %xmm3
	pmovmskb %xmm3, %eax
	test	%eax, %eax
	jnz	L(matches32)

	movdqa	48(%rdi), %xmm4
	PCMPEQ	%xmm1, %xmm4
	add	$64, %rdi
	pmovmskb %xmm4, %eax
	test	%eax, %eax
	jnz	L(matches0)

	test	$0x3f, %rdi
	jz	L(align64_loop)

	sub	$(CHAR_PER_VEC * 4), %rdx
	jbe	L(exit_loop)

	movdqa	(%rdi), %xmm0
	PCMPEQ	%xmm1, %xmm0
	pmovmskb %xmm0, %eax
	test	%eax, %eax
	jnz	L(matches)

	movdqa	16(%rdi), %xmm2
	PCMPEQ	%xmm1, %xmm2
	pmovmskb %xmm2, %eax
	test	%eax, %eax
	jnz	L(matches16)

	movdqa	32(%rdi), %xmm3
	PCMPEQ	%xmm1, %xmm3
	pmovmskb %xmm3, %eax
	test	%eax, %eax
	jnz	L(matches32)

	movdqa	48(%rdi), %xmm3
	PCMPEQ	%xmm1, %xmm3
	pmovmskb %xmm3, %eax

	add	$64, %rdi
	test	%eax, %eax
	jnz	L(matches0)

	mov	%rdi, %rcx
	and	$-64, %rdi
	and	$63, %ecx
#ifdef USE_AS_WMEMCHR
	shr	$2, %ecx
#endif
	add	%rcx, %rdx

	.p2align 4
L(align64_loop):
	sub	$(CHAR_PER_VEC * 4), %rdx
	jbe	L(exit_loop)
	movdqa	(%rdi), %xmm0
	movdqa	16(%rdi), %xmm2
	movdqa	32(%rdi), %xmm3
	movdqa	48(%rdi), %xmm4

	PCMPEQ	%xmm1, %xmm0
	PCMPEQ	%xmm1, %xmm2
	PCMPEQ	%xmm1, %xmm3
	PCMPEQ	%xmm1, %xmm4

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
	PCMPEQ	%xmm1, %xmm3

	PCMPEQ	48(%rdi), %xmm1
	pmovmskb %xmm3, %eax
	test	%eax, %eax
	jnz	L(matches32)

	pmovmskb %xmm1, %eax
	bsf	%eax, %eax
	lea	48(%rdi, %rax), %rax
	ret

	.p2align 4
L(exit_loop):
	add	$(CHAR_PER_VEC * 2), %edx
	jle	L(exit_loop_32)

	movdqa	(%rdi), %xmm0
	PCMPEQ	%xmm1, %xmm0
	pmovmskb %xmm0, %eax
	test	%eax, %eax
	jnz	L(matches)

	movdqa	16(%rdi), %xmm2
	PCMPEQ	%xmm1, %xmm2
	pmovmskb %xmm2, %eax
	test	%eax, %eax
	jnz	L(matches16)

	movdqa	32(%rdi), %xmm3
	PCMPEQ	%xmm1, %xmm3
	pmovmskb %xmm3, %eax
	test	%eax, %eax
	jnz	L(matches32_1)
	sub	$CHAR_PER_VEC, %edx
	jle	L(return_null)

	PCMPEQ	48(%rdi), %xmm1
	pmovmskb %xmm1, %eax
	test	%eax, %eax
	jnz	L(matches48_1)
	xor	%eax, %eax
	ret

	.p2align 4
L(exit_loop_32):
	add	$(CHAR_PER_VEC * 2), %edx
	movdqa	(%rdi), %xmm0
	PCMPEQ	%xmm1, %xmm0
	pmovmskb %xmm0, %eax
	test	%eax, %eax
	jnz	L(matches_1)
	sub	$CHAR_PER_VEC, %edx
	jbe	L(return_null)

	PCMPEQ	16(%rdi), %xmm1
	pmovmskb %xmm1, %eax
	test	%eax, %eax
	jnz	L(matches16_1)
	xor	%eax, %eax
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

	.p2align 4
L(matches_1):
	bsf	%eax, %eax
#ifdef USE_AS_WMEMCHR
	mov	%eax, %esi
	shr	$2, %esi
	sub	%rsi, %rdx
#else
	sub	%rax, %rdx
#endif
	jbe	L(return_null)
	add	%rdi, %rax
	ret

	.p2align 4
L(matches16_1):
	bsf	%eax, %eax
#ifdef USE_AS_WMEMCHR
	mov	%eax, %esi
	shr	$2, %esi
	sub	%rsi, %rdx
#else
	sub	%rax, %rdx
#endif
	jbe	L(return_null)
	lea	16(%rdi, %rax), %rax
	ret

	.p2align 4
L(matches32_1):
	bsf	%eax, %eax
#ifdef USE_AS_WMEMCHR
	mov	%eax, %esi
	shr	$2, %esi
	sub	%rsi, %rdx
#else
	sub	%rax, %rdx
#endif
	jbe	L(return_null)
	lea	32(%rdi, %rax), %rax
	ret

	.p2align 4
L(matches48_1):
	bsf	%eax, %eax
#ifdef USE_AS_WMEMCHR
	mov	%eax, %esi
	shr	$2, %esi
	sub	%rsi, %rdx
#else
	sub	%rax, %rdx
#endif
	jbe	L(return_null)
	lea	48(%rdi, %rax), %rax
	ret

	.p2align 4
L(return_null):
	xor	%eax, %eax
	ret
END(MEMCHR)

#ifndef USE_AS_WMEMCHR
strong_alias (memchr, __memchr)
libc_hidden_builtin_def(memchr)
#endif
