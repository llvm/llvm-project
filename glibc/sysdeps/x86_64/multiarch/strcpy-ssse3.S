/* strcpy with SSSE3
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

# ifndef USE_AS_STRCAT
#  include <sysdep.h>

#  ifndef STRCPY
#   define STRCPY  __strcpy_ssse3
#  endif

	.section .text.ssse3,"ax",@progbits
ENTRY (STRCPY)

	mov	%rsi, %rcx
#  ifdef USE_AS_STRNCPY
	mov	%RDX_LP, %R8_LP
#  endif
	mov	%rdi, %rdx
#  ifdef USE_AS_STRNCPY
	test	%R8_LP, %R8_LP
	jz	L(Exit0)
	cmp	$8, %R8_LP
	jbe	L(StrncpyExit8Bytes)
# endif
	cmpb	$0, (%rcx)
	jz	L(Exit1)
	cmpb	$0, 1(%rcx)
	jz	L(Exit2)
	cmpb	$0, 2(%rcx)
	jz	L(Exit3)
	cmpb	$0, 3(%rcx)
	jz	L(Exit4)
	cmpb	$0, 4(%rcx)
	jz	L(Exit5)
	cmpb	$0, 5(%rcx)
	jz	L(Exit6)
	cmpb	$0, 6(%rcx)
	jz	L(Exit7)
	cmpb	$0, 7(%rcx)
	jz	L(Exit8)
# ifdef USE_AS_STRNCPY
	cmp	$16, %r8
	jb	L(StrncpyExit15Bytes)
# endif
	cmpb	$0, 8(%rcx)
	jz	L(Exit9)
	cmpb	$0, 9(%rcx)
	jz	L(Exit10)
	cmpb	$0, 10(%rcx)
	jz	L(Exit11)
	cmpb	$0, 11(%rcx)
	jz	L(Exit12)
	cmpb	$0, 12(%rcx)
	jz	L(Exit13)
	cmpb	$0, 13(%rcx)
	jz	L(Exit14)
	cmpb	$0, 14(%rcx)
	jz	L(Exit15)
# ifdef USE_AS_STRNCPY
	cmp	$16, %r8
	je	L(Exit16)
# endif
	cmpb	$0, 15(%rcx)
	jz	L(Exit16)
# endif

# ifdef USE_AS_STRNCPY
	mov	%rcx, %rsi
	sub	$16, %r8
	and	$0xf, %rsi

/* add 16 bytes rcx_offset to r8 */

	add	%rsi, %r8
# endif
	lea	16(%rcx), %rsi
	and	$-16, %rsi
	pxor	%xmm0, %xmm0
	mov	(%rcx), %r9
	mov	%r9, (%rdx)
	pcmpeqb	(%rsi), %xmm0
	mov	8(%rcx), %r9
	mov	%r9, 8(%rdx)

/* convert byte mask in xmm0 to bit mask */

	pmovmskb %xmm0, %rax
	sub	%rcx, %rsi

# ifdef USE_AS_STRNCPY
	sub	$16, %r8
	jbe	L(CopyFrom1To16BytesCase2OrCase3)
# endif
	test	%rax, %rax
	jnz	L(CopyFrom1To16Bytes)

	mov	%rdx, %rax
	lea	16(%rdx), %rdx
	and	$-16, %rdx
	sub	%rdx, %rax

# ifdef USE_AS_STRNCPY
	add	%rax, %rsi
	lea	-1(%rsi), %rsi
	and	$1<<31, %esi
	test	%rsi, %rsi
	jnz	L(ContinueCopy)
	lea	16(%r8), %r8

L(ContinueCopy):
# endif
	sub	%rax, %rcx
	mov	%rcx, %rax
	and	$0xf, %rax
	mov	$0, %rsi

/* case: rcx_offset == rdx_offset */

	jz	L(Align16Both)

	cmp	$8, %rax
	jae	L(ShlHigh8)
	cmp	$1, %rax
	je	L(Shl1)
	cmp	$2, %rax
	je	L(Shl2)
	cmp	$3, %rax
	je	L(Shl3)
	cmp	$4, %rax
	je	L(Shl4)
	cmp	$5, %rax
	je	L(Shl5)
	cmp	$6, %rax
	je	L(Shl6)
	jmp	L(Shl7)

L(ShlHigh8):
	je	L(Shl8)
	cmp	$9, %rax
	je	L(Shl9)
	cmp	$10, %rax
	je	L(Shl10)
	cmp	$11, %rax
	je	L(Shl11)
	cmp	$12, %rax
	je	L(Shl12)
	cmp	$13, %rax
	je	L(Shl13)
	cmp	$14, %rax
	je	L(Shl14)
	jmp	L(Shl15)

L(Align16Both):
	movaps	(%rcx), %xmm1
	movaps	16(%rcx), %xmm2
	movaps	%xmm1, (%rdx)
	pcmpeqb	%xmm2, %xmm0
	pmovmskb %xmm0, %rax
	lea	16(%rsi), %rsi
# ifdef USE_AS_STRNCPY
	sub	$16, %r8
	jbe	L(CopyFrom1To16BytesCase2OrCase3)
# endif
	test	%rax, %rax
	jnz	L(CopyFrom1To16Bytes)

	movaps	16(%rcx, %rsi), %xmm3
	movaps	%xmm2, (%rdx, %rsi)
	pcmpeqb	%xmm3, %xmm0
	pmovmskb %xmm0, %rax
	lea	16(%rsi), %rsi
# ifdef USE_AS_STRNCPY
	sub	$16, %r8
	jbe	L(CopyFrom1To16BytesCase2OrCase3)
# endif
	test	%rax, %rax
	jnz	L(CopyFrom1To16Bytes)

	movaps	16(%rcx, %rsi), %xmm4
	movaps	%xmm3, (%rdx, %rsi)
	pcmpeqb	%xmm4, %xmm0
	pmovmskb %xmm0, %rax
	lea	16(%rsi), %rsi
# ifdef USE_AS_STRNCPY
	sub	$16, %r8
	jbe	L(CopyFrom1To16BytesCase2OrCase3)
# endif
	test	%rax, %rax
	jnz	L(CopyFrom1To16Bytes)

	movaps	16(%rcx, %rsi), %xmm1
	movaps	%xmm4, (%rdx, %rsi)
	pcmpeqb	%xmm1, %xmm0
	pmovmskb %xmm0, %rax
	lea	16(%rsi), %rsi
# ifdef USE_AS_STRNCPY
	sub	$16, %r8
	jbe	L(CopyFrom1To16BytesCase2OrCase3)
# endif
	test	%rax, %rax
	jnz	L(CopyFrom1To16Bytes)

	movaps	16(%rcx, %rsi), %xmm2
	movaps	%xmm1, (%rdx, %rsi)
	pcmpeqb	%xmm2, %xmm0
	pmovmskb %xmm0, %rax
	lea	16(%rsi), %rsi
# ifdef USE_AS_STRNCPY
	sub	$16, %r8
	jbe	L(CopyFrom1To16BytesCase2OrCase3)
# endif
	test	%rax, %rax
	jnz	L(CopyFrom1To16Bytes)

	movaps	16(%rcx, %rsi), %xmm3
	movaps	%xmm2, (%rdx, %rsi)
	pcmpeqb	%xmm3, %xmm0
	pmovmskb %xmm0, %rax
	lea	16(%rsi), %rsi
# ifdef USE_AS_STRNCPY
	sub	$16, %r8
	jbe	L(CopyFrom1To16BytesCase2OrCase3)
# endif
	test	%rax, %rax
	jnz	L(CopyFrom1To16Bytes)

	movaps	%xmm3, (%rdx, %rsi)
	mov	%rcx, %rax
	lea	16(%rcx, %rsi), %rcx
	and	$-0x40, %rcx
	sub	%rcx, %rax
	sub	%rax, %rdx
# ifdef USE_AS_STRNCPY
	lea	112(%r8, %rax), %r8
# endif
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
	pcmpeqb	%xmm0, %xmm3
	pmovmskb %xmm3, %rax
	lea	64(%rdx), %rdx
	lea	64(%rcx), %rcx
# ifdef USE_AS_STRNCPY
	sub	$64, %r8
	jbe	L(StrncpyLeaveCase2OrCase3)
# endif
	test	%rax, %rax
	jnz	L(Aligned64Leave)
	movaps	%xmm4, -64(%rdx)
	movaps	%xmm5, -48(%rdx)
	movaps	%xmm6, -32(%rdx)
	movaps	%xmm7, -16(%rdx)
	jmp	L(Aligned64Loop)

L(Aligned64Leave):
# ifdef USE_AS_STRNCPY
	lea	48(%r8), %r8
# endif
	pcmpeqb	%xmm4, %xmm0
	pmovmskb %xmm0, %rax
	test	%rax, %rax
	jnz	L(CopyFrom1To16Bytes)

	pcmpeqb	%xmm5, %xmm0
# ifdef USE_AS_STRNCPY
	lea	-16(%r8), %r8
# endif
	pmovmskb %xmm0, %rax
	movaps	%xmm4, -64(%rdx)
	test	%rax, %rax
	lea	16(%rsi), %rsi
	jnz	L(CopyFrom1To16Bytes)

	pcmpeqb	%xmm6, %xmm0
# ifdef USE_AS_STRNCPY
	lea	-16(%r8), %r8
# endif
	pmovmskb %xmm0, %rax
	movaps	%xmm5, -48(%rdx)
	test	%rax, %rax
	lea	16(%rsi), %rsi
	jnz	L(CopyFrom1To16Bytes)

	movaps	%xmm6, -32(%rdx)
	pcmpeqb	%xmm7, %xmm0
# ifdef USE_AS_STRNCPY
	lea	-16(%r8), %r8
# endif
	pmovmskb %xmm0, %rax
	lea	16(%rsi), %rsi
	jmp	L(CopyFrom1To16Bytes)

	.p2align 4
L(Shl1):
	movaps	-1(%rcx), %xmm1
	movaps	15(%rcx), %xmm2
L(Shl1Start):
	pcmpeqb	%xmm2, %xmm0
	pmovmskb %xmm0, %rax
	movaps	%xmm2, %xmm3
# ifdef USE_AS_STRNCPY
	sub	$16, %r8
	jbe	L(StrncpyExit1Case2OrCase3)
# endif
	test	%rax, %rax
	jnz	L(Shl1LoopExit)

	palignr	$1, %xmm1, %xmm2
	movaps	%xmm2, (%rdx)
	movaps	31(%rcx), %xmm2

	pcmpeqb	%xmm2, %xmm0
	lea	16(%rdx), %rdx
	pmovmskb %xmm0, %rax
	lea	16(%rcx), %rcx
	movaps	%xmm2, %xmm1
# ifdef USE_AS_STRNCPY
	sub	$16, %r8
	jbe	L(StrncpyExit1Case2OrCase3)
# endif
	test	%rax, %rax
	jnz	L(Shl1LoopExit)

	palignr	$1, %xmm3, %xmm2
	movaps	%xmm2, (%rdx)
	movaps	31(%rcx), %xmm2

	pcmpeqb	%xmm2, %xmm0
	lea	16(%rdx), %rdx
	pmovmskb %xmm0, %rax
	lea	16(%rcx), %rcx
	movaps	%xmm2, %xmm3
# ifdef USE_AS_STRNCPY
	sub	$16, %r8
	jbe	L(StrncpyExit1Case2OrCase3)
# endif
	test	%rax, %rax
	jnz	L(Shl1LoopExit)

	palignr	$1, %xmm1, %xmm2
	movaps	%xmm2, (%rdx)
	movaps	31(%rcx), %xmm2

	pcmpeqb	%xmm2, %xmm0
	lea	16(%rdx), %rdx
	pmovmskb %xmm0, %rax
	lea	16(%rcx), %rcx
# ifdef USE_AS_STRNCPY
	sub	$16, %r8
	jbe	L(StrncpyExit1Case2OrCase3)
# endif
	test	%rax, %rax
	jnz	L(Shl1LoopExit)

	palignr	$1, %xmm3, %xmm2
	movaps	%xmm2, (%rdx)
	lea	31(%rcx), %rcx
	lea	16(%rdx), %rdx

	mov	%rcx, %rax
	and	$-0x40, %rcx
	sub	%rcx, %rax
	lea	-15(%rcx), %rcx
	sub	%rax, %rdx
# ifdef USE_AS_STRNCPY
	add	%rax, %r8
# endif
	movaps	-1(%rcx), %xmm1

/* 64 bytes loop */
	.p2align 4
L(Shl1LoopStart):
	movaps	15(%rcx), %xmm2
	movaps	31(%rcx), %xmm3
	movaps	%xmm3, %xmm6
	movaps	47(%rcx), %xmm4
	movaps	%xmm4, %xmm7
	movaps	63(%rcx), %xmm5
	pminub	%xmm2, %xmm6
	pminub	%xmm5, %xmm7
	pminub	%xmm6, %xmm7
	pcmpeqb	%xmm0, %xmm7
	pmovmskb %xmm7, %rax
	movaps	%xmm5, %xmm7
	palignr	$1, %xmm4, %xmm5
	test	%rax, %rax
	palignr	$1, %xmm3, %xmm4
	jnz	L(Shl1Start)
# ifdef USE_AS_STRNCPY
	sub	$64, %r8
	jbe	L(StrncpyLeave1)
# endif
	palignr	$1, %xmm2, %xmm3
	lea	64(%rcx), %rcx
	palignr	$1, %xmm1, %xmm2
	movaps	%xmm7, %xmm1
	movaps	%xmm5, 48(%rdx)
	movaps	%xmm4, 32(%rdx)
	movaps	%xmm3, 16(%rdx)
	movaps	%xmm2, (%rdx)
	lea	64(%rdx), %rdx
	jmp	L(Shl1LoopStart)

L(Shl1LoopExit):
	movdqu	-1(%rcx), %xmm1
	mov	$15, %rsi
	movdqu	%xmm1, -1(%rdx)
	jmp	L(CopyFrom1To16Bytes)

	.p2align 4
L(Shl2):
	movaps	-2(%rcx), %xmm1
	movaps	14(%rcx), %xmm2
L(Shl2Start):
	pcmpeqb	%xmm2, %xmm0
	pmovmskb %xmm0, %rax
	movaps	%xmm2, %xmm3
# ifdef USE_AS_STRNCPY
	sub	$16, %r8
	jbe	L(StrncpyExit2Case2OrCase3)
# endif
	test	%rax, %rax
	jnz	L(Shl2LoopExit)

	palignr	$2, %xmm1, %xmm2
	movaps	%xmm2, (%rdx)
	movaps	30(%rcx), %xmm2

	pcmpeqb	%xmm2, %xmm0
	lea	16(%rdx), %rdx
	pmovmskb %xmm0, %rax
	lea	16(%rcx), %rcx
	movaps	%xmm2, %xmm1
# ifdef USE_AS_STRNCPY
	sub	$16, %r8
	jbe	L(StrncpyExit2Case2OrCase3)
# endif
	test	%rax, %rax
	jnz	L(Shl2LoopExit)

	palignr	$2, %xmm3, %xmm2
	movaps	%xmm2, (%rdx)
	movaps	30(%rcx), %xmm2

	pcmpeqb	%xmm2, %xmm0
	lea	16(%rdx), %rdx
	pmovmskb %xmm0, %rax
	lea	16(%rcx), %rcx
	movaps	%xmm2, %xmm3
# ifdef USE_AS_STRNCPY
	sub	$16, %r8
	jbe	L(StrncpyExit2Case2OrCase3)
# endif
	test	%rax, %rax
	jnz	L(Shl2LoopExit)

	palignr	$2, %xmm1, %xmm2
	movaps	%xmm2, (%rdx)
	movaps	30(%rcx), %xmm2

	pcmpeqb	%xmm2, %xmm0
	lea	16(%rdx), %rdx
	pmovmskb %xmm0, %rax
	lea	16(%rcx), %rcx
# ifdef USE_AS_STRNCPY
	sub	$16, %r8
	jbe	L(StrncpyExit2Case2OrCase3)
# endif
	test	%rax, %rax
	jnz	L(Shl2LoopExit)

	palignr	$2, %xmm3, %xmm2
	movaps	%xmm2, (%rdx)
	lea	30(%rcx), %rcx
	lea	16(%rdx), %rdx

	mov	%rcx, %rax
	and	$-0x40, %rcx
	sub	%rcx, %rax
	lea	-14(%rcx), %rcx
	sub	%rax, %rdx
# ifdef USE_AS_STRNCPY
	add	%rax, %r8
# endif
	movaps	-2(%rcx), %xmm1

/* 64 bytes loop */
	.p2align 4
L(Shl2LoopStart):
	movaps	14(%rcx), %xmm2
	movaps	30(%rcx), %xmm3
	movaps	%xmm3, %xmm6
	movaps	46(%rcx), %xmm4
	movaps	%xmm4, %xmm7
	movaps	62(%rcx), %xmm5
	pminub	%xmm2, %xmm6
	pminub	%xmm5, %xmm7
	pminub	%xmm6, %xmm7
	pcmpeqb	%xmm0, %xmm7
	pmovmskb %xmm7, %rax
	movaps	%xmm5, %xmm7
	palignr	$2, %xmm4, %xmm5
	test	%rax, %rax
	palignr	$2, %xmm3, %xmm4
	jnz	L(Shl2Start)
# ifdef USE_AS_STRNCPY
	sub	$64, %r8
	jbe	L(StrncpyLeave2)
# endif
	palignr	$2, %xmm2, %xmm3
	lea	64(%rcx), %rcx
	palignr	$2, %xmm1, %xmm2
	movaps	%xmm7, %xmm1
	movaps	%xmm5, 48(%rdx)
	movaps	%xmm4, 32(%rdx)
	movaps	%xmm3, 16(%rdx)
	movaps	%xmm2, (%rdx)
	lea	64(%rdx), %rdx
	jmp	L(Shl2LoopStart)

L(Shl2LoopExit):
	movdqu	-2(%rcx), %xmm1
	mov	$14, %rsi
	movdqu	%xmm1, -2(%rdx)
	jmp	L(CopyFrom1To16Bytes)

	.p2align 4
L(Shl3):
	movaps	-3(%rcx), %xmm1
	movaps	13(%rcx), %xmm2
L(Shl3Start):
	pcmpeqb	%xmm2, %xmm0
	pmovmskb %xmm0, %rax
	movaps	%xmm2, %xmm3
# ifdef USE_AS_STRNCPY
	sub	$16, %r8
	jbe	L(StrncpyExit3Case2OrCase3)
# endif
	test	%rax, %rax
	jnz	L(Shl3LoopExit)

	palignr	$3, %xmm1, %xmm2
	movaps	%xmm2, (%rdx)
	movaps	29(%rcx), %xmm2

	pcmpeqb	%xmm2, %xmm0
	lea	16(%rdx), %rdx
	pmovmskb %xmm0, %rax
	lea	16(%rcx), %rcx
	movaps	%xmm2, %xmm1
# ifdef USE_AS_STRNCPY
	sub	$16, %r8
	jbe	L(StrncpyExit3Case2OrCase3)
# endif
	test	%rax, %rax
	jnz	L(Shl3LoopExit)

	palignr	$3, %xmm3, %xmm2
	movaps	%xmm2, (%rdx)
	movaps	29(%rcx), %xmm2

	pcmpeqb	%xmm2, %xmm0
	lea	16(%rdx), %rdx
	pmovmskb %xmm0, %rax
	lea	16(%rcx), %rcx
	movaps	%xmm2, %xmm3
# ifdef USE_AS_STRNCPY
	sub	$16, %r8
	jbe	L(StrncpyExit3Case2OrCase3)
# endif
	test	%rax, %rax
	jnz	L(Shl3LoopExit)

	palignr	$3, %xmm1, %xmm2
	movaps	%xmm2, (%rdx)
	movaps	29(%rcx), %xmm2

	pcmpeqb	%xmm2, %xmm0
	lea	16(%rdx), %rdx
	pmovmskb %xmm0, %rax
	lea	16(%rcx), %rcx
# ifdef USE_AS_STRNCPY
	sub	$16, %r8
	jbe	L(StrncpyExit3Case2OrCase3)
# endif
	test	%rax, %rax
	jnz	L(Shl3LoopExit)

	palignr	$3, %xmm3, %xmm2
	movaps	%xmm2, (%rdx)
	lea	29(%rcx), %rcx
	lea	16(%rdx), %rdx

	mov	%rcx, %rax
	and	$-0x40, %rcx
	sub	%rcx, %rax
	lea	-13(%rcx), %rcx
	sub	%rax, %rdx
# ifdef USE_AS_STRNCPY
	add	%rax, %r8
# endif
	movaps	-3(%rcx), %xmm1

/* 64 bytes loop */
	.p2align 4
L(Shl3LoopStart):
	movaps	13(%rcx), %xmm2
	movaps	29(%rcx), %xmm3
	movaps	%xmm3, %xmm6
	movaps	45(%rcx), %xmm4
	movaps	%xmm4, %xmm7
	movaps	61(%rcx), %xmm5
	pminub	%xmm2, %xmm6
	pminub	%xmm5, %xmm7
	pminub	%xmm6, %xmm7
	pcmpeqb	%xmm0, %xmm7
	pmovmskb %xmm7, %rax
	movaps	%xmm5, %xmm7
	palignr	$3, %xmm4, %xmm5
	test	%rax, %rax
	palignr	$3, %xmm3, %xmm4
	jnz	L(Shl3Start)
# ifdef USE_AS_STRNCPY
	sub	$64, %r8
	jbe	L(StrncpyLeave3)
# endif
	palignr	$3, %xmm2, %xmm3
	lea	64(%rcx), %rcx
	palignr	$3, %xmm1, %xmm2
	movaps	%xmm7, %xmm1
	movaps	%xmm5, 48(%rdx)
	movaps	%xmm4, 32(%rdx)
	movaps	%xmm3, 16(%rdx)
	movaps	%xmm2, (%rdx)
	lea	64(%rdx), %rdx
	jmp	L(Shl3LoopStart)

L(Shl3LoopExit):
	movdqu	-3(%rcx), %xmm1
	mov	$13, %rsi
	movdqu	%xmm1, -3(%rdx)
	jmp	L(CopyFrom1To16Bytes)

	.p2align 4
L(Shl4):
	movaps	-4(%rcx), %xmm1
	movaps	12(%rcx), %xmm2
L(Shl4Start):
	pcmpeqb	%xmm2, %xmm0
	pmovmskb %xmm0, %rax
	movaps	%xmm2, %xmm3
# ifdef USE_AS_STRNCPY
	sub	$16, %r8
	jbe	L(StrncpyExit4Case2OrCase3)
# endif
	test	%rax, %rax
	jnz	L(Shl4LoopExit)

	palignr	$4, %xmm1, %xmm2
	movaps	%xmm2, (%rdx)
	movaps	28(%rcx), %xmm2

	pcmpeqb	%xmm2, %xmm0
	lea	16(%rdx), %rdx
	pmovmskb %xmm0, %rax
	lea	16(%rcx), %rcx
	movaps	%xmm2, %xmm1
# ifdef USE_AS_STRNCPY
	sub	$16, %r8
	jbe	L(StrncpyExit4Case2OrCase3)
# endif
	test	%rax, %rax
	jnz	L(Shl4LoopExit)

	palignr	$4, %xmm3, %xmm2
	movaps	%xmm2, (%rdx)
	movaps	28(%rcx), %xmm2

	pcmpeqb	%xmm2, %xmm0
	lea	16(%rdx), %rdx
	pmovmskb %xmm0, %rax
	lea	16(%rcx), %rcx
	movaps	%xmm2, %xmm3
# ifdef USE_AS_STRNCPY
	sub	$16, %r8
	jbe	L(StrncpyExit4Case2OrCase3)
# endif
	test	%rax, %rax
	jnz	L(Shl4LoopExit)

	palignr	$4, %xmm1, %xmm2
	movaps	%xmm2, (%rdx)
	movaps	28(%rcx), %xmm2

	pcmpeqb	%xmm2, %xmm0
	lea	16(%rdx), %rdx
	pmovmskb %xmm0, %rax
	lea	16(%rcx), %rcx
# ifdef USE_AS_STRNCPY
	sub	$16, %r8
	jbe	L(StrncpyExit4Case2OrCase3)
# endif
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
# ifdef USE_AS_STRNCPY
	add	%rax, %r8
# endif
	movaps	-4(%rcx), %xmm1

/* 64 bytes loop */
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
	pcmpeqb	%xmm0, %xmm7
	pmovmskb %xmm7, %rax
	movaps	%xmm5, %xmm7
	palignr	$4, %xmm4, %xmm5
	test	%rax, %rax
	palignr	$4, %xmm3, %xmm4
	jnz	L(Shl4Start)
# ifdef USE_AS_STRNCPY
	sub	$64, %r8
	jbe	L(StrncpyLeave4)
# endif
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
L(Shl5):
	movaps	-5(%rcx), %xmm1
	movaps	11(%rcx), %xmm2
L(Shl5Start):
	pcmpeqb	%xmm2, %xmm0
	pmovmskb %xmm0, %rax
	movaps	%xmm2, %xmm3
# ifdef USE_AS_STRNCPY
	sub	$16, %r8
	jbe	L(StrncpyExit5Case2OrCase3)
# endif
	test	%rax, %rax
	jnz	L(Shl5LoopExit)

	palignr	$5, %xmm1, %xmm2
	movaps	%xmm2, (%rdx)
	movaps	27(%rcx), %xmm2

	pcmpeqb	%xmm2, %xmm0
	lea	16(%rdx), %rdx
	pmovmskb %xmm0, %rax
	lea	16(%rcx), %rcx
	movaps	%xmm2, %xmm1
# ifdef USE_AS_STRNCPY
	sub	$16, %r8
	jbe	L(StrncpyExit5Case2OrCase3)
# endif
	test	%rax, %rax
	jnz	L(Shl5LoopExit)

	palignr	$5, %xmm3, %xmm2
	movaps	%xmm2, (%rdx)
	movaps	27(%rcx), %xmm2

	pcmpeqb	%xmm2, %xmm0
	lea	16(%rdx), %rdx
	pmovmskb %xmm0, %rax
	lea	16(%rcx), %rcx
	movaps	%xmm2, %xmm3
# ifdef USE_AS_STRNCPY
	sub	$16, %r8
	jbe	L(StrncpyExit5Case2OrCase3)
# endif
	test	%rax, %rax
	jnz	L(Shl5LoopExit)

	palignr	$5, %xmm1, %xmm2
	movaps	%xmm2, (%rdx)
	movaps	27(%rcx), %xmm2

	pcmpeqb	%xmm2, %xmm0
	lea	16(%rdx), %rdx
	pmovmskb %xmm0, %rax
	lea	16(%rcx), %rcx
# ifdef USE_AS_STRNCPY
	sub	$16, %r8
	jbe	L(StrncpyExit5Case2OrCase3)
# endif
	test	%rax, %rax
	jnz	L(Shl5LoopExit)

	palignr	$5, %xmm3, %xmm2
	movaps	%xmm2, (%rdx)
	lea	27(%rcx), %rcx
	lea	16(%rdx), %rdx

	mov	%rcx, %rax
	and	$-0x40, %rcx
	sub	%rcx, %rax
	lea	-11(%rcx), %rcx
	sub	%rax, %rdx
# ifdef USE_AS_STRNCPY
	add	%rax, %r8
# endif
	movaps	-5(%rcx), %xmm1

/* 64 bytes loop */
	.p2align 4
L(Shl5LoopStart):
	movaps	11(%rcx), %xmm2
	movaps	27(%rcx), %xmm3
	movaps	%xmm3, %xmm6
	movaps	43(%rcx), %xmm4
	movaps	%xmm4, %xmm7
	movaps	59(%rcx), %xmm5
	pminub	%xmm2, %xmm6
	pminub	%xmm5, %xmm7
	pminub	%xmm6, %xmm7
	pcmpeqb	%xmm0, %xmm7
	pmovmskb %xmm7, %rax
	movaps	%xmm5, %xmm7
	palignr	$5, %xmm4, %xmm5
	test	%rax, %rax
	palignr	$5, %xmm3, %xmm4
	jnz	L(Shl5Start)
# ifdef USE_AS_STRNCPY
	sub	$64, %r8
	jbe	L(StrncpyLeave5)
# endif
	palignr	$5, %xmm2, %xmm3
	lea	64(%rcx), %rcx
	palignr	$5, %xmm1, %xmm2
	movaps	%xmm7, %xmm1
	movaps	%xmm5, 48(%rdx)
	movaps	%xmm4, 32(%rdx)
	movaps	%xmm3, 16(%rdx)
	movaps	%xmm2, (%rdx)
	lea	64(%rdx), %rdx
	jmp	L(Shl5LoopStart)

L(Shl5LoopExit):
	movdqu	-5(%rcx), %xmm1
	mov	$11, %rsi
	movdqu	%xmm1, -5(%rdx)
	jmp	L(CopyFrom1To16Bytes)

	.p2align 4
L(Shl6):
	movaps	-6(%rcx), %xmm1
	movaps	10(%rcx), %xmm2
L(Shl6Start):
	pcmpeqb	%xmm2, %xmm0
	pmovmskb %xmm0, %rax
	movaps	%xmm2, %xmm3
# ifdef USE_AS_STRNCPY
	sub	$16, %r8
	jbe	L(StrncpyExit6Case2OrCase3)
# endif
	test	%rax, %rax
	jnz	L(Shl6LoopExit)

	palignr	$6, %xmm1, %xmm2
	movaps	%xmm2, (%rdx)
	movaps	26(%rcx), %xmm2

	pcmpeqb	%xmm2, %xmm0
	lea	16(%rdx), %rdx
	pmovmskb %xmm0, %rax
	lea	16(%rcx), %rcx
	movaps	%xmm2, %xmm1
# ifdef USE_AS_STRNCPY
	sub	$16, %r8
	jbe	L(StrncpyExit6Case2OrCase3)
# endif
	test	%rax, %rax
	jnz	L(Shl6LoopExit)

	palignr	$6, %xmm3, %xmm2
	movaps	%xmm2, (%rdx)
	movaps	26(%rcx), %xmm2

	pcmpeqb	%xmm2, %xmm0
	lea	16(%rdx), %rdx
	pmovmskb %xmm0, %rax
	lea	16(%rcx), %rcx
	movaps	%xmm2, %xmm3
# ifdef USE_AS_STRNCPY
	sub	$16, %r8
	jbe	L(StrncpyExit6Case2OrCase3)
# endif
	test	%rax, %rax
	jnz	L(Shl6LoopExit)

	palignr	$6, %xmm1, %xmm2
	movaps	%xmm2, (%rdx)
	movaps	26(%rcx), %xmm2

	pcmpeqb	%xmm2, %xmm0
	lea	16(%rdx), %rdx
	pmovmskb %xmm0, %rax
	lea	16(%rcx), %rcx
# ifdef USE_AS_STRNCPY
	sub	$16, %r8
	jbe	L(StrncpyExit6Case2OrCase3)
# endif
	test	%rax, %rax
	jnz	L(Shl6LoopExit)

	palignr	$6, %xmm3, %xmm2
	movaps	%xmm2, (%rdx)
	lea	26(%rcx), %rcx
	lea	16(%rdx), %rdx

	mov	%rcx, %rax
	and	$-0x40, %rcx
	sub	%rcx, %rax
	lea	-10(%rcx), %rcx
	sub	%rax, %rdx
# ifdef USE_AS_STRNCPY
	add	%rax, %r8
# endif
	movaps	-6(%rcx), %xmm1

/* 64 bytes loop */
	.p2align 4
L(Shl6LoopStart):
	movaps	10(%rcx), %xmm2
	movaps	26(%rcx), %xmm3
	movaps	%xmm3, %xmm6
	movaps	42(%rcx), %xmm4
	movaps	%xmm4, %xmm7
	movaps	58(%rcx), %xmm5
	pminub	%xmm2, %xmm6
	pminub	%xmm5, %xmm7
	pminub	%xmm6, %xmm7
	pcmpeqb	%xmm0, %xmm7
	pmovmskb %xmm7, %rax
	movaps	%xmm5, %xmm7
	palignr	$6, %xmm4, %xmm5
	test	%rax, %rax
	palignr	$6, %xmm3, %xmm4
	jnz	L(Shl6Start)
# ifdef USE_AS_STRNCPY
	sub	$64, %r8
	jbe	L(StrncpyLeave6)
# endif
	palignr	$6, %xmm2, %xmm3
	lea	64(%rcx), %rcx
	palignr	$6, %xmm1, %xmm2
	movaps	%xmm7, %xmm1
	movaps	%xmm5, 48(%rdx)
	movaps	%xmm4, 32(%rdx)
	movaps	%xmm3, 16(%rdx)
	movaps	%xmm2, (%rdx)
	lea	64(%rdx), %rdx
	jmp	L(Shl6LoopStart)

L(Shl6LoopExit):
	mov	(%rcx), %r9
	mov	6(%rcx), %esi
	mov	%r9, (%rdx)
	mov	%esi, 6(%rdx)
	mov	$10, %rsi
	jmp	L(CopyFrom1To16Bytes)

	.p2align 4
L(Shl7):
	movaps	-7(%rcx), %xmm1
	movaps	9(%rcx), %xmm2
L(Shl7Start):
	pcmpeqb	%xmm2, %xmm0
	pmovmskb %xmm0, %rax
	movaps	%xmm2, %xmm3
# ifdef USE_AS_STRNCPY
	sub	$16, %r8
	jbe	L(StrncpyExit7Case2OrCase3)
# endif
	test	%rax, %rax
	jnz	L(Shl7LoopExit)

	palignr	$7, %xmm1, %xmm2
	movaps	%xmm2, (%rdx)
	movaps	25(%rcx), %xmm2

	pcmpeqb	%xmm2, %xmm0
	lea	16(%rdx), %rdx
	pmovmskb %xmm0, %rax
	lea	16(%rcx), %rcx
	movaps	%xmm2, %xmm1
# ifdef USE_AS_STRNCPY
	sub	$16, %r8
	jbe	L(StrncpyExit7Case2OrCase3)
# endif
	test	%rax, %rax
	jnz	L(Shl7LoopExit)

	palignr	$7, %xmm3, %xmm2
	movaps	%xmm2, (%rdx)
	movaps	25(%rcx), %xmm2

	pcmpeqb	%xmm2, %xmm0
	lea	16(%rdx), %rdx
	pmovmskb %xmm0, %rax
	lea	16(%rcx), %rcx
	movaps	%xmm2, %xmm3
# ifdef USE_AS_STRNCPY
	sub	$16, %r8
	jbe	L(StrncpyExit7Case2OrCase3)
# endif
	test	%rax, %rax
	jnz	L(Shl7LoopExit)

	palignr	$7, %xmm1, %xmm2
	movaps	%xmm2, (%rdx)
	movaps	25(%rcx), %xmm2

	pcmpeqb	%xmm2, %xmm0
	lea	16(%rdx), %rdx
	pmovmskb %xmm0, %rax
	lea	16(%rcx), %rcx
# ifdef USE_AS_STRNCPY
	sub	$16, %r8
	jbe	L(StrncpyExit7Case2OrCase3)
# endif
	test	%rax, %rax
	jnz	L(Shl7LoopExit)

	palignr	$7, %xmm3, %xmm2
	movaps	%xmm2, (%rdx)
	lea	25(%rcx), %rcx
	lea	16(%rdx), %rdx

	mov	%rcx, %rax
	and	$-0x40, %rcx
	sub	%rcx, %rax
	lea	-9(%rcx), %rcx
	sub	%rax, %rdx
# ifdef USE_AS_STRNCPY
	add	%rax, %r8
# endif
	movaps	-7(%rcx), %xmm1

/* 64 bytes loop */
	.p2align 4
L(Shl7LoopStart):
	movaps	9(%rcx), %xmm2
	movaps	25(%rcx), %xmm3
	movaps	%xmm3, %xmm6
	movaps	41(%rcx), %xmm4
	movaps	%xmm4, %xmm7
	movaps	57(%rcx), %xmm5
	pminub	%xmm2, %xmm6
	pminub	%xmm5, %xmm7
	pminub	%xmm6, %xmm7
	pcmpeqb	%xmm0, %xmm7
	pmovmskb %xmm7, %rax
	movaps	%xmm5, %xmm7
	palignr	$7, %xmm4, %xmm5
	test	%rax, %rax
	palignr	$7, %xmm3, %xmm4
	jnz	L(Shl7Start)
# ifdef USE_AS_STRNCPY
	sub	$64, %r8
	jbe	L(StrncpyLeave7)
# endif
	palignr	$7, %xmm2, %xmm3
	lea	64(%rcx), %rcx
	palignr	$7, %xmm1, %xmm2
	movaps	%xmm7, %xmm1
	movaps	%xmm5, 48(%rdx)
	movaps	%xmm4, 32(%rdx)
	movaps	%xmm3, 16(%rdx)
	movaps	%xmm2, (%rdx)
	lea	64(%rdx), %rdx
	jmp	L(Shl7LoopStart)

L(Shl7LoopExit):
	mov	(%rcx), %r9
	mov	5(%rcx), %esi
	mov	%r9, (%rdx)
	mov	%esi, 5(%rdx)
	mov	$9, %rsi
	jmp	L(CopyFrom1To16Bytes)

	.p2align 4
L(Shl8):
	movaps	-8(%rcx), %xmm1
	movaps	8(%rcx), %xmm2
L(Shl8Start):
	pcmpeqb	%xmm2, %xmm0
	pmovmskb %xmm0, %rax
	movaps	%xmm2, %xmm3
# ifdef USE_AS_STRNCPY
	sub	$16, %r8
	jbe	L(StrncpyExit8Case2OrCase3)
# endif
	test	%rax, %rax
	jnz	L(Shl8LoopExit)

	palignr	$8, %xmm1, %xmm2
	movaps	%xmm2, (%rdx)
	movaps	24(%rcx), %xmm2

	pcmpeqb	%xmm2, %xmm0
	lea	16(%rdx), %rdx
	pmovmskb %xmm0, %rax
	lea	16(%rcx), %rcx
	movaps	%xmm2, %xmm1
# ifdef USE_AS_STRNCPY
	sub	$16, %r8
	jbe	L(StrncpyExit8Case2OrCase3)
# endif
	test	%rax, %rax
	jnz	L(Shl8LoopExit)

	palignr	$8, %xmm3, %xmm2
	movaps	%xmm2, (%rdx)
	movaps	24(%rcx), %xmm2

	pcmpeqb	%xmm2, %xmm0
	lea	16(%rdx), %rdx
	pmovmskb %xmm0, %rax
	lea	16(%rcx), %rcx
	movaps	%xmm2, %xmm3
# ifdef USE_AS_STRNCPY
	sub	$16, %r8
	jbe	L(StrncpyExit8Case2OrCase3)
# endif
	test	%rax, %rax
	jnz	L(Shl8LoopExit)

	palignr	$8, %xmm1, %xmm2
	movaps	%xmm2, (%rdx)
	movaps	24(%rcx), %xmm2

	pcmpeqb	%xmm2, %xmm0
	lea	16(%rdx), %rdx
	pmovmskb %xmm0, %rax
	lea	16(%rcx), %rcx
# ifdef USE_AS_STRNCPY
	sub	$16, %r8
	jbe	L(StrncpyExit8Case2OrCase3)
# endif
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
# ifdef USE_AS_STRNCPY
	add	%rax, %r8
# endif
	movaps	-8(%rcx), %xmm1

/* 64 bytes loop */
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
	pcmpeqb	%xmm0, %xmm7
	pmovmskb %xmm7, %rax
	movaps	%xmm5, %xmm7
	palignr	$8, %xmm4, %xmm5
	test	%rax, %rax
	palignr	$8, %xmm3, %xmm4
	jnz	L(Shl8Start)
# ifdef USE_AS_STRNCPY
	sub	$64, %r8
	jbe	L(StrncpyLeave8)
# endif
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
L(Shl9):
	movaps	-9(%rcx), %xmm1
	movaps	7(%rcx), %xmm2
L(Shl9Start):
	pcmpeqb	%xmm2, %xmm0
	pmovmskb %xmm0, %rax
	movaps	%xmm2, %xmm3
# ifdef USE_AS_STRNCPY
	sub	$16, %r8
	jbe	L(StrncpyExit9Case2OrCase3)
# endif
	test	%rax, %rax
	jnz	L(Shl9LoopExit)

	palignr	$9, %xmm1, %xmm2
	movaps	%xmm2, (%rdx)
	movaps	23(%rcx), %xmm2

	pcmpeqb	%xmm2, %xmm0
	lea	16(%rdx), %rdx
	pmovmskb %xmm0, %rax
	lea	16(%rcx), %rcx
	movaps	%xmm2, %xmm1
# ifdef USE_AS_STRNCPY
	sub	$16, %r8
	jbe	L(StrncpyExit9Case2OrCase3)
# endif
	test	%rax, %rax
	jnz	L(Shl9LoopExit)

	palignr	$9, %xmm3, %xmm2
	movaps	%xmm2, (%rdx)
	movaps	23(%rcx), %xmm2

	pcmpeqb	%xmm2, %xmm0
	lea	16(%rdx), %rdx
	pmovmskb %xmm0, %rax
	lea	16(%rcx), %rcx
	movaps	%xmm2, %xmm3
# ifdef USE_AS_STRNCPY
	sub	$16, %r8
	jbe	L(StrncpyExit9Case2OrCase3)
# endif
	test	%rax, %rax
	jnz	L(Shl9LoopExit)

	palignr	$9, %xmm1, %xmm2
	movaps	%xmm2, (%rdx)
	movaps	23(%rcx), %xmm2

	pcmpeqb	%xmm2, %xmm0
	lea	16(%rdx), %rdx
	pmovmskb %xmm0, %rax
	lea	16(%rcx), %rcx
# ifdef USE_AS_STRNCPY
	sub	$16, %r8
	jbe	L(StrncpyExit9Case2OrCase3)
# endif
	test	%rax, %rax
	jnz	L(Shl9LoopExit)

	palignr	$9, %xmm3, %xmm2
	movaps	%xmm2, (%rdx)
	lea	23(%rcx), %rcx
	lea	16(%rdx), %rdx

	mov	%rcx, %rax
	and	$-0x40, %rcx
	sub	%rcx, %rax
	lea	-7(%rcx), %rcx
	sub	%rax, %rdx
# ifdef USE_AS_STRNCPY
	add	%rax, %r8
# endif
	movaps	-9(%rcx), %xmm1

/* 64 bytes loop */
	.p2align 4
L(Shl9LoopStart):
	movaps	7(%rcx), %xmm2
	movaps	23(%rcx), %xmm3
	movaps	%xmm3, %xmm6
	movaps	39(%rcx), %xmm4
	movaps	%xmm4, %xmm7
	movaps	55(%rcx), %xmm5
	pminub	%xmm2, %xmm6
	pminub	%xmm5, %xmm7
	pminub	%xmm6, %xmm7
	pcmpeqb	%xmm0, %xmm7
	pmovmskb %xmm7, %rax
	movaps	%xmm5, %xmm7
	palignr	$9, %xmm4, %xmm5
	test	%rax, %rax
	palignr	$9, %xmm3, %xmm4
	jnz	L(Shl9Start)
# ifdef USE_AS_STRNCPY
	sub	$64, %r8
	jbe	L(StrncpyLeave9)
# endif
	palignr	$9, %xmm2, %xmm3
	lea	64(%rcx), %rcx
	palignr	$9, %xmm1, %xmm2
	movaps	%xmm7, %xmm1
	movaps	%xmm5, 48(%rdx)
	movaps	%xmm4, 32(%rdx)
	movaps	%xmm3, 16(%rdx)
	movaps	%xmm2, (%rdx)
	lea	64(%rdx), %rdx
	jmp	L(Shl9LoopStart)

L(Shl9LoopExit):
	mov	-1(%rcx), %r9
	mov	$7, %rsi
	mov	%r9, -1(%rdx)
	jmp	L(CopyFrom1To16Bytes)

	.p2align 4
L(Shl10):
	movaps	-10(%rcx), %xmm1
	movaps	6(%rcx), %xmm2
L(Shl10Start):
	pcmpeqb	%xmm2, %xmm0
	pmovmskb %xmm0, %rax
	movaps	%xmm2, %xmm3
# ifdef USE_AS_STRNCPY
	sub	$16, %r8
	jbe	L(StrncpyExit10Case2OrCase3)
# endif
	test	%rax, %rax
	jnz	L(Shl10LoopExit)

	palignr	$10, %xmm1, %xmm2
	movaps	%xmm2, (%rdx)
	movaps	22(%rcx), %xmm2

	pcmpeqb	%xmm2, %xmm0
	lea	16(%rdx), %rdx
	pmovmskb %xmm0, %rax
	lea	16(%rcx), %rcx
	movaps	%xmm2, %xmm1
# ifdef USE_AS_STRNCPY
	sub	$16, %r8
	jbe	L(StrncpyExit10Case2OrCase3)
# endif
	test	%rax, %rax
	jnz	L(Shl10LoopExit)

	palignr	$10, %xmm3, %xmm2
	movaps	%xmm2, (%rdx)
	movaps	22(%rcx), %xmm2

	pcmpeqb	%xmm2, %xmm0
	lea	16(%rdx), %rdx
	pmovmskb %xmm0, %rax
	lea	16(%rcx), %rcx
	movaps	%xmm2, %xmm3
# ifdef USE_AS_STRNCPY
	sub	$16, %r8
	jbe	L(StrncpyExit10Case2OrCase3)
# endif
	test	%rax, %rax
	jnz	L(Shl10LoopExit)

	palignr	$10, %xmm1, %xmm2
	movaps	%xmm2, (%rdx)
	movaps	22(%rcx), %xmm2

	pcmpeqb	%xmm2, %xmm0
	lea	16(%rdx), %rdx
	pmovmskb %xmm0, %rax
	lea	16(%rcx), %rcx
# ifdef USE_AS_STRNCPY
	sub	$16, %r8
	jbe	L(StrncpyExit10Case2OrCase3)
# endif
	test	%rax, %rax
	jnz	L(Shl10LoopExit)

	palignr	$10, %xmm3, %xmm2
	movaps	%xmm2, (%rdx)
	lea	22(%rcx), %rcx
	lea	16(%rdx), %rdx

	mov	%rcx, %rax
	and	$-0x40, %rcx
	sub	%rcx, %rax
	lea	-6(%rcx), %rcx
	sub	%rax, %rdx
# ifdef USE_AS_STRNCPY
	add	%rax, %r8
# endif
	movaps	-10(%rcx), %xmm1

/* 64 bytes loop */
	.p2align 4
L(Shl10LoopStart):
	movaps	6(%rcx), %xmm2
	movaps	22(%rcx), %xmm3
	movaps	%xmm3, %xmm6
	movaps	38(%rcx), %xmm4
	movaps	%xmm4, %xmm7
	movaps	54(%rcx), %xmm5
	pminub	%xmm2, %xmm6
	pminub	%xmm5, %xmm7
	pminub	%xmm6, %xmm7
	pcmpeqb	%xmm0, %xmm7
	pmovmskb %xmm7, %rax
	movaps	%xmm5, %xmm7
	palignr	$10, %xmm4, %xmm5
	test	%rax, %rax
	palignr	$10, %xmm3, %xmm4
	jnz	L(Shl10Start)
# ifdef USE_AS_STRNCPY
	sub	$64, %r8
	jbe	L(StrncpyLeave10)
# endif
	palignr	$10, %xmm2, %xmm3
	lea	64(%rcx), %rcx
	palignr	$10, %xmm1, %xmm2
	movaps	%xmm7, %xmm1
	movaps	%xmm5, 48(%rdx)
	movaps	%xmm4, 32(%rdx)
	movaps	%xmm3, 16(%rdx)
	movaps	%xmm2, (%rdx)
	lea	64(%rdx), %rdx
	jmp	L(Shl10LoopStart)

L(Shl10LoopExit):
	mov	-2(%rcx), %r9
	mov	$6, %rsi
	mov	%r9, -2(%rdx)
	jmp	L(CopyFrom1To16Bytes)

	.p2align 4
L(Shl11):
	movaps	-11(%rcx), %xmm1
	movaps	5(%rcx), %xmm2
L(Shl11Start):
	pcmpeqb	%xmm2, %xmm0
	pmovmskb %xmm0, %rax
	movaps	%xmm2, %xmm3
# ifdef USE_AS_STRNCPY
	sub	$16, %r8
	jbe	L(StrncpyExit11Case2OrCase3)
# endif
	test	%rax, %rax
	jnz	L(Shl11LoopExit)

	palignr	$11, %xmm1, %xmm2
	movaps	%xmm2, (%rdx)
	movaps	21(%rcx), %xmm2

	pcmpeqb	%xmm2, %xmm0
	lea	16(%rdx), %rdx
	pmovmskb %xmm0, %rax
	lea	16(%rcx), %rcx
	movaps	%xmm2, %xmm1
# ifdef USE_AS_STRNCPY
	sub	$16, %r8
	jbe	L(StrncpyExit11Case2OrCase3)
# endif
	test	%rax, %rax
	jnz	L(Shl11LoopExit)

	palignr	$11, %xmm3, %xmm2
	movaps	%xmm2, (%rdx)
	movaps	21(%rcx), %xmm2

	pcmpeqb	%xmm2, %xmm0
	lea	16(%rdx), %rdx
	pmovmskb %xmm0, %rax
	lea	16(%rcx), %rcx
	movaps	%xmm2, %xmm3
# ifdef USE_AS_STRNCPY
	sub	$16, %r8
	jbe	L(StrncpyExit11Case2OrCase3)
# endif
	test	%rax, %rax
	jnz	L(Shl11LoopExit)

	palignr	$11, %xmm1, %xmm2
	movaps	%xmm2, (%rdx)
	movaps	21(%rcx), %xmm2

	pcmpeqb	%xmm2, %xmm0
	lea	16(%rdx), %rdx
	pmovmskb %xmm0, %rax
	lea	16(%rcx), %rcx
# ifdef USE_AS_STRNCPY
	sub	$16, %r8
	jbe	L(StrncpyExit11Case2OrCase3)
# endif
	test	%rax, %rax
	jnz	L(Shl11LoopExit)

	palignr	$11, %xmm3, %xmm2
	movaps	%xmm2, (%rdx)
	lea	21(%rcx), %rcx
	lea	16(%rdx), %rdx

	mov	%rcx, %rax
	and	$-0x40, %rcx
	sub	%rcx, %rax
	lea	-5(%rcx), %rcx
	sub	%rax, %rdx
# ifdef USE_AS_STRNCPY
	add	%rax, %r8
# endif
	movaps	-11(%rcx), %xmm1

/* 64 bytes loop */
	.p2align 4
L(Shl11LoopStart):
	movaps	5(%rcx), %xmm2
	movaps	21(%rcx), %xmm3
	movaps	%xmm3, %xmm6
	movaps	37(%rcx), %xmm4
	movaps	%xmm4, %xmm7
	movaps	53(%rcx), %xmm5
	pminub	%xmm2, %xmm6
	pminub	%xmm5, %xmm7
	pminub	%xmm6, %xmm7
	pcmpeqb	%xmm0, %xmm7
	pmovmskb %xmm7, %rax
	movaps	%xmm5, %xmm7
	palignr	$11, %xmm4, %xmm5
	test	%rax, %rax
	palignr	$11, %xmm3, %xmm4
	jnz	L(Shl11Start)
# ifdef USE_AS_STRNCPY
	sub	$64, %r8
	jbe	L(StrncpyLeave11)
# endif
	palignr	$11, %xmm2, %xmm3
	lea	64(%rcx), %rcx
	palignr	$11, %xmm1, %xmm2
	movaps	%xmm7, %xmm1
	movaps	%xmm5, 48(%rdx)
	movaps	%xmm4, 32(%rdx)
	movaps	%xmm3, 16(%rdx)
	movaps	%xmm2, (%rdx)
	lea	64(%rdx), %rdx
	jmp	L(Shl11LoopStart)

L(Shl11LoopExit):
	mov	-3(%rcx), %r9
	mov	$5, %rsi
	mov	%r9, -3(%rdx)
	jmp	L(CopyFrom1To16Bytes)

	.p2align 4
L(Shl12):
	movaps	-12(%rcx), %xmm1
	movaps	4(%rcx), %xmm2
L(Shl12Start):
	pcmpeqb	%xmm2, %xmm0
	pmovmskb %xmm0, %rax
	movaps	%xmm2, %xmm3
# ifdef USE_AS_STRNCPY
	sub	$16, %r8
	jbe	L(StrncpyExit12Case2OrCase3)
# endif
	test	%rax, %rax
	jnz	L(Shl12LoopExit)

	palignr	$12, %xmm1, %xmm2
	movaps	%xmm2, (%rdx)
	movaps	20(%rcx), %xmm2

	pcmpeqb	%xmm2, %xmm0
	lea	16(%rdx), %rdx
	pmovmskb %xmm0, %rax
	lea	16(%rcx), %rcx
	movaps	%xmm2, %xmm1
# ifdef USE_AS_STRNCPY
	sub	$16, %r8
	jbe	L(StrncpyExit12Case2OrCase3)
# endif
	test	%rax, %rax
	jnz	L(Shl12LoopExit)

	palignr	$12, %xmm3, %xmm2
	movaps	%xmm2, (%rdx)
	movaps	20(%rcx), %xmm2

	pcmpeqb	%xmm2, %xmm0
	lea	16(%rdx), %rdx
	pmovmskb %xmm0, %rax
	lea	16(%rcx), %rcx
	movaps	%xmm2, %xmm3
# ifdef USE_AS_STRNCPY
	sub	$16, %r8
	jbe	L(StrncpyExit12Case2OrCase3)
# endif
	test	%rax, %rax
	jnz	L(Shl12LoopExit)

	palignr	$12, %xmm1, %xmm2
	movaps	%xmm2, (%rdx)
	movaps	20(%rcx), %xmm2

	pcmpeqb	%xmm2, %xmm0
	lea	16(%rdx), %rdx
	pmovmskb %xmm0, %rax
	lea	16(%rcx), %rcx
# ifdef USE_AS_STRNCPY
	sub	$16, %r8
	jbe	L(StrncpyExit12Case2OrCase3)
# endif
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
# ifdef USE_AS_STRNCPY
	add	%rax, %r8
# endif
	movaps	-12(%rcx), %xmm1

/* 64 bytes loop */
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
	pcmpeqb	%xmm0, %xmm7
	pmovmskb %xmm7, %rax
	movaps	%xmm5, %xmm7
	palignr	$12, %xmm4, %xmm5
	test	%rax, %rax
	palignr	$12, %xmm3, %xmm4
	jnz	L(Shl12Start)
# ifdef USE_AS_STRNCPY
	sub	$64, %r8
	jbe	L(StrncpyLeave12)
# endif
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
L(Shl13):
	movaps	-13(%rcx), %xmm1
	movaps	3(%rcx), %xmm2
L(Shl13Start):
	pcmpeqb	%xmm2, %xmm0
	pmovmskb %xmm0, %rax
	movaps	%xmm2, %xmm3
# ifdef USE_AS_STRNCPY
	sub	$16, %r8
	jbe	L(StrncpyExit13Case2OrCase3)
# endif
	test	%rax, %rax
	jnz	L(Shl13LoopExit)

	palignr	$13, %xmm1, %xmm2
	movaps	%xmm2, (%rdx)
	movaps	19(%rcx), %xmm2

	pcmpeqb	%xmm2, %xmm0
	lea	16(%rdx), %rdx
	pmovmskb %xmm0, %rax
	lea	16(%rcx), %rcx
	movaps	%xmm2, %xmm1
# ifdef USE_AS_STRNCPY
	sub	$16, %r8
	jbe	L(StrncpyExit13Case2OrCase3)
# endif
	test	%rax, %rax
	jnz	L(Shl13LoopExit)

	palignr	$13, %xmm3, %xmm2
	movaps	%xmm2, (%rdx)
	movaps	19(%rcx), %xmm2

	pcmpeqb	%xmm2, %xmm0
	lea	16(%rdx), %rdx
	pmovmskb %xmm0, %rax
	lea	16(%rcx), %rcx
	movaps	%xmm2, %xmm3
# ifdef USE_AS_STRNCPY
	sub	$16, %r8
	jbe	L(StrncpyExit13Case2OrCase3)
# endif
	test	%rax, %rax
	jnz	L(Shl13LoopExit)

	palignr	$13, %xmm1, %xmm2
	movaps	%xmm2, (%rdx)
	movaps	19(%rcx), %xmm2

	pcmpeqb	%xmm2, %xmm0
	lea	16(%rdx), %rdx
	pmovmskb %xmm0, %rax
	lea	16(%rcx), %rcx
# ifdef USE_AS_STRNCPY
	sub	$16, %r8
	jbe	L(StrncpyExit13Case2OrCase3)
# endif
	test	%rax, %rax
	jnz	L(Shl13LoopExit)

	palignr	$13, %xmm3, %xmm2
	movaps	%xmm2, (%rdx)
	lea	19(%rcx), %rcx
	lea	16(%rdx), %rdx

	mov	%rcx, %rax
	and	$-0x40, %rcx
	sub	%rcx, %rax
	lea	-3(%rcx), %rcx
	sub	%rax, %rdx
# ifdef USE_AS_STRNCPY
	add	%rax, %r8
# endif
	movaps	-13(%rcx), %xmm1

/* 64 bytes loop */
	.p2align 4
L(Shl13LoopStart):
	movaps	3(%rcx), %xmm2
	movaps	19(%rcx), %xmm3
	movaps	%xmm3, %xmm6
	movaps	35(%rcx), %xmm4
	movaps	%xmm4, %xmm7
	movaps	51(%rcx), %xmm5
	pminub	%xmm2, %xmm6
	pminub	%xmm5, %xmm7
	pminub	%xmm6, %xmm7
	pcmpeqb	%xmm0, %xmm7
	pmovmskb %xmm7, %rax
	movaps	%xmm5, %xmm7
	palignr	$13, %xmm4, %xmm5
	test	%rax, %rax
	palignr	$13, %xmm3, %xmm4
	jnz	L(Shl13Start)
# ifdef USE_AS_STRNCPY
	sub	$64, %r8
	jbe	L(StrncpyLeave13)
# endif
	palignr	$13, %xmm2, %xmm3
	lea	64(%rcx), %rcx
	palignr	$13, %xmm1, %xmm2
	movaps	%xmm7, %xmm1
	movaps	%xmm5, 48(%rdx)
	movaps	%xmm4, 32(%rdx)
	movaps	%xmm3, 16(%rdx)
	movaps	%xmm2, (%rdx)
	lea	64(%rdx), %rdx
	jmp	L(Shl13LoopStart)

L(Shl13LoopExit):
	mov	-1(%rcx), %r9d
	mov	$3, %rsi
	mov	%r9d, -1(%rdx)
	jmp	L(CopyFrom1To16Bytes)

	.p2align 4
L(Shl14):
	movaps	-14(%rcx), %xmm1
	movaps	2(%rcx), %xmm2
L(Shl14Start):
	pcmpeqb	%xmm2, %xmm0
	pmovmskb %xmm0, %rax
	movaps	%xmm2, %xmm3
# ifdef USE_AS_STRNCPY
	sub	$16, %r8
	jbe	L(StrncpyExit14Case2OrCase3)
# endif
	test	%rax, %rax
	jnz	L(Shl14LoopExit)

	palignr	$14, %xmm1, %xmm2
	movaps	%xmm2, (%rdx)
	movaps	18(%rcx), %xmm2

	pcmpeqb	%xmm2, %xmm0
	lea	16(%rdx), %rdx
	pmovmskb %xmm0, %rax
	lea	16(%rcx), %rcx
	movaps	%xmm2, %xmm1
# ifdef USE_AS_STRNCPY
	sub	$16, %r8
	jbe	L(StrncpyExit14Case2OrCase3)
# endif
	test	%rax, %rax
	jnz	L(Shl14LoopExit)

	palignr	$14, %xmm3, %xmm2
	movaps	%xmm2, (%rdx)
	movaps	18(%rcx), %xmm2

	pcmpeqb	%xmm2, %xmm0
	lea	16(%rdx), %rdx
	pmovmskb %xmm0, %rax
	lea	16(%rcx), %rcx
	movaps	%xmm2, %xmm3
# ifdef USE_AS_STRNCPY
	sub	$16, %r8
	jbe	L(StrncpyExit14Case2OrCase3)
# endif
	test	%rax, %rax
	jnz	L(Shl14LoopExit)

	palignr	$14, %xmm1, %xmm2
	movaps	%xmm2, (%rdx)
	movaps	18(%rcx), %xmm2

	pcmpeqb	%xmm2, %xmm0
	lea	16(%rdx), %rdx
	pmovmskb %xmm0, %rax
	lea	16(%rcx), %rcx
# ifdef USE_AS_STRNCPY
	sub	$16, %r8
	jbe	L(StrncpyExit14Case2OrCase3)
# endif
	test	%rax, %rax
	jnz	L(Shl14LoopExit)

	palignr	$14, %xmm3, %xmm2
	movaps	%xmm2, (%rdx)
	lea	18(%rcx), %rcx
	lea	16(%rdx), %rdx

	mov	%rcx, %rax
	and	$-0x40, %rcx
	sub	%rcx, %rax
	lea	-2(%rcx), %rcx
	sub	%rax, %rdx
# ifdef USE_AS_STRNCPY
	add	%rax, %r8
# endif
	movaps	-14(%rcx), %xmm1

/* 64 bytes loop */
	.p2align 4
L(Shl14LoopStart):
	movaps	2(%rcx), %xmm2
	movaps	18(%rcx), %xmm3
	movaps	%xmm3, %xmm6
	movaps	34(%rcx), %xmm4
	movaps	%xmm4, %xmm7
	movaps	50(%rcx), %xmm5
	pminub	%xmm2, %xmm6
	pminub	%xmm5, %xmm7
	pminub	%xmm6, %xmm7
	pcmpeqb	%xmm0, %xmm7
	pmovmskb %xmm7, %rax
	movaps	%xmm5, %xmm7
	palignr	$14, %xmm4, %xmm5
	test	%rax, %rax
	palignr	$14, %xmm3, %xmm4
	jnz	L(Shl14Start)
# ifdef USE_AS_STRNCPY
	sub	$64, %r8
	jbe	L(StrncpyLeave14)
# endif
	palignr	$14, %xmm2, %xmm3
	lea	64(%rcx), %rcx
	palignr	$14, %xmm1, %xmm2
	movaps	%xmm7, %xmm1
	movaps	%xmm5, 48(%rdx)
	movaps	%xmm4, 32(%rdx)
	movaps	%xmm3, 16(%rdx)
	movaps	%xmm2, (%rdx)
	lea	64(%rdx), %rdx
	jmp	L(Shl14LoopStart)

L(Shl14LoopExit):
	mov	-2(%rcx), %r9d
	mov	$2, %rsi
	mov	%r9d, -2(%rdx)
	jmp	L(CopyFrom1To16Bytes)

	.p2align 4
L(Shl15):
	movaps	-15(%rcx), %xmm1
	movaps	1(%rcx), %xmm2
L(Shl15Start):
	pcmpeqb	%xmm2, %xmm0
	pmovmskb %xmm0, %rax
	movaps	%xmm2, %xmm3
# ifdef USE_AS_STRNCPY
	sub	$16, %r8
	jbe	L(StrncpyExit15Case2OrCase3)
# endif
	test	%rax, %rax
	jnz	L(Shl15LoopExit)

	palignr	$15, %xmm1, %xmm2
	movaps	%xmm2, (%rdx)
	movaps	17(%rcx), %xmm2

	pcmpeqb	%xmm2, %xmm0
	lea	16(%rdx), %rdx
	pmovmskb %xmm0, %rax
	lea	16(%rcx), %rcx
	movaps	%xmm2, %xmm1
# ifdef USE_AS_STRNCPY
	sub	$16, %r8
	jbe	L(StrncpyExit15Case2OrCase3)
# endif
	test	%rax, %rax
	jnz	L(Shl15LoopExit)

	palignr	$15, %xmm3, %xmm2
	movaps	%xmm2, (%rdx)
	movaps	17(%rcx), %xmm2

	pcmpeqb	%xmm2, %xmm0
	lea	16(%rdx), %rdx
	pmovmskb %xmm0, %rax
	lea	16(%rcx), %rcx
	movaps	%xmm2, %xmm3
# ifdef USE_AS_STRNCPY
	sub	$16, %r8
	jbe	L(StrncpyExit15Case2OrCase3)
# endif
	test	%rax, %rax
	jnz	L(Shl15LoopExit)

	palignr	$15, %xmm1, %xmm2
	movaps	%xmm2, (%rdx)
	movaps	17(%rcx), %xmm2

	pcmpeqb	%xmm2, %xmm0
	lea	16(%rdx), %rdx
	pmovmskb %xmm0, %rax
	lea	16(%rcx), %rcx
# ifdef USE_AS_STRNCPY
	sub	$16, %r8
	jbe	L(StrncpyExit15Case2OrCase3)
# endif
	test	%rax, %rax
	jnz	L(Shl15LoopExit)

	palignr	$15, %xmm3, %xmm2
	movaps	%xmm2, (%rdx)
	lea	17(%rcx), %rcx
	lea	16(%rdx), %rdx

	mov	%rcx, %rax
	and	$-0x40, %rcx
	sub	%rcx, %rax
	lea	-1(%rcx), %rcx
	sub	%rax, %rdx
# ifdef USE_AS_STRNCPY
	add	%rax, %r8
# endif
	movaps	-15(%rcx), %xmm1

/* 64 bytes loop */
	.p2align 4
L(Shl15LoopStart):
	movaps	1(%rcx), %xmm2
	movaps	17(%rcx), %xmm3
	movaps	%xmm3, %xmm6
	movaps	33(%rcx), %xmm4
	movaps	%xmm4, %xmm7
	movaps	49(%rcx), %xmm5
	pminub	%xmm2, %xmm6
	pminub	%xmm5, %xmm7
	pminub	%xmm6, %xmm7
	pcmpeqb	%xmm0, %xmm7
	pmovmskb %xmm7, %rax
	movaps	%xmm5, %xmm7
	palignr	$15, %xmm4, %xmm5
	test	%rax, %rax
	palignr	$15, %xmm3, %xmm4
	jnz	L(Shl15Start)
# ifdef USE_AS_STRNCPY
	sub	$64, %r8
	jbe	L(StrncpyLeave15)
# endif
	palignr	$15, %xmm2, %xmm3
	lea	64(%rcx), %rcx
	palignr	$15, %xmm1, %xmm2
	movaps	%xmm7, %xmm1
	movaps	%xmm5, 48(%rdx)
	movaps	%xmm4, 32(%rdx)
	movaps	%xmm3, 16(%rdx)
	movaps	%xmm2, (%rdx)
	lea	64(%rdx), %rdx
	jmp	L(Shl15LoopStart)

L(Shl15LoopExit):
	mov	-3(%rcx), %r9d
	mov	$1, %rsi
	mov	%r9d, -3(%rdx)
# ifdef USE_AS_STRCAT
	jmp	L(CopyFrom1To16Bytes)
# endif

# ifndef USE_AS_STRCAT

	.p2align 4
L(CopyFrom1To16Bytes):
#  ifdef USE_AS_STRNCPY
	add	$16, %r8
#  endif
	add	%rsi, %rdx
	add	%rsi, %rcx

	test	%al, %al
	jz	L(ExitHigh)
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

	.p2align 4
L(Exit8):
	mov	(%rcx), %rax
	mov	%rax, (%rdx)
#  ifdef USE_AS_STPCPY
	lea	7(%rdx), %rax
#  else
	mov	%rdi, %rax
#  endif
#  ifdef USE_AS_STRNCPY
	sub	$8, %r8
	lea	8(%rdx), %rcx
	jnz	L(StrncpyFillTailWithZero1)
#   ifdef USE_AS_STPCPY
	cmpb	$1, (%rax)
	sbb	$-1, %rax
#   endif
#  endif
	ret

	.p2align 4
L(ExitHigh):
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

	.p2align 4
L(Exit16):
	mov	(%rcx), %rax
	mov	%rax, (%rdx)
	mov	8(%rcx), %rax
	mov	%rax, 8(%rdx)
#  ifdef USE_AS_STPCPY
	lea	15(%rdx), %rax
#  else
	mov	%rdi, %rax
#  endif
#  ifdef USE_AS_STRNCPY
	sub	$16, %r8
	lea	16(%rdx), %rcx
	jnz	L(StrncpyFillTailWithZero1)
#   ifdef USE_AS_STPCPY
	cmpb	$1, (%rax)
	sbb	$-1, %rax
#   endif
#  endif
	ret

#  ifdef USE_AS_STRNCPY

	.p2align 4
L(CopyFrom1To16BytesCase2):
	add	$16, %r8
	add	%rsi, %rcx
	lea	(%rsi, %rdx), %rsi
	lea	-9(%r8), %rdx
	and	$1<<7, %dh
	or	%al, %dh
	test	%dh, %dh
	lea	(%rsi), %rdx
	jz	L(ExitHighCase2)

	cmp	$1, %r8
	je	L(Exit1)
	test	$0x01, %al
	jnz	L(Exit1)
	cmp	$2, %r8
	je	L(Exit2)
	test	$0x02, %al
	jnz	L(Exit2)
	cmp	$3, %r8
	je	L(Exit3)
	test	$0x04, %al
	jnz	L(Exit3)
	cmp	$4, %r8
	je	L(Exit4)
	test	$0x08, %al
	jnz	L(Exit4)
	cmp	$5, %r8
	je	L(Exit5)
	test	$0x10, %al
	jnz	L(Exit5)
	cmp	$6, %r8
	je	L(Exit6)
	test	$0x20, %al
	jnz	L(Exit6)
	cmp	$7, %r8
	je	L(Exit7)
	test	$0x40, %al
	jnz	L(Exit7)
	jmp	L(Exit8)

	.p2align 4
L(ExitHighCase2):
	cmp	$9, %r8
	je	L(Exit9)
	test	$0x01, %ah
	jnz	L(Exit9)
	cmp	$10, %r8
	je	L(Exit10)
	test	$0x02, %ah
	jnz	L(Exit10)
	cmp	$11, %r8
	je	L(Exit11)
	test	$0x04, %ah
	jnz	L(Exit11)
	cmp	$12, %r8
	je	L(Exit12)
	test	$0x8, %ah
	jnz	L(Exit12)
	cmp	$13, %r8
	je	L(Exit13)
	test	$0x10, %ah
	jnz	L(Exit13)
	cmp	$14, %r8
	je	L(Exit14)
	test	$0x20, %ah
	jnz	L(Exit14)
	cmp	$15, %r8
	je	L(Exit15)
	test	$0x40, %ah
	jnz	L(Exit15)
	jmp	L(Exit16)

L(CopyFrom1To16BytesCase2OrCase3):
	test	%rax, %rax
	jnz	L(CopyFrom1To16BytesCase2)

	.p2align 4
L(CopyFrom1To16BytesCase3):
	add	$16, %r8
	add	%rsi, %rdx
	add	%rsi, %rcx

	cmp	$16, %r8
	je	L(Exit16)
	cmp	$8, %r8
	je	L(Exit8)
	jg	L(More8Case3)
	cmp	$4, %r8
	je	L(Exit4)
	jg	L(More4Case3)
	cmp	$2, %r8
	jl	L(Exit1)
	je	L(Exit2)
	jg	L(Exit3)
L(More8Case3): /* but less than 16 */
	cmp	$12, %r8
	je	L(Exit12)
	jl	L(Less12Case3)
	cmp	$14, %r8
	jl	L(Exit13)
	je	L(Exit14)
	jg	L(Exit15)
L(More4Case3): /* but less than 8 */
	cmp	$6, %r8
	jl	L(Exit5)
	je	L(Exit6)
	jg	L(Exit7)
L(Less12Case3): /* but more than 8 */
	cmp	$10, %r8
	jl	L(Exit9)
	je	L(Exit10)
	jg	L(Exit11)
#  endif

	.p2align 4
L(Exit1):
	movb	(%rcx), %al
	movb	%al, (%rdx)
#  ifdef USE_AS_STPCPY
	lea	(%rdx), %rax
#  else
	mov	%rdi, %rax
#  endif
#  ifdef USE_AS_STRNCPY
	sub	$1, %r8
	lea	1(%rdx), %rcx
	jnz	L(StrncpyFillTailWithZero1)
#   ifdef USE_AS_STPCPY
	cmpb	$1, (%rax)
	sbb	$-1, %rax
#   endif
#  endif
	ret

	.p2align 4
L(Exit2):
	movw	(%rcx), %ax
	movw	%ax, (%rdx)
#  ifdef USE_AS_STPCPY
	lea	1(%rdx), %rax
#  else
	mov	%rdi, %rax
#  endif
#  ifdef USE_AS_STRNCPY
	sub	$2, %r8
	lea	2(%rdx), %rcx
	jnz	L(StrncpyFillTailWithZero1)
#   ifdef USE_AS_STPCPY
	cmpb	$1, (%rax)
	sbb	$-1, %rax
#   endif
#  endif
	ret

	.p2align 4
L(Exit3):
	movw	(%rcx), %ax
	movw	%ax, (%rdx)
	movb	2(%rcx), %al
	movb	%al, 2(%rdx)
#  ifdef USE_AS_STPCPY
	lea	2(%rdx), %rax
#  else
	mov	%rdi, %rax
#  endif
#  ifdef USE_AS_STRNCPY
	sub	$3, %r8
	lea	3(%rdx), %rcx
	jnz	L(StrncpyFillTailWithZero1)
#   ifdef USE_AS_STPCPY
	cmpb	$1, (%rax)
	sbb	$-1, %rax
#   endif
#  endif
	ret

	.p2align 4
L(Exit4):
	movl	(%rcx), %eax
	movl	%eax, (%rdx)
#  ifdef USE_AS_STPCPY
	lea	3(%rdx), %rax
#  else
	mov	%rdi, %rax
#  endif
#  ifdef USE_AS_STRNCPY
	sub	$4, %r8
	lea	4(%rdx), %rcx
	jnz	L(StrncpyFillTailWithZero1)
#   ifdef USE_AS_STPCPY
	cmpb	$1, (%rax)
	sbb	$-1, %rax
#   endif
#  endif
	ret

	.p2align 4
L(Exit5):
	movl	(%rcx), %eax
	movl	%eax, (%rdx)
	movb	4(%rcx), %al
	movb	%al, 4(%rdx)
#  ifdef USE_AS_STPCPY
	lea	4(%rdx), %rax
#  else
	mov	%rdi, %rax
#  endif
#  ifdef USE_AS_STRNCPY
	sub	$5, %r8
	lea	5(%rdx), %rcx
	jnz	L(StrncpyFillTailWithZero1)
#   ifdef USE_AS_STPCPY
	cmpb	$1, (%rax)
	sbb	$-1, %rax
#  endif
#  endif
	ret

	.p2align 4
L(Exit6):
	movl	(%rcx), %eax
	movl	%eax, (%rdx)
	movw	4(%rcx), %ax
	movw	%ax, 4(%rdx)
#  ifdef USE_AS_STPCPY
	lea	5(%rdx), %rax
#  else
	mov	%rdi, %rax
#  endif
#  ifdef USE_AS_STRNCPY
	sub	$6, %r8
	lea	6(%rdx), %rcx
	jnz	L(StrncpyFillTailWithZero1)
#   ifdef USE_AS_STPCPY
	cmpb	$1, (%rax)
	sbb	$-1, %rax
#  endif
#  endif
	ret

	.p2align 4
L(Exit7):
	movl	(%rcx), %eax
	movl	%eax, (%rdx)
	movl	3(%rcx), %eax
	movl	%eax, 3(%rdx)
#  ifdef USE_AS_STPCPY
	lea	6(%rdx), %rax
#  else
	mov	%rdi, %rax
#  endif
#  ifdef USE_AS_STRNCPY
	sub	$7, %r8
	lea	7(%rdx), %rcx
	jnz	L(StrncpyFillTailWithZero1)
#   ifdef USE_AS_STPCPY
	cmpb	$1, (%rax)
	sbb	$-1, %rax
#   endif
#  endif
	ret

	.p2align 4
L(Exit9):
	mov	(%rcx), %rax
	mov	%rax, (%rdx)
	mov	5(%rcx), %eax
	mov	%eax, 5(%rdx)
#  ifdef USE_AS_STPCPY
	lea	8(%rdx), %rax
#  else
	mov	%rdi, %rax
#  endif
#  ifdef USE_AS_STRNCPY
	sub	$9, %r8
	lea	9(%rdx), %rcx
	jnz	L(StrncpyFillTailWithZero1)
#   ifdef USE_AS_STPCPY
	cmpb	$1, (%rax)
	sbb	$-1, %rax
#   endif
#  endif
	ret

	.p2align 4
L(Exit10):
	mov	(%rcx), %rax
	mov	%rax, (%rdx)
	mov	6(%rcx), %eax
	mov	%eax, 6(%rdx)
#  ifdef USE_AS_STPCPY
	lea	9(%rdx), %rax
#  else
	mov	%rdi, %rax
#  endif
#  ifdef USE_AS_STRNCPY
	sub	$10, %r8
	lea	10(%rdx), %rcx
	jnz	L(StrncpyFillTailWithZero1)
#   ifdef USE_AS_STPCPY
	cmpb	$1, (%rax)
	sbb	$-1, %rax
#   endif
#  endif
	ret

	.p2align 4
L(Exit11):
	mov	(%rcx), %rax
	mov	%rax, (%rdx)
	mov	7(%rcx), %eax
	mov	%eax, 7(%rdx)
#  ifdef USE_AS_STPCPY
	lea	10(%rdx), %rax
#  else
	mov	%rdi, %rax
#  endif
#  ifdef USE_AS_STRNCPY
	sub	$11, %r8
	lea	11(%rdx), %rcx
	jnz	L(StrncpyFillTailWithZero1)
#   ifdef USE_AS_STPCPY
	cmpb	$1, (%rax)
	sbb	$-1, %rax
#   endif
#  endif
	ret

	.p2align 4
L(Exit12):
	mov	(%rcx), %rax
	mov	%rax, (%rdx)
	mov	8(%rcx), %eax
	mov	%eax, 8(%rdx)
#  ifdef USE_AS_STPCPY
	lea	11(%rdx), %rax
#  else
	mov	%rdi, %rax
#  endif
#  ifdef USE_AS_STRNCPY
	sub	$12, %r8
	lea	12(%rdx), %rcx
	jnz	L(StrncpyFillTailWithZero1)
#   ifdef USE_AS_STPCPY
	cmpb	$1, (%rax)
	sbb	$-1, %rax
#  endif
#  endif
	ret

	.p2align 4
L(Exit13):
	mov	(%rcx), %rax
	mov	%rax, (%rdx)
	mov	5(%rcx), %rax
	mov	%rax, 5(%rdx)
#  ifdef USE_AS_STPCPY
	lea	12(%rdx), %rax
#  else
	mov	%rdi, %rax
#  endif
#  ifdef USE_AS_STRNCPY
	sub	$13, %r8
	lea	13(%rdx), %rcx
	jnz	L(StrncpyFillTailWithZero1)
#   ifdef USE_AS_STPCPY
	cmpb	$1, (%rax)
	sbb	$-1, %rax
#   endif
#  endif
	ret

	.p2align 4
L(Exit14):
	mov	(%rcx), %rax
	mov	%rax, (%rdx)
	mov	6(%rcx), %rax
	mov	%rax, 6(%rdx)
#  ifdef USE_AS_STPCPY
	lea	13(%rdx), %rax
#  else
	mov	%rdi, %rax
#  endif
#  ifdef USE_AS_STRNCPY
	sub	$14, %r8
	lea	14(%rdx), %rcx
	jnz	L(StrncpyFillTailWithZero1)
#   ifdef USE_AS_STPCPY
	cmpb	$1, (%rax)
	sbb	$-1, %rax
#   endif
#  endif
	ret

	.p2align 4
L(Exit15):
	mov	(%rcx), %rax
	mov	%rax, (%rdx)
	mov	7(%rcx), %rax
	mov	%rax, 7(%rdx)
#  ifdef USE_AS_STPCPY
	lea	14(%rdx), %rax
#  else
	mov	%rdi, %rax
#  endif
#  ifdef USE_AS_STRNCPY
	sub	$15, %r8
	lea	15(%rdx), %rcx
	jnz	L(StrncpyFillTailWithZero1)
#   ifdef USE_AS_STPCPY
	cmpb	$1, (%rax)
	sbb	$-1, %rax
#   endif
#  endif
	ret

#  ifdef USE_AS_STRNCPY
	.p2align 4
L(Fill0):
	ret

	.p2align 4
L(Fill1):
	movb	%dl, (%rcx)
	ret

	.p2align 4
L(Fill2):
	movw	%dx, (%rcx)
	ret

	.p2align 4
L(Fill3):
	movw	%dx, (%rcx)
	movb	%dl, 2(%rcx)
	ret

	.p2align 4
L(Fill4):
	movl	%edx, (%rcx)
	ret

	.p2align 4
L(Fill5):
	movl	%edx, (%rcx)
	movb	%dl, 4(%rcx)
	ret

	.p2align 4
L(Fill6):
	movl	%edx, (%rcx)
	movw	%dx, 4(%rcx)
	ret

	.p2align 4
L(Fill7):
	movl	%edx, (%rcx)
	movl	%edx, 3(%rcx)
	ret

	.p2align 4
L(Fill8):
	mov	%rdx, (%rcx)
	ret

	.p2align 4
L(Fill9):
	mov	%rdx, (%rcx)
	movb	%dl, 8(%rcx)
	ret

	.p2align 4
L(Fill10):
	mov	%rdx, (%rcx)
	movw	%dx, 8(%rcx)
	ret

	.p2align 4
L(Fill11):
	mov	%rdx, (%rcx)
	movl	%edx, 7(%rcx)
	ret

	.p2align 4
L(Fill12):
	mov	%rdx, (%rcx)
	movl	%edx, 8(%rcx)
	ret

	.p2align 4
L(Fill13):
	mov	%rdx, (%rcx)
	mov	%rdx, 5(%rcx)
	ret

	.p2align 4
L(Fill14):
	mov	%rdx, (%rcx)
	mov	%rdx, 6(%rcx)
	ret

	.p2align 4
L(Fill15):
	mov	%rdx, (%rcx)
	mov	%rdx, 7(%rcx)
	ret

	.p2align 4
L(Fill16):
	mov	%rdx, (%rcx)
	mov	%rdx, 8(%rcx)
	ret

	.p2align 4
L(StrncpyFillExit1):
	lea	16(%r8), %r8
L(FillFrom1To16Bytes):
	test	%r8, %r8
	jz	L(Fill0)
	cmp	$16, %r8
	je	L(Fill16)
	cmp	$8, %r8
	je	L(Fill8)
	jg	L(FillMore8)
	cmp	$4, %r8
	je	L(Fill4)
	jg	L(FillMore4)
	cmp	$2, %r8
	jl	L(Fill1)
	je	L(Fill2)
	jg	L(Fill3)
L(FillMore8): /* but less than 16 */
	cmp	$12, %r8
	je	L(Fill12)
	jl	L(FillLess12)
	cmp	$14, %r8
	jl	L(Fill13)
	je	L(Fill14)
	jg	L(Fill15)
L(FillMore4): /* but less than 8 */
	cmp	$6, %r8
	jl	L(Fill5)
	je	L(Fill6)
	jg	L(Fill7)
L(FillLess12): /* but more than 8 */
	cmp	$10, %r8
	jl	L(Fill9)
	je	L(Fill10)
	jmp	L(Fill11)

	.p2align 4
L(StrncpyFillTailWithZero1):
	xor	%rdx, %rdx
	sub	$16, %r8
	jbe	L(StrncpyFillExit1)

	pxor	%xmm0, %xmm0
	mov	%rdx, (%rcx)
	mov	%rdx, 8(%rcx)

	lea	16(%rcx), %rcx

	mov	%rcx, %rdx
	and	$0xf, %rdx
	sub	%rdx, %rcx
	add	%rdx, %r8
	xor	%rdx, %rdx
	sub	$64, %r8
	jb	L(StrncpyFillLess64)

L(StrncpyFillLoopMovdqa):
	movdqa	%xmm0, (%rcx)
	movdqa	%xmm0, 16(%rcx)
	movdqa	%xmm0, 32(%rcx)
	movdqa	%xmm0, 48(%rcx)
	lea	64(%rcx), %rcx
	sub	$64, %r8
	jae	L(StrncpyFillLoopMovdqa)

L(StrncpyFillLess64):
	add	$32, %r8
	jl	L(StrncpyFillLess32)
	movdqa	%xmm0, (%rcx)
	movdqa	%xmm0, 16(%rcx)
	lea	32(%rcx), %rcx
	sub	$16, %r8
	jl	L(StrncpyFillExit1)
	movdqa	%xmm0, (%rcx)
	lea	16(%rcx), %rcx
	jmp	L(FillFrom1To16Bytes)

L(StrncpyFillLess32):
	add	$16, %r8
	jl	L(StrncpyFillExit1)
	movdqa	%xmm0, (%rcx)
	lea	16(%rcx), %rcx
	jmp	L(FillFrom1To16Bytes)

	.p2align 4
L(Exit0):
	mov	%rdx, %rax
	ret

	.p2align 4
L(StrncpyExit15Bytes):
	cmp	$9, %r8
	je	L(Exit9)
	cmpb	$0, 8(%rcx)
	jz	L(Exit9)
	cmp	$10, %r8
	je	L(Exit10)
	cmpb	$0, 9(%rcx)
	jz	L(Exit10)
	cmp	$11, %r8
	je	L(Exit11)
	cmpb	$0, 10(%rcx)
	jz	L(Exit11)
	cmp	$12, %r8
	je	L(Exit12)
	cmpb	$0, 11(%rcx)
	jz	L(Exit12)
	cmp	$13, %r8
	je	L(Exit13)
	cmpb	$0, 12(%rcx)
	jz	L(Exit13)
	cmp	$14, %r8
	je	L(Exit14)
	cmpb	$0, 13(%rcx)
	jz	L(Exit14)
	mov	(%rcx), %rax
	mov	%rax, (%rdx)
	mov	7(%rcx), %rax
	mov	%rax, 7(%rdx)
#   ifdef USE_AS_STPCPY
	lea	14(%rdx), %rax
	cmpb	$1, (%rax)
	sbb	$-1, %rax
#   else
	mov	%rdi, %rax
#   endif
	ret

	.p2align 4
L(StrncpyExit8Bytes):
	cmp	$1, %r8
	je	L(Exit1)
	cmpb	$0, (%rcx)
	jz	L(Exit1)
	cmp	$2, %r8
	je	L(Exit2)
	cmpb	$0, 1(%rcx)
	jz	L(Exit2)
	cmp	$3, %r8
	je	L(Exit3)
	cmpb	$0, 2(%rcx)
	jz	L(Exit3)
	cmp	$4, %r8
	je	L(Exit4)
	cmpb	$0, 3(%rcx)
	jz	L(Exit4)
	cmp	$5, %r8
	je	L(Exit5)
	cmpb	$0, 4(%rcx)
	jz	L(Exit5)
	cmp	$6, %r8
	je	L(Exit6)
	cmpb	$0, 5(%rcx)
	jz	L(Exit6)
	cmp	$7, %r8
	je	L(Exit7)
	cmpb	$0, 6(%rcx)
	jz	L(Exit7)
	mov	(%rcx), %rax
	mov	%rax, (%rdx)
#   ifdef USE_AS_STPCPY
	lea	7(%rdx), %rax
	cmpb	$1, (%rax)
	sbb	$-1, %rax
#   else
	mov	%rdi, %rax
#   endif
	ret

#  endif
# endif

# ifdef USE_AS_STRNCPY
	.p2align 4
L(StrncpyLeaveCase2OrCase3):
	test	%rax, %rax
	jnz	L(Aligned64LeaveCase2)

L(Aligned64LeaveCase3):
	lea	64(%r8), %r8
	sub	$16, %r8
	jbe	L(CopyFrom1To16BytesCase3)
	movaps	%xmm4, -64(%rdx)
	lea	16(%rsi), %rsi
	sub	$16, %r8
	jbe	L(CopyFrom1To16BytesCase3)
	movaps	%xmm5, -48(%rdx)
	lea	16(%rsi), %rsi
	sub	$16, %r8
	jbe	L(CopyFrom1To16BytesCase3)
	movaps	%xmm6, -32(%rdx)
	lea	16(%rsi), %rsi
	lea	-16(%r8), %r8
	jmp	L(CopyFrom1To16BytesCase3)

L(Aligned64LeaveCase2):
	pcmpeqb	%xmm4, %xmm0
	pmovmskb %xmm0, %rax
	add	$48, %r8
	jle	L(CopyFrom1To16BytesCase2OrCase3)
	test	%rax, %rax
	jnz	L(CopyFrom1To16Bytes)

	pcmpeqb	%xmm5, %xmm0
	pmovmskb %xmm0, %rax
	movaps	%xmm4, -64(%rdx)
	lea	16(%rsi), %rsi
	sub	$16, %r8
	jbe	L(CopyFrom1To16BytesCase2OrCase3)
	test	%rax, %rax
	jnz	L(CopyFrom1To16Bytes)

	pcmpeqb	%xmm6, %xmm0
	pmovmskb %xmm0, %rax
	movaps	%xmm5, -48(%rdx)
	lea	16(%rsi), %rsi
	sub	$16, %r8
	jbe	L(CopyFrom1To16BytesCase2OrCase3)
	test	%rax, %rax
	jnz	L(CopyFrom1To16Bytes)

	pcmpeqb	%xmm7, %xmm0
	pmovmskb %xmm0, %rax
	movaps	%xmm6, -32(%rdx)
	lea	16(%rsi), %rsi
	lea	-16(%r8), %r8
	jmp	L(CopyFrom1To16BytesCase2)
/*--------------------------------------------------*/
	.p2align 4
L(StrncpyExit1Case2OrCase3):
	movdqu	-1(%rcx), %xmm0
	movdqu	%xmm0, -1(%rdx)
	mov	$15, %rsi
	test	%rax, %rax
	jnz	L(CopyFrom1To16BytesCase2)
	jmp	L(CopyFrom1To16BytesCase3)

	.p2align 4
L(StrncpyExit2Case2OrCase3):
	movdqu	-2(%rcx), %xmm0
	movdqu	%xmm0, -2(%rdx)
	mov	$14, %rsi
	test	%rax, %rax
	jnz	L(CopyFrom1To16BytesCase2)
	jmp	L(CopyFrom1To16BytesCase3)

	.p2align 4
L(StrncpyExit3Case2OrCase3):
	movdqu	-3(%rcx), %xmm0
	movdqu	%xmm0, -3(%rdx)
	mov	$13, %rsi
	test	%rax, %rax
	jnz	L(CopyFrom1To16BytesCase2)
	jmp	L(CopyFrom1To16BytesCase3)

	.p2align 4
L(StrncpyExit4Case2OrCase3):
	movdqu	-4(%rcx), %xmm0
	movdqu	%xmm0, -4(%rdx)
	mov	$12, %rsi
	test	%rax, %rax
	jnz	L(CopyFrom1To16BytesCase2)
	jmp	L(CopyFrom1To16BytesCase3)

	.p2align 4
L(StrncpyExit5Case2OrCase3):
	movdqu	-5(%rcx), %xmm0
	movdqu	%xmm0, -5(%rdx)
	mov	$11, %rsi
	test	%rax, %rax
	jnz	L(CopyFrom1To16BytesCase2)
	jmp	L(CopyFrom1To16BytesCase3)

	.p2align 4
L(StrncpyExit6Case2OrCase3):
	mov	(%rcx), %rsi
	mov	6(%rcx), %r9d
	mov	%r9d, 6(%rdx)
	mov	%rsi, (%rdx)
	test	%rax, %rax
	mov	$10, %rsi
	jnz	L(CopyFrom1To16BytesCase2)
	jmp	L(CopyFrom1To16BytesCase3)

	.p2align 4
L(StrncpyExit7Case2OrCase3):
	mov	(%rcx), %rsi
	mov	5(%rcx), %r9d
	mov	%r9d, 5(%rdx)
	mov	%rsi, (%rdx)
	test	%rax, %rax
	mov	$9, %rsi
	jnz	L(CopyFrom1To16BytesCase2)
	jmp	L(CopyFrom1To16BytesCase3)

	.p2align 4
L(StrncpyExit8Case2OrCase3):
	mov	(%rcx), %r9
	mov	$8, %rsi
	mov	%r9, (%rdx)
	test	%rax, %rax
	jnz	L(CopyFrom1To16BytesCase2)
	jmp	L(CopyFrom1To16BytesCase3)

	.p2align 4
L(StrncpyExit9Case2OrCase3):
	mov	-1(%rcx), %r9
	mov	$7, %rsi
	mov	%r9, -1(%rdx)
	test	%rax, %rax
	jnz	L(CopyFrom1To16BytesCase2)
	jmp	L(CopyFrom1To16BytesCase3)

	.p2align 4
L(StrncpyExit10Case2OrCase3):
	mov	-2(%rcx), %r9
	mov	$6, %rsi
	mov	%r9, -2(%rdx)
	test	%rax, %rax
	jnz	L(CopyFrom1To16BytesCase2)
	jmp	L(CopyFrom1To16BytesCase3)

	.p2align 4
L(StrncpyExit11Case2OrCase3):
	mov	-3(%rcx), %r9
	mov	$5, %rsi
	mov	%r9, -3(%rdx)
	test	%rax, %rax
	jnz	L(CopyFrom1To16BytesCase2)
	jmp	L(CopyFrom1To16BytesCase3)

	.p2align 4
L(StrncpyExit12Case2OrCase3):
	mov	(%rcx), %r9d
	mov	$4, %rsi
	mov	%r9d, (%rdx)
	test	%rax, %rax
	jnz	L(CopyFrom1To16BytesCase2)
	jmp	L(CopyFrom1To16BytesCase3)

	.p2align 4
L(StrncpyExit13Case2OrCase3):
	mov	-1(%rcx), %r9d
	mov	$3, %rsi
	mov	%r9d, -1(%rdx)
	test	%rax, %rax
	jnz	L(CopyFrom1To16BytesCase2)
	jmp	L(CopyFrom1To16BytesCase3)

	.p2align 4
L(StrncpyExit14Case2OrCase3):
	mov	-2(%rcx), %r9d
	mov	$2, %rsi
	mov	%r9d, -2(%rdx)
	test	%rax, %rax
	jnz	L(CopyFrom1To16BytesCase2)
	jmp	L(CopyFrom1To16BytesCase3)

	.p2align 4
L(StrncpyExit15Case2OrCase3):
	mov	-3(%rcx), %r9d
	mov	$1, %rsi
	mov	%r9d, -3(%rdx)
	test	%rax, %rax
	jnz	L(CopyFrom1To16BytesCase2)
	jmp	L(CopyFrom1To16BytesCase3)

	.p2align 4
L(StrncpyLeave1):
	movaps	%xmm2, %xmm3
	add	$48, %r8
	jle	L(StrncpyExit1)
	palignr	$1, %xmm1, %xmm2
	movaps	%xmm2, (%rdx)
	movaps	31(%rcx), %xmm2
	lea	16(%rsi), %rsi
	sub	$16, %r8
	jbe	L(StrncpyExit1)
	palignr	$1, %xmm3, %xmm2
	movaps	%xmm2, 16(%rdx)
	lea	16(%rsi), %rsi
	sub	$16, %r8
	jbe	L(StrncpyExit1)
	movaps	%xmm4, 32(%rdx)
	lea	16(%rsi), %rsi
	sub	$16, %r8
	jbe	L(StrncpyExit1)
	movaps	%xmm5, 48(%rdx)
	lea	16(%rsi), %rsi
	lea	-16(%r8), %r8

L(StrncpyExit1):
	lea	15(%rdx, %rsi), %rdx
	lea	15(%rcx, %rsi), %rcx
	mov	-15(%rcx), %rsi
	mov	-8(%rcx), %rax
	mov	%rsi, -15(%rdx)
	mov	%rax, -8(%rdx)
	xor	%rsi, %rsi
	jmp	L(CopyFrom1To16BytesCase3)

	.p2align 4
L(StrncpyLeave2):
	movaps	%xmm2, %xmm3
	add	$48, %r8
	jle	L(StrncpyExit2)
	palignr	$2, %xmm1, %xmm2
	movaps	%xmm2, (%rdx)
	movaps	30(%rcx), %xmm2
	lea	16(%rsi), %rsi
	sub	$16, %r8
	jbe	L(StrncpyExit2)
	palignr	$2, %xmm3, %xmm2
	movaps	%xmm2, 16(%rdx)
	lea	16(%rsi), %rsi
	sub	$16, %r8
	jbe	L(StrncpyExit2)
	movaps	%xmm4, 32(%rdx)
	lea	16(%rsi), %rsi
	sub	$16, %r8
	jbe	L(StrncpyExit2)
	movaps	%xmm5, 48(%rdx)
	lea	16(%rsi), %rsi
	lea	-16(%r8), %r8

L(StrncpyExit2):
	lea	14(%rdx, %rsi), %rdx
	lea	14(%rcx, %rsi), %rcx
	mov	-14(%rcx), %rsi
	mov	-8(%rcx), %rax
	mov	%rsi, -14(%rdx)
	mov	%rax, -8(%rdx)
	xor	%rsi, %rsi
	jmp	L(CopyFrom1To16BytesCase3)

	.p2align 4
L(StrncpyLeave3):
	movaps	%xmm2, %xmm3
	add	$48, %r8
	jle	L(StrncpyExit3)
	palignr	$3, %xmm1, %xmm2
	movaps	%xmm2, (%rdx)
	movaps	29(%rcx), %xmm2
	lea	16(%rsi), %rsi
	sub	$16, %r8
	jbe	L(StrncpyExit3)
	palignr	$3, %xmm3, %xmm2
	movaps	%xmm2, 16(%rdx)
	lea	16(%rsi), %rsi
	sub	$16, %r8
	jbe	L(StrncpyExit3)
	movaps	%xmm4, 32(%rdx)
	lea	16(%rsi), %rsi
	sub	$16, %r8
	jbe	L(StrncpyExit3)
	movaps	%xmm5, 48(%rdx)
	lea	16(%rsi), %rsi
	lea	-16(%r8), %r8

L(StrncpyExit3):
	lea	13(%rdx, %rsi), %rdx
	lea	13(%rcx, %rsi), %rcx
	mov	-13(%rcx), %rsi
	mov	-8(%rcx), %rax
	mov	%rsi, -13(%rdx)
	mov	%rax, -8(%rdx)
	xor	%rsi, %rsi
	jmp	L(CopyFrom1To16BytesCase3)

	.p2align 4
L(StrncpyLeave4):
	movaps	%xmm2, %xmm3
	add	$48, %r8
	jle	L(StrncpyExit4)
	palignr	$4, %xmm1, %xmm2
	movaps	%xmm2, (%rdx)
	movaps	28(%rcx), %xmm2
	lea	16(%rsi), %rsi
	sub	$16, %r8
	jbe	L(StrncpyExit4)
	palignr	$4, %xmm3, %xmm2
	movaps	%xmm2, 16(%rdx)
	lea	16(%rsi), %rsi
	sub	$16, %r8
	jbe	L(StrncpyExit4)
	movaps	%xmm4, 32(%rdx)
	lea	16(%rsi), %rsi
	sub	$16, %r8
	jbe	L(StrncpyExit4)
	movaps	%xmm5, 48(%rdx)
	lea	16(%rsi), %rsi
	lea	-16(%r8), %r8

L(StrncpyExit4):
	lea	12(%rdx, %rsi), %rdx
	lea	12(%rcx, %rsi), %rcx
	mov	-12(%rcx), %rsi
	mov	-4(%rcx), %eax
	mov	%rsi, -12(%rdx)
	mov	%eax, -4(%rdx)
	xor	%rsi, %rsi
	jmp	L(CopyFrom1To16BytesCase3)

	.p2align 4
L(StrncpyLeave5):
	movaps	%xmm2, %xmm3
	add	$48, %r8
	jle	L(StrncpyExit5)
	palignr	$5, %xmm1, %xmm2
	movaps	%xmm2, (%rdx)
	movaps	27(%rcx), %xmm2
	lea	16(%rsi), %rsi
	sub	$16, %r8
	jbe	L(StrncpyExit5)
	palignr	$5, %xmm3, %xmm2
	movaps	%xmm2, 16(%rdx)
	lea	16(%rsi), %rsi
	sub	$16, %r8
	jbe	L(StrncpyExit5)
	movaps	%xmm4, 32(%rdx)
	lea	16(%rsi), %rsi
	sub	$16, %r8
	jbe	L(StrncpyExit5)
	movaps	%xmm5, 48(%rdx)
	lea	16(%rsi), %rsi
	lea	-16(%r8), %r8

L(StrncpyExit5):
	lea	11(%rdx, %rsi), %rdx
	lea	11(%rcx, %rsi), %rcx
	mov	-11(%rcx), %rsi
	mov	-4(%rcx), %eax
	mov	%rsi, -11(%rdx)
	mov	%eax, -4(%rdx)
	xor	%rsi, %rsi
	jmp	L(CopyFrom1To16BytesCase3)

	.p2align 4
L(StrncpyLeave6):
	movaps	%xmm2, %xmm3
	add	$48, %r8
	jle	L(StrncpyExit6)
	palignr	$6, %xmm1, %xmm2
	movaps	%xmm2, (%rdx)
	movaps	26(%rcx), %xmm2
	lea	16(%rsi), %rsi
	sub	$16, %r8
	jbe	L(StrncpyExit6)
	palignr	$6, %xmm3, %xmm2
	movaps	%xmm2, 16(%rdx)
	lea	16(%rsi), %rsi
	sub	$16, %r8
	jbe	L(StrncpyExit6)
	movaps	%xmm4, 32(%rdx)
	lea	16(%rsi), %rsi
	sub	$16, %r8
	jbe	L(StrncpyExit6)
	movaps	%xmm5, 48(%rdx)
	lea	16(%rsi), %rsi
	lea	-16(%r8), %r8

L(StrncpyExit6):
	lea	10(%rdx, %rsi), %rdx
	lea	10(%rcx, %rsi), %rcx
	mov	-10(%rcx), %rsi
	movw	-2(%rcx), %ax
	mov	%rsi, -10(%rdx)
	movw	%ax, -2(%rdx)
	xor	%rsi, %rsi
	jmp	L(CopyFrom1To16BytesCase3)

	.p2align 4
L(StrncpyLeave7):
	movaps	%xmm2, %xmm3
	add	$48, %r8
	jle	L(StrncpyExit7)
	palignr	$7, %xmm1, %xmm2
	movaps	%xmm2, (%rdx)
	movaps	25(%rcx), %xmm2
	lea	16(%rsi), %rsi
	sub	$16, %r8
	jbe	L(StrncpyExit7)
	palignr	$7, %xmm3, %xmm2
	movaps	%xmm2, 16(%rdx)
	lea	16(%rsi), %rsi
	sub	$16, %r8
	jbe	L(StrncpyExit7)
	movaps	%xmm4, 32(%rdx)
	lea	16(%rsi), %rsi
	sub	$16, %r8
	jbe	L(StrncpyExit7)
	movaps	%xmm5, 48(%rdx)
	lea	16(%rsi), %rsi
	lea	-16(%r8), %r8

L(StrncpyExit7):
	lea	9(%rdx, %rsi), %rdx
	lea	9(%rcx, %rsi), %rcx
	mov	-9(%rcx), %rsi
	movb	-1(%rcx), %ah
	mov	%rsi, -9(%rdx)
	movb	%ah, -1(%rdx)
	xor	%rsi, %rsi
	jmp	L(CopyFrom1To16BytesCase3)

	.p2align 4
L(StrncpyLeave8):
	movaps	%xmm2, %xmm3
	add	$48, %r8
	jle	L(StrncpyExit8)
	palignr	$8, %xmm1, %xmm2
	movaps	%xmm2, (%rdx)
	movaps	24(%rcx), %xmm2
	lea	16(%rsi), %rsi
	sub	$16, %r8
	jbe	L(StrncpyExit8)
	palignr	$8, %xmm3, %xmm2
	movaps	%xmm2, 16(%rdx)
	lea	16(%rsi), %rsi
	sub	$16, %r8
	jbe	L(StrncpyExit8)
	movaps	%xmm4, 32(%rdx)
	lea	16(%rsi), %rsi
	sub	$16, %r8
	jbe	L(StrncpyExit8)
	movaps	%xmm5, 48(%rdx)
	lea	16(%rsi), %rsi
	lea	-16(%r8), %r8

L(StrncpyExit8):
	lea	8(%rdx, %rsi), %rdx
	lea	8(%rcx, %rsi), %rcx
	mov	-8(%rcx), %rax
	xor	%rsi, %rsi
	mov	%rax, -8(%rdx)
	jmp	L(CopyFrom1To16BytesCase3)

	.p2align 4
L(StrncpyLeave9):
	movaps	%xmm2, %xmm3
	add	$48, %r8
	jle	L(StrncpyExit9)
	palignr	$9, %xmm1, %xmm2
	movaps	%xmm2, (%rdx)
	movaps	23(%rcx), %xmm2
	lea	16(%rsi), %rsi
	sub	$16, %r8
	jbe	L(StrncpyExit9)
	palignr	$9, %xmm3, %xmm2
	movaps	%xmm2, 16(%rdx)
	lea	16(%rsi), %rsi
	sub	$16, %r8
	jbe	L(StrncpyExit9)
	movaps	%xmm4, 32(%rdx)
	lea	16(%rsi), %rsi
	sub	$16, %r8
	jbe	L(StrncpyExit9)
	movaps	%xmm5, 48(%rdx)
	lea	16(%rsi), %rsi
	lea	-16(%r8), %r8

L(StrncpyExit9):
	lea	7(%rdx, %rsi), %rdx
	lea	7(%rcx, %rsi), %rcx
	mov	-8(%rcx), %rax
	xor	%rsi, %rsi
	mov	%rax, -8(%rdx)
	jmp	L(CopyFrom1To16BytesCase3)

	.p2align 4
L(StrncpyLeave10):
	movaps	%xmm2, %xmm3
	add	$48, %r8
	jle	L(StrncpyExit10)
	palignr	$10, %xmm1, %xmm2
	movaps	%xmm2, (%rdx)
	movaps	22(%rcx), %xmm2
	lea	16(%rsi), %rsi
	sub	$16, %r8
	jbe	L(StrncpyExit10)
	palignr	$10, %xmm3, %xmm2
	movaps	%xmm2, 16(%rdx)
	lea	16(%rsi), %rsi
	sub	$16, %r8
	jbe	L(StrncpyExit10)
	movaps	%xmm4, 32(%rdx)
	lea	16(%rsi), %rsi
	sub	$16, %r8
	jbe	L(StrncpyExit10)
	movaps	%xmm5, 48(%rdx)
	lea	16(%rsi), %rsi
	lea	-16(%r8), %r8

L(StrncpyExit10):
	lea	6(%rdx, %rsi), %rdx
	lea	6(%rcx, %rsi), %rcx
	mov	-8(%rcx), %rax
	xor	%rsi, %rsi
	mov	%rax, -8(%rdx)
	jmp	L(CopyFrom1To16BytesCase3)

	.p2align 4
L(StrncpyLeave11):
	movaps	%xmm2, %xmm3
	add	$48, %r8
	jle	L(StrncpyExit11)
	palignr	$11, %xmm1, %xmm2
	movaps	%xmm2, (%rdx)
	movaps	21(%rcx), %xmm2
	lea	16(%rsi), %rsi
	sub	$16, %r8
	jbe	L(StrncpyExit11)
	palignr	$11, %xmm3, %xmm2
	movaps	%xmm2, 16(%rdx)
	lea	16(%rsi), %rsi
	sub	$16, %r8
	jbe	L(StrncpyExit11)
	movaps	%xmm4, 32(%rdx)
	lea	16(%rsi), %rsi
	sub	$16, %r8
	jbe	L(StrncpyExit11)
	movaps	%xmm5, 48(%rdx)
	lea	16(%rsi), %rsi
	lea	-16(%r8), %r8

L(StrncpyExit11):
	lea	5(%rdx, %rsi), %rdx
	lea	5(%rcx, %rsi), %rcx
	mov	-8(%rcx), %rax
	xor	%rsi, %rsi
	mov	%rax, -8(%rdx)
	jmp	L(CopyFrom1To16BytesCase3)

	.p2align 4
L(StrncpyLeave12):
	movaps	%xmm2, %xmm3
	add	$48, %r8
	jle	L(StrncpyExit12)
	palignr	$12, %xmm1, %xmm2
	movaps	%xmm2, (%rdx)
	movaps	20(%rcx), %xmm2
	lea	16(%rsi), %rsi
	sub	$16, %r8
	jbe	L(StrncpyExit12)
	palignr	$12, %xmm3, %xmm2
	movaps	%xmm2, 16(%rdx)
	lea	16(%rsi), %rsi
	sub	$16, %r8
	jbe	L(StrncpyExit12)
	movaps	%xmm4, 32(%rdx)
	lea	16(%rsi), %rsi
	sub	$16, %r8
	jbe	L(StrncpyExit12)
	movaps	%xmm5, 48(%rdx)
	lea	16(%rsi), %rsi
	lea	-16(%r8), %r8

L(StrncpyExit12):
	lea	4(%rdx, %rsi), %rdx
	lea	4(%rcx, %rsi), %rcx
	mov	-4(%rcx), %eax
	xor	%rsi, %rsi
	mov	%eax, -4(%rdx)
	jmp	L(CopyFrom1To16BytesCase3)

	.p2align 4
L(StrncpyLeave13):
	movaps	%xmm2, %xmm3
	add	$48, %r8
	jle	L(StrncpyExit13)
	palignr	$13, %xmm1, %xmm2
	movaps	%xmm2, (%rdx)
	movaps	19(%rcx), %xmm2
	lea	16(%rsi), %rsi
	sub	$16, %r8
	jbe	L(StrncpyExit13)
	palignr	$13, %xmm3, %xmm2
	movaps	%xmm2, 16(%rdx)
	lea	16(%rsi), %rsi
	sub	$16, %r8
	jbe	L(StrncpyExit13)
	movaps	%xmm4, 32(%rdx)
	lea	16(%rsi), %rsi
	sub	$16, %r8
	jbe	L(StrncpyExit13)
	movaps	%xmm5, 48(%rdx)
	lea	16(%rsi), %rsi
	lea	-16(%r8), %r8

L(StrncpyExit13):
	lea	3(%rdx, %rsi), %rdx
	lea	3(%rcx, %rsi), %rcx
	mov	-4(%rcx), %eax
	xor	%rsi, %rsi
	mov	%eax, -4(%rdx)
	jmp	L(CopyFrom1To16BytesCase3)

	.p2align 4
L(StrncpyLeave14):
	movaps	%xmm2, %xmm3
	add	$48, %r8
	jle	L(StrncpyExit14)
	palignr	$14, %xmm1, %xmm2
	movaps	%xmm2, (%rdx)
	movaps	18(%rcx), %xmm2
	lea	16(%rsi), %rsi
	sub	$16, %r8
	jbe	L(StrncpyExit14)
	palignr	$14, %xmm3, %xmm2
	movaps	%xmm2, 16(%rdx)
	lea	16(%rsi), %rsi
	sub	$16, %r8
	jbe	L(StrncpyExit14)
	movaps	%xmm4, 32(%rdx)
	lea	16(%rsi), %rsi
	sub	$16, %r8
	jbe	L(StrncpyExit14)
	movaps	%xmm5, 48(%rdx)
	lea	16(%rsi), %rsi
	lea	-16(%r8), %r8

L(StrncpyExit14):
	lea	2(%rdx, %rsi), %rdx
	lea	2(%rcx, %rsi), %rcx
	movw	-2(%rcx), %ax
	xor	%rsi, %rsi
	movw	%ax, -2(%rdx)
	jmp	L(CopyFrom1To16BytesCase3)

	.p2align 4
L(StrncpyLeave15):
	movaps	%xmm2, %xmm3
	add	$48, %r8
	jle	L(StrncpyExit15)
	palignr	$15, %xmm1, %xmm2
	movaps	%xmm2, (%rdx)
	movaps	17(%rcx), %xmm2
	lea	16(%rsi), %rsi
	sub	$16, %r8
	jbe	L(StrncpyExit15)
	palignr	$15, %xmm3, %xmm2
	movaps	%xmm2, 16(%rdx)
	lea	16(%rsi), %rsi
	sub	$16, %r8
	jbe	L(StrncpyExit15)
	movaps	%xmm4, 32(%rdx)
	lea	16(%rsi), %rsi
	sub	$16, %r8
	jbe	L(StrncpyExit15)
	movaps	%xmm5, 48(%rdx)
	lea	16(%rsi), %rsi
	lea	-16(%r8), %r8

L(StrncpyExit15):
	lea	1(%rdx, %rsi), %rdx
	lea	1(%rcx, %rsi), %rcx
	movb	-1(%rcx), %ah
	xor	%rsi, %rsi
	movb	%ah, -1(%rdx)
	jmp	L(CopyFrom1To16BytesCase3)

# endif
# ifndef USE_AS_STRCAT
END (STRCPY)
# endif
#endif
