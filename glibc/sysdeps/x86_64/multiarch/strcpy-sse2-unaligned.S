/* strcpy with SSE2 and unaligned load
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
#   define STRCPY  __strcpy_sse2_unaligned
#  endif

# endif

# define JMPTBL(I, B)	I - B
# define BRANCH_TO_JMPTBL_ENTRY(TABLE, INDEX, SCALE)             \
	lea	TABLE(%rip), %r11;                              \
	movslq	(%r11, INDEX, SCALE), %rcx;                     \
	lea	(%r11, %rcx), %rcx;                             \
	_CET_NOTRACK jmp *%rcx

# ifndef USE_AS_STRCAT

.text
ENTRY (STRCPY)
#  ifdef USE_AS_STRNCPY
	mov	%RDX_LP, %R8_LP
	test	%R8_LP, %R8_LP
	jz	L(ExitZero)
#  endif
	mov	%rsi, %rcx
#  ifndef USE_AS_STPCPY
	mov	%rdi, %rax      /* save result */
#  endif

# endif

	and	$63, %rcx
	cmp	$32, %rcx
	jbe	L(SourceStringAlignmentLess32)

	and	$-16, %rsi
	and	$15, %rcx
	pxor	%xmm0, %xmm0
	pxor	%xmm1, %xmm1

	pcmpeqb	(%rsi), %xmm1
	pmovmskb %xmm1, %rdx
	shr	%cl, %rdx

# ifdef USE_AS_STRNCPY
#  if defined USE_AS_STPCPY || defined USE_AS_STRCAT
	mov	$16, %r10
	sub	%rcx, %r10
	cmp	%r10, %r8
#  else
	mov	$17, %r10
	sub	%rcx, %r10
	cmp	%r10, %r8
#  endif
	jbe	L(CopyFrom1To16BytesTailCase2OrCase3)
# endif
	test	%rdx, %rdx
	jnz	L(CopyFrom1To16BytesTail)

	pcmpeqb	16(%rsi), %xmm0
	pmovmskb %xmm0, %rdx

# ifdef USE_AS_STRNCPY
	add	$16, %r10
	cmp	%r10, %r8
	jbe	L(CopyFrom1To32BytesCase2OrCase3)
# endif
	test	%rdx, %rdx
	jnz	L(CopyFrom1To32Bytes)

	movdqu	(%rsi, %rcx), %xmm1   /* copy 16 bytes */
	movdqu	%xmm1, (%rdi)

/* If source address alignment != destination address alignment */
	.p2align 4
L(Unalign16Both):
	sub	%rcx, %rdi
# ifdef USE_AS_STRNCPY
	add	%rcx, %r8
	sbb	%rcx, %rcx
	or	%rcx, %r8
# endif
	mov	$16, %rcx
	movdqa	(%rsi, %rcx), %xmm1
	movaps	16(%rsi, %rcx), %xmm2
	movdqu	%xmm1, (%rdi, %rcx)
	pcmpeqb	%xmm2, %xmm0
	pmovmskb %xmm0, %rdx
	add	$16, %rcx
# ifdef USE_AS_STRNCPY
	sub	$48, %r8
	jbe	L(CopyFrom1To16BytesCase2OrCase3)
# endif
	test	%rdx, %rdx
# if defined USE_AS_STRNCPY && !defined USE_AS_STRCAT
	jnz	L(CopyFrom1To16BytesUnalignedXmm2)
# else
	jnz	L(CopyFrom1To16Bytes)
# endif

	movaps	16(%rsi, %rcx), %xmm3
	movdqu	%xmm2, (%rdi, %rcx)
	pcmpeqb	%xmm3, %xmm0
	pmovmskb %xmm0, %rdx
	add	$16, %rcx
# ifdef USE_AS_STRNCPY
	sub	$16, %r8
	jbe	L(CopyFrom1To16BytesCase2OrCase3)
# endif
	test	%rdx, %rdx
# if defined USE_AS_STRNCPY && !defined USE_AS_STRCAT
	jnz	L(CopyFrom1To16BytesUnalignedXmm3)
# else
	jnz	L(CopyFrom1To16Bytes)
# endif

	movaps	16(%rsi, %rcx), %xmm4
	movdqu	%xmm3, (%rdi, %rcx)
	pcmpeqb	%xmm4, %xmm0
	pmovmskb %xmm0, %rdx
	add	$16, %rcx
# ifdef USE_AS_STRNCPY
	sub	$16, %r8
	jbe	L(CopyFrom1To16BytesCase2OrCase3)
# endif
	test	%rdx, %rdx
# if defined USE_AS_STRNCPY && !defined USE_AS_STRCAT
	jnz	L(CopyFrom1To16BytesUnalignedXmm4)
# else
	jnz	L(CopyFrom1To16Bytes)
# endif

	movaps	16(%rsi, %rcx), %xmm1
	movdqu	%xmm4, (%rdi, %rcx)
	pcmpeqb	%xmm1, %xmm0
	pmovmskb %xmm0, %rdx
	add	$16, %rcx
# ifdef USE_AS_STRNCPY
	sub	$16, %r8
	jbe	L(CopyFrom1To16BytesCase2OrCase3)
# endif
	test	%rdx, %rdx
# if defined USE_AS_STRNCPY && !defined USE_AS_STRCAT
	jnz	L(CopyFrom1To16BytesUnalignedXmm1)
# else
	jnz	L(CopyFrom1To16Bytes)
# endif

	movaps	16(%rsi, %rcx), %xmm2
	movdqu	%xmm1, (%rdi, %rcx)
	pcmpeqb	%xmm2, %xmm0
	pmovmskb %xmm0, %rdx
	add	$16, %rcx
# ifdef USE_AS_STRNCPY
	sub	$16, %r8
	jbe	L(CopyFrom1To16BytesCase2OrCase3)
# endif
	test	%rdx, %rdx
# if defined USE_AS_STRNCPY && !defined USE_AS_STRCAT
	jnz	L(CopyFrom1To16BytesUnalignedXmm2)
# else
	jnz	L(CopyFrom1To16Bytes)
# endif

	movaps	16(%rsi, %rcx), %xmm3
	movdqu	%xmm2, (%rdi, %rcx)
	pcmpeqb	%xmm3, %xmm0
	pmovmskb %xmm0, %rdx
	add	$16, %rcx
# ifdef USE_AS_STRNCPY
	sub	$16, %r8
	jbe	L(CopyFrom1To16BytesCase2OrCase3)
# endif
	test	%rdx, %rdx
# if defined USE_AS_STRNCPY && !defined USE_AS_STRCAT
	jnz	L(CopyFrom1To16BytesUnalignedXmm3)
# else
	jnz	L(CopyFrom1To16Bytes)
# endif

	movdqu	%xmm3, (%rdi, %rcx)
	mov	%rsi, %rdx
	lea	16(%rsi, %rcx), %rsi
	and	$-0x40, %rsi
	sub	%rsi, %rdx
	sub	%rdx, %rdi
# ifdef USE_AS_STRNCPY
	lea	128(%r8, %rdx), %r8
# endif
L(Unaligned64Loop):
	movaps	(%rsi), %xmm2
	movaps	%xmm2, %xmm4
	movaps	16(%rsi), %xmm5
	movaps	32(%rsi), %xmm3
	movaps	%xmm3, %xmm6
	movaps	48(%rsi), %xmm7
	pminub	%xmm5, %xmm2
	pminub	%xmm7, %xmm3
	pminub	%xmm2, %xmm3
	pcmpeqb	%xmm0, %xmm3
	pmovmskb %xmm3, %rdx
# ifdef USE_AS_STRNCPY
	sub	$64, %r8
	jbe	L(UnalignedLeaveCase2OrCase3)
# endif
	test	%rdx, %rdx
	jnz	L(Unaligned64Leave)

L(Unaligned64Loop_start):
	add	$64, %rdi
	add	$64, %rsi
	movdqu	%xmm4, -64(%rdi)
	movaps	(%rsi), %xmm2
	movdqa	%xmm2, %xmm4
	movdqu	%xmm5, -48(%rdi)
	movaps	16(%rsi), %xmm5
	pminub	%xmm5, %xmm2
	movaps	32(%rsi), %xmm3
	movdqu	%xmm6, -32(%rdi)
	movaps	%xmm3, %xmm6
	movdqu	%xmm7, -16(%rdi)
	movaps	48(%rsi), %xmm7
	pminub	%xmm7, %xmm3
	pminub	%xmm2, %xmm3
	pcmpeqb	%xmm0, %xmm3
	pmovmskb %xmm3, %rdx
# ifdef USE_AS_STRNCPY
	sub	$64, %r8
	jbe	L(UnalignedLeaveCase2OrCase3)
# endif
	test	%rdx, %rdx
	jz	L(Unaligned64Loop_start)

L(Unaligned64Leave):
	pxor	%xmm1, %xmm1

	pcmpeqb	%xmm4, %xmm0
	pcmpeqb	%xmm5, %xmm1
	pmovmskb %xmm0, %rdx
	pmovmskb %xmm1, %rcx
	test	%rdx, %rdx
	jnz	L(CopyFrom1To16BytesUnaligned_0)
	test	%rcx, %rcx
	jnz	L(CopyFrom1To16BytesUnaligned_16)

	pcmpeqb	%xmm6, %xmm0
	pcmpeqb	%xmm7, %xmm1
	pmovmskb %xmm0, %rdx
	pmovmskb %xmm1, %rcx
	test	%rdx, %rdx
	jnz	L(CopyFrom1To16BytesUnaligned_32)

	bsf	%rcx, %rdx
	movdqu	%xmm4, (%rdi)
	movdqu	%xmm5, 16(%rdi)
	movdqu	%xmm6, 32(%rdi)
# if defined USE_AS_STRNCPY && !defined USE_AS_STRCAT
# ifdef USE_AS_STPCPY
	lea	48(%rdi, %rdx), %rax
# endif
	movdqu	%xmm7, 48(%rdi)
	add	$15, %r8
	sub	%rdx, %r8
	lea	49(%rdi, %rdx), %rdi
	jmp	L(StrncpyFillTailWithZero)
# else
	add	$48, %rsi
	add	$48, %rdi
	BRANCH_TO_JMPTBL_ENTRY (L(ExitTable), %rdx, 4)
# endif

/* If source address alignment == destination address alignment */

L(SourceStringAlignmentLess32):
	pxor	%xmm0, %xmm0
	movdqu	(%rsi), %xmm1
	movdqu	16(%rsi), %xmm2
	pcmpeqb	%xmm1, %xmm0
	pmovmskb %xmm0, %rdx

# ifdef USE_AS_STRNCPY
#  if defined USE_AS_STPCPY || defined USE_AS_STRCAT
	cmp	$16, %r8
#  else
	cmp	$17, %r8
#  endif
	jbe	L(CopyFrom1To16BytesTail1Case2OrCase3)
# endif
	test	%rdx, %rdx
	jnz	L(CopyFrom1To16BytesTail1)

	pcmpeqb	%xmm2, %xmm0
	movdqu	%xmm1, (%rdi)
	pmovmskb %xmm0, %rdx

# ifdef USE_AS_STRNCPY
#  if defined USE_AS_STPCPY || defined USE_AS_STRCAT
	cmp	$32, %r8
#  else
	cmp	$33, %r8
#  endif
	jbe	L(CopyFrom1To32Bytes1Case2OrCase3)
# endif
	test	%rdx, %rdx
	jnz	L(CopyFrom1To32Bytes1)

	and	$-16, %rsi
	and	$15, %rcx
	jmp	L(Unalign16Both)

/*------End of main part with loops---------------------*/

/* Case1 */

# if (!defined USE_AS_STRNCPY) || (defined USE_AS_STRCAT)
	.p2align 4
L(CopyFrom1To16Bytes):
	add	%rcx, %rdi
	add	%rcx, %rsi
	bsf	%rdx, %rdx
	BRANCH_TO_JMPTBL_ENTRY (L(ExitTable), %rdx, 4)
# endif
	.p2align 4
L(CopyFrom1To16BytesTail):
	add	%rcx, %rsi
	bsf	%rdx, %rdx
	BRANCH_TO_JMPTBL_ENTRY (L(ExitTable), %rdx, 4)

	.p2align 4
L(CopyFrom1To32Bytes1):
	add	$16, %rsi
	add	$16, %rdi
# if defined USE_AS_STRNCPY && !defined USE_AS_STRCAT
	sub	$16, %r8
# endif
L(CopyFrom1To16BytesTail1):
	bsf	%rdx, %rdx
	BRANCH_TO_JMPTBL_ENTRY (L(ExitTable), %rdx, 4)

	.p2align 4
L(CopyFrom1To32Bytes):
	bsf	%rdx, %rdx
	add	%rcx, %rsi
	add	$16, %rdx
	sub	%rcx, %rdx
	BRANCH_TO_JMPTBL_ENTRY (L(ExitTable), %rdx, 4)

	.p2align 4
L(CopyFrom1To16BytesUnaligned_0):
	bsf	%rdx, %rdx
# if defined USE_AS_STRNCPY && !defined USE_AS_STRCAT
# ifdef USE_AS_STPCPY
	lea	(%rdi, %rdx), %rax
# endif
	movdqu	%xmm4, (%rdi)
	add	$63, %r8
	sub	%rdx, %r8
	lea	1(%rdi, %rdx), %rdi
	jmp	L(StrncpyFillTailWithZero)
# else
	BRANCH_TO_JMPTBL_ENTRY (L(ExitTable), %rdx, 4)
# endif

	.p2align 4
L(CopyFrom1To16BytesUnaligned_16):
	bsf	%rcx, %rdx
	movdqu	%xmm4, (%rdi)
# if defined USE_AS_STRNCPY && !defined USE_AS_STRCAT
# ifdef USE_AS_STPCPY
	lea	16(%rdi, %rdx), %rax
# endif
	movdqu	%xmm5, 16(%rdi)
	add	$47, %r8
	sub	%rdx, %r8
	lea	17(%rdi, %rdx), %rdi
	jmp	L(StrncpyFillTailWithZero)
# else
	add	$16, %rsi
	add	$16, %rdi
	BRANCH_TO_JMPTBL_ENTRY (L(ExitTable), %rdx, 4)
# endif

	.p2align 4
L(CopyFrom1To16BytesUnaligned_32):
	bsf	%rdx, %rdx
	movdqu	%xmm4, (%rdi)
	movdqu	%xmm5, 16(%rdi)
# if defined USE_AS_STRNCPY && !defined USE_AS_STRCAT
# ifdef USE_AS_STPCPY
	lea	32(%rdi, %rdx), %rax
# endif
	movdqu	%xmm6, 32(%rdi)
	add	$31, %r8
	sub	%rdx, %r8
	lea	33(%rdi, %rdx), %rdi
	jmp	L(StrncpyFillTailWithZero)
# else
	add	$32, %rsi
	add	$32, %rdi
	BRANCH_TO_JMPTBL_ENTRY (L(ExitTable), %rdx, 4)
# endif

# ifdef USE_AS_STRNCPY
#  ifndef USE_AS_STRCAT
	.p2align 4
L(CopyFrom1To16BytesUnalignedXmm6):
	movdqu	%xmm6, (%rdi, %rcx)
	jmp	L(CopyFrom1To16BytesXmmExit)

	.p2align 4
L(CopyFrom1To16BytesUnalignedXmm5):
	movdqu	%xmm5, (%rdi, %rcx)
	jmp	L(CopyFrom1To16BytesXmmExit)

	.p2align 4
L(CopyFrom1To16BytesUnalignedXmm4):
	movdqu	%xmm4, (%rdi, %rcx)
	jmp	L(CopyFrom1To16BytesXmmExit)

	.p2align 4
L(CopyFrom1To16BytesUnalignedXmm3):
	movdqu	%xmm3, (%rdi, %rcx)
	jmp	L(CopyFrom1To16BytesXmmExit)

	.p2align 4
L(CopyFrom1To16BytesUnalignedXmm1):
	movdqu	%xmm1, (%rdi, %rcx)
	jmp	L(CopyFrom1To16BytesXmmExit)
#  endif

	.p2align 4
L(CopyFrom1To16BytesExit):
	BRANCH_TO_JMPTBL_ENTRY (L(ExitTable), %rdx, 4)

/* Case2 */

	.p2align 4
L(CopyFrom1To16BytesCase2):
	add	$16, %r8
	add	%rcx, %rdi
	add	%rcx, %rsi
	bsf	%rdx, %rdx
	cmp	%r8, %rdx
	jb	L(CopyFrom1To16BytesExit)
	BRANCH_TO_JMPTBL_ENTRY (L(ExitStrncpyTable), %r8, 4)

	.p2align 4
L(CopyFrom1To32BytesCase2):
	add	%rcx, %rsi
	bsf	%rdx, %rdx
	add	$16, %rdx
	sub	%rcx, %rdx
	cmp	%r8, %rdx
	jb	L(CopyFrom1To16BytesExit)
	BRANCH_TO_JMPTBL_ENTRY (L(ExitStrncpyTable), %r8, 4)

L(CopyFrom1To16BytesTailCase2):
	add	%rcx, %rsi
	bsf	%rdx, %rdx
	cmp	%r8, %rdx
	jb	L(CopyFrom1To16BytesExit)
	BRANCH_TO_JMPTBL_ENTRY (L(ExitStrncpyTable), %r8, 4)

L(CopyFrom1To16BytesTail1Case2):
	bsf	%rdx, %rdx
	cmp	%r8, %rdx
	jb	L(CopyFrom1To16BytesExit)
	BRANCH_TO_JMPTBL_ENTRY (L(ExitStrncpyTable), %r8, 4)

/* Case2 or Case3,  Case3 */

	.p2align 4
L(CopyFrom1To16BytesCase2OrCase3):
	test	%rdx, %rdx
	jnz	L(CopyFrom1To16BytesCase2)
L(CopyFrom1To16BytesCase3):
	add	$16, %r8
	add	%rcx, %rdi
	add	%rcx, %rsi
	BRANCH_TO_JMPTBL_ENTRY (L(ExitStrncpyTable), %r8, 4)

	.p2align 4
L(CopyFrom1To32BytesCase2OrCase3):
	test	%rdx, %rdx
	jnz	L(CopyFrom1To32BytesCase2)
	add	%rcx, %rsi
	BRANCH_TO_JMPTBL_ENTRY (L(ExitStrncpyTable), %r8, 4)

	.p2align 4
L(CopyFrom1To16BytesTailCase2OrCase3):
	test	%rdx, %rdx
	jnz	L(CopyFrom1To16BytesTailCase2)
	add	%rcx, %rsi
	BRANCH_TO_JMPTBL_ENTRY (L(ExitStrncpyTable), %r8, 4)

	.p2align 4
L(CopyFrom1To32Bytes1Case2OrCase3):
	add	$16, %rdi
	add	$16, %rsi
	sub	$16, %r8
L(CopyFrom1To16BytesTail1Case2OrCase3):
	test	%rdx, %rdx
	jnz	L(CopyFrom1To16BytesTail1Case2)
	BRANCH_TO_JMPTBL_ENTRY (L(ExitStrncpyTable), %r8, 4)

# endif

/*------------End labels regarding with copying 1-16 bytes--and 1-32 bytes----*/

	.p2align 4
L(Exit1):
	mov	%dh, (%rdi)
# ifdef USE_AS_STPCPY
	lea	(%rdi), %rax
# endif
# if defined USE_AS_STRNCPY && !defined USE_AS_STRCAT
	sub	$1, %r8
	lea	1(%rdi), %rdi
	jnz	L(StrncpyFillTailWithZero)
# endif
	ret

	.p2align 4
L(Exit2):
	mov	(%rsi), %dx
	mov	%dx, (%rdi)
# ifdef USE_AS_STPCPY
	lea	1(%rdi), %rax
# endif
# if defined USE_AS_STRNCPY && !defined USE_AS_STRCAT
	sub	$2, %r8
	lea	2(%rdi), %rdi
	jnz	L(StrncpyFillTailWithZero)
# endif
	ret

	.p2align 4
L(Exit3):
	mov	(%rsi), %cx
	mov	%cx, (%rdi)
	mov	%dh, 2(%rdi)
# ifdef USE_AS_STPCPY
	lea	2(%rdi), %rax
# endif
# if defined USE_AS_STRNCPY && !defined USE_AS_STRCAT
	sub	$3, %r8
	lea	3(%rdi), %rdi
	jnz	L(StrncpyFillTailWithZero)
# endif
	ret

	.p2align 4
L(Exit4):
	mov	(%rsi), %edx
	mov	%edx, (%rdi)
# ifdef USE_AS_STPCPY
	lea	3(%rdi), %rax
# endif
# if defined USE_AS_STRNCPY && !defined USE_AS_STRCAT
	sub	$4, %r8
	lea	4(%rdi), %rdi
	jnz	L(StrncpyFillTailWithZero)
# endif
	ret

	.p2align 4
L(Exit5):
	mov	(%rsi), %ecx
	mov	%dh, 4(%rdi)
	mov	%ecx, (%rdi)
# ifdef USE_AS_STPCPY
	lea	4(%rdi), %rax
# endif
# if defined USE_AS_STRNCPY && !defined USE_AS_STRCAT
	sub	$5, %r8
	lea	5(%rdi), %rdi
	jnz	L(StrncpyFillTailWithZero)
# endif
	ret

	.p2align 4
L(Exit6):
	mov	(%rsi), %ecx
	mov	4(%rsi), %dx
	mov	%ecx, (%rdi)
	mov	%dx, 4(%rdi)
# ifdef USE_AS_STPCPY
	lea	5(%rdi), %rax
# endif
# if defined USE_AS_STRNCPY && !defined USE_AS_STRCAT
	sub	$6, %r8
	lea	6(%rdi), %rdi
	jnz	L(StrncpyFillTailWithZero)
# endif
	ret

	.p2align 4
L(Exit7):
	mov	(%rsi), %ecx
	mov	3(%rsi), %edx
	mov	%ecx, (%rdi)
	mov	%edx, 3(%rdi)
# ifdef USE_AS_STPCPY
	lea	6(%rdi), %rax
# endif
# if defined USE_AS_STRNCPY && !defined USE_AS_STRCAT
	sub	$7, %r8
	lea	7(%rdi), %rdi
	jnz	L(StrncpyFillTailWithZero)
# endif
	ret

	.p2align 4
L(Exit8):
	mov	(%rsi), %rdx
	mov	%rdx, (%rdi)
# ifdef USE_AS_STPCPY
	lea	7(%rdi), %rax
# endif
# if defined USE_AS_STRNCPY && !defined USE_AS_STRCAT
	sub	$8, %r8
	lea	8(%rdi), %rdi
	jnz	L(StrncpyFillTailWithZero)
# endif
	ret

	.p2align 4
L(Exit9):
	mov	(%rsi), %rcx
	mov	%dh, 8(%rdi)
	mov	%rcx, (%rdi)
# ifdef USE_AS_STPCPY
	lea	8(%rdi), %rax
# endif
# if defined USE_AS_STRNCPY && !defined USE_AS_STRCAT
	sub	$9, %r8
	lea	9(%rdi), %rdi
	jnz	L(StrncpyFillTailWithZero)
# endif
	ret

	.p2align 4
L(Exit10):
	mov	(%rsi), %rcx
	mov	8(%rsi), %dx
	mov	%rcx, (%rdi)
	mov	%dx, 8(%rdi)
# ifdef USE_AS_STPCPY
	lea	9(%rdi), %rax
# endif
# if defined USE_AS_STRNCPY && !defined USE_AS_STRCAT
	sub	$10, %r8
	lea	10(%rdi), %rdi
	jnz	L(StrncpyFillTailWithZero)
# endif
	ret

	.p2align 4
L(Exit11):
	mov	(%rsi), %rcx
	mov	7(%rsi), %edx
	mov	%rcx, (%rdi)
	mov	%edx, 7(%rdi)
# ifdef USE_AS_STPCPY
	lea	10(%rdi), %rax
# endif
# if defined USE_AS_STRNCPY && !defined USE_AS_STRCAT
	sub	$11, %r8
	lea	11(%rdi), %rdi
	jnz	L(StrncpyFillTailWithZero)
# endif
	ret

	.p2align 4
L(Exit12):
	mov	(%rsi), %rcx
	mov	8(%rsi), %edx
	mov	%rcx, (%rdi)
	mov	%edx, 8(%rdi)
# ifdef USE_AS_STPCPY
	lea	11(%rdi), %rax
# endif
# if defined USE_AS_STRNCPY && !defined USE_AS_STRCAT
	sub	$12, %r8
	lea	12(%rdi), %rdi
	jnz	L(StrncpyFillTailWithZero)
# endif
	ret

	.p2align 4
L(Exit13):
	mov	(%rsi), %rcx
	mov	5(%rsi), %rdx
	mov	%rcx, (%rdi)
	mov	%rdx, 5(%rdi)
# ifdef USE_AS_STPCPY
	lea	12(%rdi), %rax
# endif
# if defined USE_AS_STRNCPY && !defined USE_AS_STRCAT
	sub	$13, %r8
	lea	13(%rdi), %rdi
	jnz	L(StrncpyFillTailWithZero)
# endif
	ret

	.p2align 4
L(Exit14):
	mov	(%rsi), %rcx
	mov	6(%rsi), %rdx
	mov	%rcx, (%rdi)
	mov	%rdx, 6(%rdi)
# ifdef USE_AS_STPCPY
	lea	13(%rdi), %rax
# endif
# if defined USE_AS_STRNCPY && !defined USE_AS_STRCAT
	sub	$14, %r8
	lea	14(%rdi), %rdi
	jnz	L(StrncpyFillTailWithZero)
# endif
	ret

	.p2align 4
L(Exit15):
	mov	(%rsi), %rcx
	mov	7(%rsi), %rdx
	mov	%rcx, (%rdi)
	mov	%rdx, 7(%rdi)
# ifdef USE_AS_STPCPY
	lea	14(%rdi), %rax
# endif
# if defined USE_AS_STRNCPY && !defined USE_AS_STRCAT
	sub	$15, %r8
	lea	15(%rdi), %rdi
	jnz	L(StrncpyFillTailWithZero)
# endif
	ret

	.p2align 4
L(Exit16):
	movdqu	(%rsi), %xmm0
	movdqu	%xmm0, (%rdi)
# ifdef USE_AS_STPCPY
	lea	15(%rdi), %rax
# endif
# if defined USE_AS_STRNCPY && !defined USE_AS_STRCAT
	sub	$16, %r8
	lea	16(%rdi), %rdi
	jnz	L(StrncpyFillTailWithZero)
# endif
	ret

	.p2align 4
L(Exit17):
	movdqu	(%rsi), %xmm0
	movdqu	%xmm0, (%rdi)
	mov	%dh, 16(%rdi)
# ifdef USE_AS_STPCPY
	lea	16(%rdi), %rax
# endif
# if defined USE_AS_STRNCPY && !defined USE_AS_STRCAT
	sub	$17, %r8
	lea	17(%rdi), %rdi
	jnz	L(StrncpyFillTailWithZero)
# endif
	ret

	.p2align 4
L(Exit18):
	movdqu	(%rsi), %xmm0
	mov	16(%rsi), %cx
	movdqu	%xmm0, (%rdi)
	mov	%cx, 16(%rdi)
# ifdef USE_AS_STPCPY
	lea	17(%rdi), %rax
# endif
# if defined USE_AS_STRNCPY && !defined USE_AS_STRCAT
	sub	$18, %r8
	lea	18(%rdi), %rdi
	jnz	L(StrncpyFillTailWithZero)
# endif
	ret

	.p2align 4
L(Exit19):
	movdqu	(%rsi), %xmm0
	mov	15(%rsi), %ecx
	movdqu	%xmm0, (%rdi)
	mov	%ecx, 15(%rdi)
# ifdef USE_AS_STPCPY
	lea	18(%rdi), %rax
# endif
# if defined USE_AS_STRNCPY && !defined USE_AS_STRCAT
	sub	$19, %r8
	lea	19(%rdi), %rdi
	jnz	L(StrncpyFillTailWithZero)
# endif
	ret

	.p2align 4
L(Exit20):
	movdqu	(%rsi), %xmm0
	mov	16(%rsi), %ecx
	movdqu	%xmm0, (%rdi)
	mov	%ecx, 16(%rdi)
# ifdef USE_AS_STPCPY
	lea	19(%rdi), %rax
# endif
# if defined USE_AS_STRNCPY && !defined USE_AS_STRCAT
	sub	$20, %r8
	lea	20(%rdi), %rdi
	jnz	L(StrncpyFillTailWithZero)
# endif
	ret

	.p2align 4
L(Exit21):
	movdqu	(%rsi), %xmm0
	mov	16(%rsi), %ecx
	movdqu	%xmm0, (%rdi)
	mov	%ecx, 16(%rdi)
	mov	%dh, 20(%rdi)
# ifdef USE_AS_STPCPY
	lea	20(%rdi), %rax
# endif
# if defined USE_AS_STRNCPY && !defined USE_AS_STRCAT
	sub	$21, %r8
	lea	21(%rdi), %rdi
	jnz	L(StrncpyFillTailWithZero)
# endif
	ret

	.p2align 4
L(Exit22):
	movdqu	(%rsi), %xmm0
	mov	14(%rsi), %rcx
	movdqu	%xmm0, (%rdi)
	mov	%rcx, 14(%rdi)
# ifdef USE_AS_STPCPY
	lea	21(%rdi), %rax
# endif
# if defined USE_AS_STRNCPY && !defined USE_AS_STRCAT
	sub	$22, %r8
	lea	22(%rdi), %rdi
	jnz	L(StrncpyFillTailWithZero)
# endif
	ret

	.p2align 4
L(Exit23):
	movdqu	(%rsi), %xmm0
	mov	15(%rsi), %rcx
	movdqu	%xmm0, (%rdi)
	mov	%rcx, 15(%rdi)
# ifdef USE_AS_STPCPY
	lea	22(%rdi), %rax
# endif
# if defined USE_AS_STRNCPY && !defined USE_AS_STRCAT
	sub	$23, %r8
	lea	23(%rdi), %rdi
	jnz	L(StrncpyFillTailWithZero)
# endif
	ret

	.p2align 4
L(Exit24):
	movdqu	(%rsi), %xmm0
	mov	16(%rsi), %rcx
	movdqu	%xmm0, (%rdi)
	mov	%rcx, 16(%rdi)
# ifdef USE_AS_STPCPY
	lea	23(%rdi), %rax
# endif
# if defined USE_AS_STRNCPY && !defined USE_AS_STRCAT
	sub	$24, %r8
	lea	24(%rdi), %rdi
	jnz	L(StrncpyFillTailWithZero)
# endif
	ret

	.p2align 4
L(Exit25):
	movdqu	(%rsi), %xmm0
	mov	16(%rsi), %rcx
	movdqu	%xmm0, (%rdi)
	mov	%rcx, 16(%rdi)
	mov	%dh, 24(%rdi)
# ifdef USE_AS_STPCPY
	lea	24(%rdi), %rax
# endif
# if defined USE_AS_STRNCPY && !defined USE_AS_STRCAT
	sub	$25, %r8
	lea	25(%rdi), %rdi
	jnz	L(StrncpyFillTailWithZero)
# endif
	ret

	.p2align 4
L(Exit26):
	movdqu	(%rsi), %xmm0
	mov	16(%rsi), %rdx
	mov	24(%rsi), %cx
	movdqu	%xmm0, (%rdi)
	mov	%rdx, 16(%rdi)
	mov	%cx, 24(%rdi)
# ifdef USE_AS_STPCPY
	lea	25(%rdi), %rax
# endif
# if defined USE_AS_STRNCPY && !defined USE_AS_STRCAT
	sub	$26, %r8
	lea	26(%rdi), %rdi
	jnz	L(StrncpyFillTailWithZero)
# endif
	ret

	.p2align 4
L(Exit27):
	movdqu	(%rsi), %xmm0
	mov	16(%rsi), %rdx
	mov	23(%rsi), %ecx
	movdqu	%xmm0, (%rdi)
	mov	%rdx, 16(%rdi)
	mov	%ecx, 23(%rdi)
# ifdef USE_AS_STPCPY
	lea	26(%rdi), %rax
# endif
# if defined USE_AS_STRNCPY && !defined USE_AS_STRCAT
	sub	$27, %r8
	lea	27(%rdi), %rdi
	jnz	L(StrncpyFillTailWithZero)
# endif
	ret

	.p2align 4
L(Exit28):
	movdqu	(%rsi), %xmm0
	mov	16(%rsi), %rdx
	mov	24(%rsi), %ecx
	movdqu	%xmm0, (%rdi)
	mov	%rdx, 16(%rdi)
	mov	%ecx, 24(%rdi)
# ifdef USE_AS_STPCPY
	lea	27(%rdi), %rax
# endif
# if defined USE_AS_STRNCPY && !defined USE_AS_STRCAT
	sub	$28, %r8
	lea	28(%rdi), %rdi
	jnz	L(StrncpyFillTailWithZero)
# endif
	ret

	.p2align 4
L(Exit29):
	movdqu	(%rsi), %xmm0
	movdqu	13(%rsi), %xmm2
	movdqu	%xmm0, (%rdi)
	movdqu	%xmm2, 13(%rdi)
# ifdef USE_AS_STPCPY
	lea	28(%rdi), %rax
# endif
# if defined USE_AS_STRNCPY && !defined USE_AS_STRCAT
	sub	$29, %r8
	lea	29(%rdi), %rdi
	jnz	L(StrncpyFillTailWithZero)
# endif
	ret

	.p2align 4
L(Exit30):
	movdqu	(%rsi), %xmm0
	movdqu	14(%rsi), %xmm2
	movdqu	%xmm0, (%rdi)
	movdqu	%xmm2, 14(%rdi)
# ifdef USE_AS_STPCPY
	lea	29(%rdi), %rax
# endif
# if defined USE_AS_STRNCPY && !defined USE_AS_STRCAT
	sub	$30, %r8
	lea	30(%rdi), %rdi
	jnz	L(StrncpyFillTailWithZero)
# endif
	ret

	.p2align 4
L(Exit31):
	movdqu	(%rsi), %xmm0
	movdqu	15(%rsi), %xmm2
	movdqu	%xmm0, (%rdi)
	movdqu	%xmm2, 15(%rdi)
# ifdef USE_AS_STPCPY
	lea	30(%rdi), %rax
# endif
# if defined USE_AS_STRNCPY && !defined USE_AS_STRCAT
	sub	$31, %r8
	lea	31(%rdi), %rdi
	jnz	L(StrncpyFillTailWithZero)
# endif
	ret

	.p2align 4
L(Exit32):
	movdqu	(%rsi), %xmm0
	movdqu	16(%rsi), %xmm2
	movdqu	%xmm0, (%rdi)
	movdqu	%xmm2, 16(%rdi)
# ifdef USE_AS_STPCPY
	lea	31(%rdi), %rax
# endif
# if defined USE_AS_STRNCPY && !defined USE_AS_STRCAT
	sub	$32, %r8
	lea	32(%rdi), %rdi
	jnz	L(StrncpyFillTailWithZero)
# endif
	ret

# ifdef USE_AS_STRNCPY

	.p2align 4
L(StrncpyExit0):
#  ifdef USE_AS_STPCPY
	mov	%rdi, %rax
#  endif
#  ifdef USE_AS_STRCAT
	xor	%ch, %ch
	movb	%ch, (%rdi)
#  endif
	ret

	.p2align 4
L(StrncpyExit1):
	mov	(%rsi), %dl
	mov	%dl, (%rdi)
#  ifdef USE_AS_STPCPY
	lea	1(%rdi), %rax
#  endif
#  ifdef USE_AS_STRCAT
	xor	%ch, %ch
	movb	%ch, 1(%rdi)
#  endif
	ret

	.p2align 4
L(StrncpyExit2):
	mov	(%rsi), %dx
	mov	%dx, (%rdi)
#  ifdef USE_AS_STPCPY
	lea	2(%rdi), %rax
#  endif
#  ifdef USE_AS_STRCAT
	xor	%ch, %ch
	movb	%ch, 2(%rdi)
#  endif
	ret

	.p2align 4
L(StrncpyExit3):
	mov	(%rsi), %cx
	mov	2(%rsi), %dl
	mov	%cx, (%rdi)
	mov	%dl, 2(%rdi)
#  ifdef USE_AS_STPCPY
	lea	3(%rdi), %rax
#  endif
#  ifdef USE_AS_STRCAT
	xor	%ch, %ch
	movb	%ch, 3(%rdi)
#  endif
	ret

	.p2align 4
L(StrncpyExit4):
	mov	(%rsi), %edx
	mov	%edx, (%rdi)
#  ifdef USE_AS_STPCPY
	lea	4(%rdi), %rax
#  endif
#  ifdef USE_AS_STRCAT
	xor	%ch, %ch
	movb	%ch, 4(%rdi)
#  endif
	ret

	.p2align 4
L(StrncpyExit5):
	mov	(%rsi), %ecx
	mov	4(%rsi), %dl
	mov	%ecx, (%rdi)
	mov	%dl, 4(%rdi)
#  ifdef USE_AS_STPCPY
	lea	5(%rdi), %rax
#  endif
#  ifdef USE_AS_STRCAT
	xor	%ch, %ch
	movb	%ch, 5(%rdi)
#  endif
	ret

	.p2align 4
L(StrncpyExit6):
	mov	(%rsi), %ecx
	mov	4(%rsi), %dx
	mov	%ecx, (%rdi)
	mov	%dx, 4(%rdi)
#  ifdef USE_AS_STPCPY
	lea	6(%rdi), %rax
#  endif
#  ifdef USE_AS_STRCAT
	xor	%ch, %ch
	movb	%ch, 6(%rdi)
#  endif
	ret

	.p2align 4
L(StrncpyExit7):
	mov	(%rsi), %ecx
	mov	3(%rsi), %edx
	mov	%ecx, (%rdi)
	mov	%edx, 3(%rdi)
#  ifdef USE_AS_STPCPY
	lea	7(%rdi), %rax
#  endif
#  ifdef USE_AS_STRCAT
	xor	%ch, %ch
	movb	%ch, 7(%rdi)
#  endif
	ret

	.p2align 4
L(StrncpyExit8):
	mov	(%rsi), %rdx
	mov	%rdx, (%rdi)
#  ifdef USE_AS_STPCPY
	lea	8(%rdi), %rax
#  endif
#  ifdef USE_AS_STRCAT
	xor	%ch, %ch
	movb	%ch, 8(%rdi)
#  endif
	ret

	.p2align 4
L(StrncpyExit9):
	mov	(%rsi), %rcx
	mov	8(%rsi), %dl
	mov	%rcx, (%rdi)
	mov	%dl, 8(%rdi)
#  ifdef USE_AS_STPCPY
	lea	9(%rdi), %rax
#  endif
#  ifdef USE_AS_STRCAT
	xor	%ch, %ch
	movb	%ch, 9(%rdi)
#  endif
	ret

	.p2align 4
L(StrncpyExit10):
	mov	(%rsi), %rcx
	mov	8(%rsi), %dx
	mov	%rcx, (%rdi)
	mov	%dx, 8(%rdi)
#  ifdef USE_AS_STPCPY
	lea	10(%rdi), %rax
#  endif
#  ifdef USE_AS_STRCAT
	xor	%ch, %ch
	movb	%ch, 10(%rdi)
#  endif
	ret

	.p2align 4
L(StrncpyExit11):
	mov	(%rsi), %rcx
	mov	7(%rsi), %edx
	mov	%rcx, (%rdi)
	mov	%edx, 7(%rdi)
#  ifdef USE_AS_STPCPY
	lea	11(%rdi), %rax
#  endif
#  ifdef USE_AS_STRCAT
	xor	%ch, %ch
	movb	%ch, 11(%rdi)
#  endif
	ret

	.p2align 4
L(StrncpyExit12):
	mov	(%rsi), %rcx
	mov	8(%rsi), %edx
	mov	%rcx, (%rdi)
	mov	%edx, 8(%rdi)
#  ifdef USE_AS_STPCPY
	lea	12(%rdi), %rax
#  endif
#  ifdef USE_AS_STRCAT
	xor	%ch, %ch
	movb	%ch, 12(%rdi)
#  endif
	ret

	.p2align 4
L(StrncpyExit13):
	mov	(%rsi), %rcx
	mov	5(%rsi), %rdx
	mov	%rcx, (%rdi)
	mov	%rdx, 5(%rdi)
#  ifdef USE_AS_STPCPY
	lea	13(%rdi), %rax
#  endif
#  ifdef USE_AS_STRCAT
	xor	%ch, %ch
	movb	%ch, 13(%rdi)
#  endif
	ret

	.p2align 4
L(StrncpyExit14):
	mov	(%rsi), %rcx
	mov	6(%rsi), %rdx
	mov	%rcx, (%rdi)
	mov	%rdx, 6(%rdi)
#  ifdef USE_AS_STPCPY
	lea	14(%rdi), %rax
#  endif
#  ifdef USE_AS_STRCAT
	xor	%ch, %ch
	movb	%ch, 14(%rdi)
#  endif
	ret

	.p2align 4
L(StrncpyExit15):
	mov	(%rsi), %rcx
	mov	7(%rsi), %rdx
	mov	%rcx, (%rdi)
	mov	%rdx, 7(%rdi)
#  ifdef USE_AS_STPCPY
	lea	15(%rdi), %rax
#  endif
#  ifdef USE_AS_STRCAT
	xor	%ch, %ch
	movb	%ch, 15(%rdi)
#  endif
	ret

	.p2align 4
L(StrncpyExit16):
	movdqu	(%rsi), %xmm0
	movdqu	%xmm0, (%rdi)
#  ifdef USE_AS_STPCPY
	lea	16(%rdi), %rax
#  endif
#  ifdef USE_AS_STRCAT
	xor	%ch, %ch
	movb	%ch, 16(%rdi)
#  endif
	ret

	.p2align 4
L(StrncpyExit17):
	movdqu	(%rsi), %xmm0
	mov	16(%rsi), %cl
	movdqu	%xmm0, (%rdi)
	mov	%cl, 16(%rdi)
#  ifdef USE_AS_STPCPY
	lea	17(%rdi), %rax
#  endif
#  ifdef USE_AS_STRCAT
	xor	%ch, %ch
	movb	%ch, 17(%rdi)
#  endif
	ret

	.p2align 4
L(StrncpyExit18):
	movdqu	(%rsi), %xmm0
	mov	16(%rsi), %cx
	movdqu	%xmm0, (%rdi)
	mov	%cx, 16(%rdi)
#  ifdef USE_AS_STPCPY
	lea	18(%rdi), %rax
#  endif
#  ifdef USE_AS_STRCAT
	xor	%ch, %ch
	movb	%ch, 18(%rdi)
#  endif
	ret

	.p2align 4
L(StrncpyExit19):
	movdqu	(%rsi), %xmm0
	mov	15(%rsi), %ecx
	movdqu	%xmm0, (%rdi)
	mov	%ecx, 15(%rdi)
#  ifdef USE_AS_STPCPY
	lea	19(%rdi), %rax
#  endif
#  ifdef USE_AS_STRCAT
	xor	%ch, %ch
	movb	%ch, 19(%rdi)
#  endif
	ret

	.p2align 4
L(StrncpyExit20):
	movdqu	(%rsi), %xmm0
	mov	16(%rsi), %ecx
	movdqu	%xmm0, (%rdi)
	mov	%ecx, 16(%rdi)
#  ifdef USE_AS_STPCPY
	lea	20(%rdi), %rax
#  endif
#  ifdef USE_AS_STRCAT
	xor	%ch, %ch
	movb	%ch, 20(%rdi)
#  endif
	ret

	.p2align 4
L(StrncpyExit21):
	movdqu	(%rsi), %xmm0
	mov	16(%rsi), %ecx
	mov	20(%rsi), %dl
	movdqu	%xmm0, (%rdi)
	mov	%ecx, 16(%rdi)
	mov	%dl, 20(%rdi)
#  ifdef USE_AS_STPCPY
	lea	21(%rdi), %rax
#  endif
#  ifdef USE_AS_STRCAT
	xor	%ch, %ch
	movb	%ch, 21(%rdi)
#  endif
	ret

	.p2align 4
L(StrncpyExit22):
	movdqu	(%rsi), %xmm0
	mov	14(%rsi), %rcx
	movdqu	%xmm0, (%rdi)
	mov	%rcx, 14(%rdi)
#  ifdef USE_AS_STPCPY
	lea	22(%rdi), %rax
#  endif
#  ifdef USE_AS_STRCAT
	xor	%ch, %ch
	movb	%ch, 22(%rdi)
#  endif
	ret

	.p2align 4
L(StrncpyExit23):
	movdqu	(%rsi), %xmm0
	mov	15(%rsi), %rcx
	movdqu	%xmm0, (%rdi)
	mov	%rcx, 15(%rdi)
#  ifdef USE_AS_STPCPY
	lea	23(%rdi), %rax
#  endif
#  ifdef USE_AS_STRCAT
	xor	%ch, %ch
	movb	%ch, 23(%rdi)
#  endif
	ret

	.p2align 4
L(StrncpyExit24):
	movdqu	(%rsi), %xmm0
	mov	16(%rsi), %rcx
	movdqu	%xmm0, (%rdi)
	mov	%rcx, 16(%rdi)
#  ifdef USE_AS_STPCPY
	lea	24(%rdi), %rax
#  endif
#  ifdef USE_AS_STRCAT
	xor	%ch, %ch
	movb	%ch, 24(%rdi)
#  endif
	ret

	.p2align 4
L(StrncpyExit25):
	movdqu	(%rsi), %xmm0
	mov	16(%rsi), %rdx
	mov	24(%rsi), %cl
	movdqu	%xmm0, (%rdi)
	mov	%rdx, 16(%rdi)
	mov	%cl, 24(%rdi)
#  ifdef USE_AS_STPCPY
	lea	25(%rdi), %rax
#  endif
#  ifdef USE_AS_STRCAT
	xor	%ch, %ch
	movb	%ch, 25(%rdi)
#  endif
	ret

	.p2align 4
L(StrncpyExit26):
	movdqu	(%rsi), %xmm0
	mov	16(%rsi), %rdx
	mov	24(%rsi), %cx
	movdqu	%xmm0, (%rdi)
	mov	%rdx, 16(%rdi)
	mov	%cx, 24(%rdi)
#  ifdef USE_AS_STPCPY
	lea	26(%rdi), %rax
#  endif
#  ifdef USE_AS_STRCAT
	xor	%ch, %ch
	movb	%ch, 26(%rdi)
#  endif
	ret

	.p2align 4
L(StrncpyExit27):
	movdqu	(%rsi), %xmm0
	mov	16(%rsi), %rdx
	mov	23(%rsi), %ecx
	movdqu	%xmm0, (%rdi)
	mov	%rdx, 16(%rdi)
	mov	%ecx, 23(%rdi)
#  ifdef USE_AS_STPCPY
	lea	27(%rdi), %rax
#  endif
#  ifdef USE_AS_STRCAT
	xor	%ch, %ch
	movb	%ch, 27(%rdi)
#  endif
	ret

	.p2align 4
L(StrncpyExit28):
	movdqu	(%rsi), %xmm0
	mov	16(%rsi), %rdx
	mov	24(%rsi), %ecx
	movdqu	%xmm0, (%rdi)
	mov	%rdx, 16(%rdi)
	mov	%ecx, 24(%rdi)
#  ifdef USE_AS_STPCPY
	lea	28(%rdi), %rax
#  endif
#  ifdef USE_AS_STRCAT
	xor	%ch, %ch
	movb	%ch, 28(%rdi)
#  endif
	ret

	.p2align 4
L(StrncpyExit29):
	movdqu	(%rsi), %xmm0
	movdqu	13(%rsi), %xmm2
	movdqu	%xmm0, (%rdi)
	movdqu	%xmm2, 13(%rdi)
#  ifdef USE_AS_STPCPY
	lea	29(%rdi), %rax
#  endif
#  ifdef USE_AS_STRCAT
	xor	%ch, %ch
	movb	%ch, 29(%rdi)
#  endif
	ret

	.p2align 4
L(StrncpyExit30):
	movdqu	(%rsi), %xmm0
	movdqu	14(%rsi), %xmm2
	movdqu	%xmm0, (%rdi)
	movdqu	%xmm2, 14(%rdi)
#  ifdef USE_AS_STPCPY
	lea	30(%rdi), %rax
#  endif
#  ifdef USE_AS_STRCAT
	xor	%ch, %ch
	movb	%ch, 30(%rdi)
#  endif
	ret

	.p2align 4
L(StrncpyExit31):
	movdqu	(%rsi), %xmm0
	movdqu	15(%rsi), %xmm2
	movdqu	%xmm0, (%rdi)
	movdqu	%xmm2, 15(%rdi)
#  ifdef USE_AS_STPCPY
	lea	31(%rdi), %rax
#  endif
#  ifdef USE_AS_STRCAT
	xor	%ch, %ch
	movb	%ch, 31(%rdi)
#  endif
	ret

	.p2align 4
L(StrncpyExit32):
	movdqu	(%rsi), %xmm0
	movdqu	16(%rsi), %xmm2
	movdqu	%xmm0, (%rdi)
	movdqu	%xmm2, 16(%rdi)
#  ifdef USE_AS_STPCPY
	lea	32(%rdi), %rax
#  endif
#  ifdef USE_AS_STRCAT
	xor	%ch, %ch
	movb	%ch, 32(%rdi)
#  endif
	ret

	.p2align 4
L(StrncpyExit33):
	movdqu	(%rsi), %xmm0
	movdqu	16(%rsi), %xmm2
	mov	32(%rsi), %cl
	movdqu	%xmm0, (%rdi)
	movdqu	%xmm2, 16(%rdi)
	mov	%cl, 32(%rdi)
#  ifdef USE_AS_STRCAT
	xor	%ch, %ch
	movb	%ch, 33(%rdi)
#  endif
	ret

#  ifndef USE_AS_STRCAT

	.p2align 4
L(Fill0):
	ret

	.p2align 4
L(Fill1):
	mov	%dl, (%rdi)
	ret

	.p2align 4
L(Fill2):
	mov	%dx, (%rdi)
	ret

	.p2align 4
L(Fill3):
	mov	%edx, -1(%rdi)
	ret

	.p2align 4
L(Fill4):
	mov	%edx, (%rdi)
	ret

	.p2align 4
L(Fill5):
	mov	%edx, (%rdi)
	mov	%dl, 4(%rdi)
	ret

	.p2align 4
L(Fill6):
	mov	%edx, (%rdi)
	mov	%dx, 4(%rdi)
	ret

	.p2align 4
L(Fill7):
	mov	%rdx, -1(%rdi)
	ret

	.p2align 4
L(Fill8):
	mov	%rdx, (%rdi)
	ret

	.p2align 4
L(Fill9):
	mov	%rdx, (%rdi)
	mov	%dl, 8(%rdi)
	ret

	.p2align 4
L(Fill10):
	mov	%rdx, (%rdi)
	mov	%dx, 8(%rdi)
	ret

	.p2align 4
L(Fill11):
	mov	%rdx, (%rdi)
	mov	%edx, 7(%rdi)
	ret

	.p2align 4
L(Fill12):
	mov	%rdx, (%rdi)
	mov	%edx, 8(%rdi)
	ret

	.p2align 4
L(Fill13):
	mov	%rdx, (%rdi)
	mov	%rdx, 5(%rdi)
	ret

	.p2align 4
L(Fill14):
	mov	%rdx, (%rdi)
	mov	%rdx, 6(%rdi)
	ret

	.p2align 4
L(Fill15):
	movdqu	%xmm0, -1(%rdi)
	ret

	.p2align 4
L(Fill16):
	movdqu	%xmm0, (%rdi)
	ret

	.p2align 4
L(CopyFrom1To16BytesUnalignedXmm2):
	movdqu	%xmm2, (%rdi, %rcx)

	.p2align 4
L(CopyFrom1To16BytesXmmExit):
	bsf	%rdx, %rdx
	add	$15, %r8
	add	%rcx, %rdi
#   ifdef USE_AS_STPCPY
	lea	(%rdi, %rdx), %rax
#   endif
	sub	%rdx, %r8
	lea	1(%rdi, %rdx), %rdi

	.p2align 4
L(StrncpyFillTailWithZero):
	pxor	%xmm0, %xmm0
	xor	%rdx, %rdx
	sub	$16, %r8
	jbe	L(StrncpyFillExit)

	movdqu	%xmm0, (%rdi)
	add	$16, %rdi

	mov	%rdi, %rsi
	and	$0xf, %rsi
	sub	%rsi, %rdi
	add	%rsi, %r8
	sub	$64, %r8
	jb	L(StrncpyFillLess64)

L(StrncpyFillLoopMovdqa):
	movdqa	%xmm0, (%rdi)
	movdqa	%xmm0, 16(%rdi)
	movdqa	%xmm0, 32(%rdi)
	movdqa	%xmm0, 48(%rdi)
	add	$64, %rdi
	sub	$64, %r8
	jae	L(StrncpyFillLoopMovdqa)

L(StrncpyFillLess64):
	add	$32, %r8
	jl	L(StrncpyFillLess32)
	movdqa	%xmm0, (%rdi)
	movdqa	%xmm0, 16(%rdi)
	add	$32, %rdi
	sub	$16, %r8
	jl	L(StrncpyFillExit)
	movdqa	%xmm0, (%rdi)
	add	$16, %rdi
	BRANCH_TO_JMPTBL_ENTRY (L(FillTable), %r8, 4)

L(StrncpyFillLess32):
	add	$16, %r8
	jl	L(StrncpyFillExit)
	movdqa	%xmm0, (%rdi)
	add	$16, %rdi
	BRANCH_TO_JMPTBL_ENTRY (L(FillTable), %r8, 4)

L(StrncpyFillExit):
	add	$16, %r8
	BRANCH_TO_JMPTBL_ENTRY (L(FillTable), %r8, 4)

/* end of ifndef USE_AS_STRCAT */
#  endif

	.p2align 4
L(UnalignedLeaveCase2OrCase3):
	test	%rdx, %rdx
	jnz	L(Unaligned64LeaveCase2)
L(Unaligned64LeaveCase3):
	lea	64(%r8), %rcx
	and	$-16, %rcx
	add	$48, %r8
	jl	L(CopyFrom1To16BytesCase3)
	movdqu	%xmm4, (%rdi)
	sub	$16, %r8
	jb	L(CopyFrom1To16BytesCase3)
	movdqu	%xmm5, 16(%rdi)
	sub	$16, %r8
	jb	L(CopyFrom1To16BytesCase3)
	movdqu	%xmm6, 32(%rdi)
	sub	$16, %r8
	jb	L(CopyFrom1To16BytesCase3)
	movdqu	%xmm7, 48(%rdi)
#  ifdef USE_AS_STPCPY
	lea	64(%rdi), %rax
#  endif
#  ifdef USE_AS_STRCAT
	xor	%ch, %ch
	movb	%ch, 64(%rdi)
#  endif
	ret

	.p2align 4
L(Unaligned64LeaveCase2):
	xor	%rcx, %rcx
	pcmpeqb	%xmm4, %xmm0
	pmovmskb %xmm0, %rdx
	add	$48, %r8
	jle	L(CopyFrom1To16BytesCase2OrCase3)
	test	%rdx, %rdx
#  ifndef USE_AS_STRCAT
	jnz	L(CopyFrom1To16BytesUnalignedXmm4)
#  else
	jnz	L(CopyFrom1To16Bytes)
#  endif
	pcmpeqb	%xmm5, %xmm0
	pmovmskb %xmm0, %rdx
	movdqu	%xmm4, (%rdi)
	add	$16, %rcx
	sub	$16, %r8
	jbe	L(CopyFrom1To16BytesCase2OrCase3)
	test	%rdx, %rdx
#  ifndef USE_AS_STRCAT
	jnz	L(CopyFrom1To16BytesUnalignedXmm5)
#  else
	jnz	L(CopyFrom1To16Bytes)
#  endif

	pcmpeqb	%xmm6, %xmm0
	pmovmskb %xmm0, %rdx
	movdqu	%xmm5, 16(%rdi)
	add	$16, %rcx
	sub	$16, %r8
	jbe	L(CopyFrom1To16BytesCase2OrCase3)
	test	%rdx, %rdx
#  ifndef USE_AS_STRCAT
	jnz	L(CopyFrom1To16BytesUnalignedXmm6)
#  else
	jnz	L(CopyFrom1To16Bytes)
#  endif

	pcmpeqb	%xmm7, %xmm0
	pmovmskb %xmm0, %rdx
	movdqu	%xmm6, 32(%rdi)
	lea	16(%rdi, %rcx), %rdi
	lea	16(%rsi, %rcx), %rsi
	bsf	%rdx, %rdx
	cmp	%r8, %rdx
	jb	L(CopyFrom1To16BytesExit)
	BRANCH_TO_JMPTBL_ENTRY (L(ExitStrncpyTable), %r8, 4)

	.p2align 4
L(ExitZero):
#  ifndef USE_AS_STRCAT
	mov	%rdi, %rax
#  endif
	ret

# endif

# ifndef USE_AS_STRCAT
END (STRCPY)
# else
END (STRCAT)
# endif
	.p2align 4
	.section .rodata
L(ExitTable):
	.int	JMPTBL(L(Exit1), L(ExitTable))
	.int	JMPTBL(L(Exit2), L(ExitTable))
	.int	JMPTBL(L(Exit3), L(ExitTable))
	.int	JMPTBL(L(Exit4), L(ExitTable))
	.int	JMPTBL(L(Exit5), L(ExitTable))
	.int	JMPTBL(L(Exit6), L(ExitTable))
	.int	JMPTBL(L(Exit7), L(ExitTable))
	.int	JMPTBL(L(Exit8), L(ExitTable))
	.int	JMPTBL(L(Exit9), L(ExitTable))
	.int	JMPTBL(L(Exit10), L(ExitTable))
	.int	JMPTBL(L(Exit11), L(ExitTable))
	.int	JMPTBL(L(Exit12), L(ExitTable))
	.int	JMPTBL(L(Exit13), L(ExitTable))
	.int	JMPTBL(L(Exit14), L(ExitTable))
	.int	JMPTBL(L(Exit15), L(ExitTable))
	.int	JMPTBL(L(Exit16), L(ExitTable))
	.int	JMPTBL(L(Exit17), L(ExitTable))
	.int	JMPTBL(L(Exit18), L(ExitTable))
	.int	JMPTBL(L(Exit19), L(ExitTable))
	.int	JMPTBL(L(Exit20), L(ExitTable))
	.int	JMPTBL(L(Exit21), L(ExitTable))
	.int	JMPTBL(L(Exit22), L(ExitTable))
	.int    JMPTBL(L(Exit23), L(ExitTable))
	.int	JMPTBL(L(Exit24), L(ExitTable))
	.int	JMPTBL(L(Exit25), L(ExitTable))
	.int	JMPTBL(L(Exit26), L(ExitTable))
	.int	JMPTBL(L(Exit27), L(ExitTable))
	.int	JMPTBL(L(Exit28), L(ExitTable))
	.int	JMPTBL(L(Exit29), L(ExitTable))
	.int	JMPTBL(L(Exit30), L(ExitTable))
	.int	JMPTBL(L(Exit31), L(ExitTable))
	.int	JMPTBL(L(Exit32), L(ExitTable))
# ifdef USE_AS_STRNCPY
L(ExitStrncpyTable):
	.int	JMPTBL(L(StrncpyExit0), L(ExitStrncpyTable))
	.int	JMPTBL(L(StrncpyExit1), L(ExitStrncpyTable))
	.int	JMPTBL(L(StrncpyExit2), L(ExitStrncpyTable))
	.int	JMPTBL(L(StrncpyExit3), L(ExitStrncpyTable))
	.int	JMPTBL(L(StrncpyExit4), L(ExitStrncpyTable))
	.int	JMPTBL(L(StrncpyExit5), L(ExitStrncpyTable))
	.int	JMPTBL(L(StrncpyExit6), L(ExitStrncpyTable))
	.int	JMPTBL(L(StrncpyExit7), L(ExitStrncpyTable))
	.int	JMPTBL(L(StrncpyExit8), L(ExitStrncpyTable))
	.int	JMPTBL(L(StrncpyExit9), L(ExitStrncpyTable))
	.int	JMPTBL(L(StrncpyExit10), L(ExitStrncpyTable))
	.int	JMPTBL(L(StrncpyExit11), L(ExitStrncpyTable))
	.int	JMPTBL(L(StrncpyExit12), L(ExitStrncpyTable))
	.int	JMPTBL(L(StrncpyExit13), L(ExitStrncpyTable))
	.int	JMPTBL(L(StrncpyExit14), L(ExitStrncpyTable))
	.int	JMPTBL(L(StrncpyExit15), L(ExitStrncpyTable))
	.int	JMPTBL(L(StrncpyExit16), L(ExitStrncpyTable))
	.int	JMPTBL(L(StrncpyExit17), L(ExitStrncpyTable))
	.int	JMPTBL(L(StrncpyExit18), L(ExitStrncpyTable))
	.int	JMPTBL(L(StrncpyExit19), L(ExitStrncpyTable))
	.int	JMPTBL(L(StrncpyExit20), L(ExitStrncpyTable))
	.int	JMPTBL(L(StrncpyExit21), L(ExitStrncpyTable))
	.int	JMPTBL(L(StrncpyExit22), L(ExitStrncpyTable))
	.int    JMPTBL(L(StrncpyExit23), L(ExitStrncpyTable))
	.int	JMPTBL(L(StrncpyExit24), L(ExitStrncpyTable))
	.int	JMPTBL(L(StrncpyExit25), L(ExitStrncpyTable))
	.int	JMPTBL(L(StrncpyExit26), L(ExitStrncpyTable))
	.int	JMPTBL(L(StrncpyExit27), L(ExitStrncpyTable))
	.int	JMPTBL(L(StrncpyExit28), L(ExitStrncpyTable))
	.int	JMPTBL(L(StrncpyExit29), L(ExitStrncpyTable))
	.int	JMPTBL(L(StrncpyExit30), L(ExitStrncpyTable))
	.int	JMPTBL(L(StrncpyExit31), L(ExitStrncpyTable))
	.int	JMPTBL(L(StrncpyExit32), L(ExitStrncpyTable))
	.int	JMPTBL(L(StrncpyExit33), L(ExitStrncpyTable))
#  ifndef USE_AS_STRCAT
	.p2align 4
L(FillTable):
	.int	JMPTBL(L(Fill0), L(FillTable))
	.int	JMPTBL(L(Fill1), L(FillTable))
	.int	JMPTBL(L(Fill2), L(FillTable))
	.int	JMPTBL(L(Fill3), L(FillTable))
	.int	JMPTBL(L(Fill4), L(FillTable))
	.int	JMPTBL(L(Fill5), L(FillTable))
	.int	JMPTBL(L(Fill6), L(FillTable))
	.int	JMPTBL(L(Fill7), L(FillTable))
	.int	JMPTBL(L(Fill8), L(FillTable))
	.int	JMPTBL(L(Fill9), L(FillTable))
	.int	JMPTBL(L(Fill10), L(FillTable))
	.int	JMPTBL(L(Fill11), L(FillTable))
	.int	JMPTBL(L(Fill12), L(FillTable))
	.int	JMPTBL(L(Fill13), L(FillTable))
	.int	JMPTBL(L(Fill14), L(FillTable))
	.int	JMPTBL(L(Fill15), L(FillTable))
	.int	JMPTBL(L(Fill16), L(FillTable))
#  endif
# endif
#endif
