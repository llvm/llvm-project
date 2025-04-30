/* strcpy with 256-bit EVEX instructions.
   Copyright (C) 2021 Free Software Foundation, Inc.
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
#   define STRCPY  __strcpy_evex
#  endif

# endif

# define VMOVU		vmovdqu64
# define VMOVA		vmovdqa64

/* Number of bytes in a vector register */
# ifndef VEC_SIZE
#  define VEC_SIZE	32
# endif

# define XMM2		xmm18
# define XMM3		xmm19

# define YMM2		ymm18
# define YMM3		ymm19
# define YMM4		ymm20
# define YMM5		ymm21
# define YMM6		ymm22
# define YMM7		ymm23

# ifndef USE_AS_STRCAT

/* zero register */
#  define XMMZERO	xmm16
#  define YMMZERO	ymm16
#  define YMM1		ymm17

	.section .text.evex,"ax",@progbits
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

	vpxorq	%XMMZERO, %XMMZERO, %XMMZERO
# endif

	and	$((VEC_SIZE * 4) - 1), %ecx
	cmp	$(VEC_SIZE * 2), %ecx
	jbe	L(SourceStringAlignmentLessTwoVecSize)

	and	$-VEC_SIZE, %rsi
	and	$(VEC_SIZE - 1), %ecx

	vpcmpb	$0, (%rsi), %YMMZERO, %k0
	kmovd	%k0, %edx
	shr	%cl, %rdx

# ifdef USE_AS_STRNCPY
#  if defined USE_AS_STPCPY || defined USE_AS_STRCAT
	mov	$VEC_SIZE, %r10
	sub	%rcx, %r10
	cmp	%r10, %r8
#  else
	mov	$(VEC_SIZE + 1), %r10
	sub	%rcx, %r10
	cmp	%r10, %r8
#  endif
	jbe	L(CopyVecSizeTailCase2OrCase3)
# endif
	test	%edx, %edx
	jnz	L(CopyVecSizeTail)

	vpcmpb	$0, VEC_SIZE(%rsi), %YMMZERO, %k1
	kmovd	%k1, %edx

# ifdef USE_AS_STRNCPY
	add	$VEC_SIZE, %r10
	cmp	%r10, %r8
	jbe	L(CopyTwoVecSizeCase2OrCase3)
# endif
	test	%edx, %edx
	jnz	L(CopyTwoVecSize)

	VMOVU	(%rsi, %rcx), %YMM2   /* copy VEC_SIZE bytes */
	VMOVU	%YMM2, (%rdi)

/* If source address alignment != destination address alignment */
	.p2align 4
L(UnalignVecSizeBoth):
	sub	%rcx, %rdi
# ifdef USE_AS_STRNCPY
	add	%rcx, %r8
	sbb	%rcx, %rcx
	or	%rcx, %r8
# endif
	mov	$VEC_SIZE, %rcx
	VMOVA	(%rsi, %rcx), %YMM2
	VMOVU	%YMM2, (%rdi, %rcx)
	VMOVA	VEC_SIZE(%rsi, %rcx), %YMM2
	vpcmpb	$0, %YMM2, %YMMZERO, %k0
	kmovd	%k0, %edx
	add	$VEC_SIZE, %rcx
# ifdef USE_AS_STRNCPY
	sub	$(VEC_SIZE * 3), %r8
	jbe	L(CopyVecSizeCase2OrCase3)
# endif
	test	%edx, %edx
# if defined USE_AS_STRNCPY && !defined USE_AS_STRCAT
	jnz	L(CopyVecSizeUnalignedVec2)
# else
	jnz	L(CopyVecSize)
# endif

	VMOVU	%YMM2, (%rdi, %rcx)
	VMOVA	VEC_SIZE(%rsi, %rcx), %YMM3
	vpcmpb	$0, %YMM3, %YMMZERO, %k0
	kmovd	%k0, %edx
	add	$VEC_SIZE, %rcx
# ifdef USE_AS_STRNCPY
	sub	$VEC_SIZE, %r8
	jbe	L(CopyVecSizeCase2OrCase3)
# endif
	test	%edx, %edx
# if defined USE_AS_STRNCPY && !defined USE_AS_STRCAT
	jnz	L(CopyVecSizeUnalignedVec3)
# else
	jnz	L(CopyVecSize)
# endif

	VMOVU	%YMM3, (%rdi, %rcx)
	VMOVA	VEC_SIZE(%rsi, %rcx), %YMM4
	vpcmpb	$0, %YMM4, %YMMZERO, %k0
	kmovd	%k0, %edx
	add	$VEC_SIZE, %rcx
# ifdef USE_AS_STRNCPY
	sub	$VEC_SIZE, %r8
	jbe	L(CopyVecSizeCase2OrCase3)
# endif
	test	%edx, %edx
# if defined USE_AS_STRNCPY && !defined USE_AS_STRCAT
	jnz	L(CopyVecSizeUnalignedVec4)
# else
	jnz	L(CopyVecSize)
# endif

	VMOVU	%YMM4, (%rdi, %rcx)
	VMOVA	VEC_SIZE(%rsi, %rcx), %YMM2
	vpcmpb	$0, %YMM2, %YMMZERO, %k0
	kmovd	%k0, %edx
	add	$VEC_SIZE, %rcx
# ifdef USE_AS_STRNCPY
	sub	$VEC_SIZE, %r8
	jbe	L(CopyVecSizeCase2OrCase3)
# endif
	test	%edx, %edx
# if defined USE_AS_STRNCPY && !defined USE_AS_STRCAT
	jnz	L(CopyVecSizeUnalignedVec2)
# else
	jnz	L(CopyVecSize)
# endif

	VMOVU	%YMM2, (%rdi, %rcx)
	VMOVA	VEC_SIZE(%rsi, %rcx), %YMM2
	vpcmpb	$0, %YMM2, %YMMZERO, %k0
	kmovd	%k0, %edx
	add	$VEC_SIZE, %rcx
# ifdef USE_AS_STRNCPY
	sub	$VEC_SIZE, %r8
	jbe	L(CopyVecSizeCase2OrCase3)
# endif
	test	%edx, %edx
# if defined USE_AS_STRNCPY && !defined USE_AS_STRCAT
	jnz	L(CopyVecSizeUnalignedVec2)
# else
	jnz	L(CopyVecSize)
# endif

	VMOVA	VEC_SIZE(%rsi, %rcx), %YMM3
	VMOVU	%YMM2, (%rdi, %rcx)
	vpcmpb	$0, %YMM3, %YMMZERO, %k0
	kmovd	%k0, %edx
	add	$VEC_SIZE, %rcx
# ifdef USE_AS_STRNCPY
	sub	$VEC_SIZE, %r8
	jbe	L(CopyVecSizeCase2OrCase3)
# endif
	test	%edx, %edx
# if defined USE_AS_STRNCPY && !defined USE_AS_STRCAT
	jnz	L(CopyVecSizeUnalignedVec3)
# else
	jnz	L(CopyVecSize)
# endif

	VMOVU	%YMM3, (%rdi, %rcx)
	mov	%rsi, %rdx
	lea	VEC_SIZE(%rsi, %rcx), %rsi
	and	$-(VEC_SIZE * 4), %rsi
	sub	%rsi, %rdx
	sub	%rdx, %rdi
# ifdef USE_AS_STRNCPY
	lea	(VEC_SIZE * 8)(%r8, %rdx), %r8
# endif
L(UnalignedFourVecSizeLoop):
	VMOVA	(%rsi), %YMM4
	VMOVA	VEC_SIZE(%rsi), %YMM5
	VMOVA	(VEC_SIZE * 2)(%rsi), %YMM6
	VMOVA	(VEC_SIZE * 3)(%rsi), %YMM7
	vpminub	%YMM5, %YMM4, %YMM2
	vpminub	%YMM7, %YMM6, %YMM3
	vpminub	%YMM2, %YMM3, %YMM2
	/* If K7 != 0, there is a null byte.  */
	vpcmpb	$0, %YMM2, %YMMZERO, %k7
	kmovd	%k7, %edx
# ifdef USE_AS_STRNCPY
	sub	$(VEC_SIZE * 4), %r8
	jbe	L(UnalignedLeaveCase2OrCase3)
# endif
	test	%edx, %edx
	jnz	L(UnalignedFourVecSizeLeave)

L(UnalignedFourVecSizeLoop_start):
	add	$(VEC_SIZE * 4), %rdi
	add	$(VEC_SIZE * 4), %rsi
	VMOVU	%YMM4, -(VEC_SIZE * 4)(%rdi)
	VMOVA	(%rsi), %YMM4
	VMOVU	%YMM5, -(VEC_SIZE * 3)(%rdi)
	VMOVA	VEC_SIZE(%rsi), %YMM5
	vpminub	%YMM5, %YMM4, %YMM2
	VMOVU	%YMM6, -(VEC_SIZE * 2)(%rdi)
	VMOVA	(VEC_SIZE * 2)(%rsi), %YMM6
	VMOVU	%YMM7, -VEC_SIZE(%rdi)
	VMOVA	(VEC_SIZE * 3)(%rsi), %YMM7
	vpminub	%YMM7, %YMM6, %YMM3
	vpminub	%YMM2, %YMM3, %YMM2
	/* If K7 != 0, there is a null byte.  */
	vpcmpb	$0, %YMM2, %YMMZERO, %k7
	kmovd	%k7, %edx
# ifdef USE_AS_STRNCPY
	sub	$(VEC_SIZE * 4), %r8
	jbe	L(UnalignedLeaveCase2OrCase3)
# endif
	test	%edx, %edx
	jz	L(UnalignedFourVecSizeLoop_start)

L(UnalignedFourVecSizeLeave):
	vpcmpb	$0, %YMM4, %YMMZERO, %k1
	kmovd	%k1, %edx
	test	%edx, %edx
	jnz	L(CopyVecSizeUnaligned_0)

	vpcmpb	$0, %YMM5, %YMMZERO, %k2
	kmovd	%k2, %ecx
	test	%ecx, %ecx
	jnz	L(CopyVecSizeUnaligned_16)

	vpcmpb	$0, %YMM6, %YMMZERO, %k3
	kmovd	%k3, %edx
	test	%edx, %edx
	jnz	L(CopyVecSizeUnaligned_32)

	vpcmpb	$0, %YMM7, %YMMZERO, %k4
	kmovd	%k4, %ecx
	bsf	%ecx, %edx
	VMOVU	%YMM4, (%rdi)
	VMOVU	%YMM5, VEC_SIZE(%rdi)
	VMOVU	%YMM6, (VEC_SIZE * 2)(%rdi)
# if defined USE_AS_STRNCPY && !defined USE_AS_STRCAT
# ifdef USE_AS_STPCPY
	lea	(VEC_SIZE * 3)(%rdi, %rdx), %rax
# endif
	VMOVU	%YMM7, (VEC_SIZE * 3)(%rdi)
	add	$(VEC_SIZE - 1), %r8
	sub	%rdx, %r8
	lea	((VEC_SIZE * 3) + 1)(%rdi, %rdx), %rdi
	jmp	L(StrncpyFillTailWithZero)
# else
	add	$(VEC_SIZE * 3), %rsi
	add	$(VEC_SIZE * 3), %rdi
	jmp	L(CopyVecSizeExit)
# endif

/* If source address alignment == destination address alignment */

L(SourceStringAlignmentLessTwoVecSize):
	VMOVU	(%rsi), %YMM3
	VMOVU	VEC_SIZE(%rsi), %YMM2
	vpcmpb	$0, %YMM3, %YMMZERO, %k0
	kmovd	%k0, %edx

# ifdef USE_AS_STRNCPY
#  if defined USE_AS_STPCPY || defined USE_AS_STRCAT
	cmp	$VEC_SIZE, %r8
#  else
	cmp	$(VEC_SIZE + 1), %r8
#  endif
	jbe	L(CopyVecSizeTail1Case2OrCase3)
# endif
	test	%edx, %edx
	jnz	L(CopyVecSizeTail1)

	VMOVU	%YMM3, (%rdi)
	vpcmpb	$0, %YMM2, %YMMZERO, %k0
	kmovd	%k0, %edx

# ifdef USE_AS_STRNCPY
#  if defined USE_AS_STPCPY || defined USE_AS_STRCAT
	cmp	$(VEC_SIZE * 2), %r8
#  else
	cmp	$((VEC_SIZE * 2) + 1), %r8
#  endif
	jbe	L(CopyTwoVecSize1Case2OrCase3)
# endif
	test	%edx, %edx
	jnz	L(CopyTwoVecSize1)

	and	$-VEC_SIZE, %rsi
	and	$(VEC_SIZE - 1), %ecx
	jmp	L(UnalignVecSizeBoth)

/*------End of main part with loops---------------------*/

/* Case1 */

# if (!defined USE_AS_STRNCPY) || (defined USE_AS_STRCAT)
	.p2align 4
L(CopyVecSize):
	add	%rcx, %rdi
# endif
L(CopyVecSizeTail):
	add	%rcx, %rsi
L(CopyVecSizeTail1):
	bsf	%edx, %edx
L(CopyVecSizeExit):
	cmp	$32, %edx
	jae	L(Exit32_63)
	cmp	$16, %edx
	jae	L(Exit16_31)
	cmp	$8, %edx
	jae	L(Exit8_15)
	cmp	$4, %edx
	jae	L(Exit4_7)
	cmp	$3, %edx
	je	L(Exit3)
	cmp	$1, %edx
	ja	L(Exit2)
	je	L(Exit1)
	movb	$0, (%rdi)
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
L(CopyTwoVecSize1):
	add	$VEC_SIZE, %rsi
	add	$VEC_SIZE, %rdi
# if defined USE_AS_STRNCPY && !defined USE_AS_STRCAT
	sub	$VEC_SIZE, %r8
# endif
	jmp	L(CopyVecSizeTail1)

	.p2align 4
L(CopyTwoVecSize):
	bsf	%edx, %edx
	add	%rcx, %rsi
	add	$VEC_SIZE, %edx
	sub	%ecx, %edx
	jmp	L(CopyVecSizeExit)

	.p2align 4
L(CopyVecSizeUnaligned_0):
	bsf	%edx, %edx
# if defined USE_AS_STRNCPY && !defined USE_AS_STRCAT
# ifdef USE_AS_STPCPY
	lea	(%rdi, %rdx), %rax
# endif
	VMOVU	%YMM4, (%rdi)
	add	$((VEC_SIZE * 4) - 1), %r8
	sub	%rdx, %r8
	lea	1(%rdi, %rdx), %rdi
	jmp	L(StrncpyFillTailWithZero)
# else
	jmp	L(CopyVecSizeExit)
# endif

	.p2align 4
L(CopyVecSizeUnaligned_16):
	bsf	%ecx, %edx
	VMOVU	%YMM4, (%rdi)
# if defined USE_AS_STRNCPY && !defined USE_AS_STRCAT
# ifdef USE_AS_STPCPY
	lea	VEC_SIZE(%rdi, %rdx), %rax
# endif
	VMOVU	%YMM5, VEC_SIZE(%rdi)
	add	$((VEC_SIZE * 3) - 1), %r8
	sub	%rdx, %r8
	lea	(VEC_SIZE + 1)(%rdi, %rdx), %rdi
	jmp	L(StrncpyFillTailWithZero)
# else
	add	$VEC_SIZE, %rsi
	add	$VEC_SIZE, %rdi
	jmp	L(CopyVecSizeExit)
# endif

	.p2align 4
L(CopyVecSizeUnaligned_32):
	bsf	%edx, %edx
	VMOVU	%YMM4, (%rdi)
	VMOVU	%YMM5, VEC_SIZE(%rdi)
# if defined USE_AS_STRNCPY && !defined USE_AS_STRCAT
# ifdef USE_AS_STPCPY
	lea	(VEC_SIZE * 2)(%rdi, %rdx), %rax
# endif
	VMOVU	%YMM6, (VEC_SIZE * 2)(%rdi)
	add	$((VEC_SIZE * 2) - 1), %r8
	sub	%rdx, %r8
	lea	((VEC_SIZE * 2) + 1)(%rdi, %rdx), %rdi
	jmp	L(StrncpyFillTailWithZero)
# else
	add	$(VEC_SIZE * 2), %rsi
	add	$(VEC_SIZE * 2), %rdi
	jmp	L(CopyVecSizeExit)
# endif

# ifdef USE_AS_STRNCPY
#  ifndef USE_AS_STRCAT
	.p2align 4
L(CopyVecSizeUnalignedVec6):
	VMOVU	%YMM6, (%rdi, %rcx)
	jmp	L(CopyVecSizeVecExit)

	.p2align 4
L(CopyVecSizeUnalignedVec5):
	VMOVU	%YMM5, (%rdi, %rcx)
	jmp	L(CopyVecSizeVecExit)

	.p2align 4
L(CopyVecSizeUnalignedVec4):
	VMOVU	%YMM4, (%rdi, %rcx)
	jmp	L(CopyVecSizeVecExit)

	.p2align 4
L(CopyVecSizeUnalignedVec3):
	VMOVU	%YMM3, (%rdi, %rcx)
	jmp	L(CopyVecSizeVecExit)
#  endif

/* Case2 */

	.p2align 4
L(CopyVecSizeCase2):
	add	$VEC_SIZE, %r8
	add	%rcx, %rdi
	add	%rcx, %rsi
	bsf	%edx, %edx
	cmp	%r8d, %edx
	jb	L(CopyVecSizeExit)
	jmp	L(StrncpyExit)

	.p2align 4
L(CopyTwoVecSizeCase2):
	add	%rcx, %rsi
	bsf	%edx, %edx
	add	$VEC_SIZE, %edx
	sub	%ecx, %edx
	cmp	%r8d, %edx
	jb	L(CopyVecSizeExit)
	jmp	L(StrncpyExit)

L(CopyVecSizeTailCase2):
	add	%rcx, %rsi
	bsf	%edx, %edx
	cmp	%r8d, %edx
	jb	L(CopyVecSizeExit)
	jmp	L(StrncpyExit)

L(CopyVecSizeTail1Case2):
	bsf	%edx, %edx
	cmp	%r8d, %edx
	jb	L(CopyVecSizeExit)
	jmp	L(StrncpyExit)

/* Case2 or Case3,  Case3 */

	.p2align 4
L(CopyVecSizeCase2OrCase3):
	test	%rdx, %rdx
	jnz	L(CopyVecSizeCase2)
L(CopyVecSizeCase3):
	add	$VEC_SIZE, %r8
	add	%rcx, %rdi
	add	%rcx, %rsi
	jmp	L(StrncpyExit)

	.p2align 4
L(CopyTwoVecSizeCase2OrCase3):
	test	%rdx, %rdx
	jnz	L(CopyTwoVecSizeCase2)
	add	%rcx, %rsi
	jmp	L(StrncpyExit)

	.p2align 4
L(CopyVecSizeTailCase2OrCase3):
	test	%rdx, %rdx
	jnz	L(CopyVecSizeTailCase2)
	add	%rcx, %rsi
	jmp	L(StrncpyExit)

	.p2align 4
L(CopyTwoVecSize1Case2OrCase3):
	add	$VEC_SIZE, %rdi
	add	$VEC_SIZE, %rsi
	sub	$VEC_SIZE, %r8
L(CopyVecSizeTail1Case2OrCase3):
	test	%rdx, %rdx
	jnz	L(CopyVecSizeTail1Case2)
	jmp	L(StrncpyExit)
# endif

/*------------End labels regarding with copying 1-VEC_SIZE bytes--and 1-(VEC_SIZE*2) bytes----*/

	.p2align 4
L(Exit1):
	movzwl	(%rsi), %edx
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
L(Exit2):
	movzwl	(%rsi), %ecx
	mov	%cx, (%rdi)
	movb	$0, 2(%rdi)
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
L(Exit3):
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
L(Exit4_7):
	mov	(%rsi), %ecx
	mov	%ecx, (%rdi)
	mov	-3(%rsi, %rdx), %ecx
	mov	%ecx, -3(%rdi, %rdx)
# ifdef USE_AS_STPCPY
	lea	(%rdi, %rdx), %rax
# endif
# if defined USE_AS_STRNCPY && !defined USE_AS_STRCAT
	sub	%rdx, %r8
	sub	$1, %r8
	lea	1(%rdi, %rdx), %rdi
	jnz	L(StrncpyFillTailWithZero)
# endif
	ret

	.p2align 4
L(Exit8_15):
	mov	(%rsi), %rcx
	mov	-7(%rsi, %rdx), %r9
	mov	%rcx, (%rdi)
	mov	%r9, -7(%rdi, %rdx)
# ifdef USE_AS_STPCPY
	lea	(%rdi, %rdx), %rax
# endif
# if defined USE_AS_STRNCPY && !defined USE_AS_STRCAT
	sub	%rdx, %r8
	sub	$1, %r8
	lea	1(%rdi, %rdx), %rdi
	jnz	L(StrncpyFillTailWithZero)
# endif
	ret

	.p2align 4
L(Exit16_31):
	VMOVU	(%rsi), %XMM2
	VMOVU	-15(%rsi, %rdx), %XMM3
	VMOVU	%XMM2, (%rdi)
	VMOVU	%XMM3, -15(%rdi, %rdx)
# ifdef USE_AS_STPCPY
	lea	(%rdi, %rdx), %rax
# endif
# if defined USE_AS_STRNCPY && !defined USE_AS_STRCAT
	sub %rdx, %r8
	sub $1, %r8
	lea 1(%rdi, %rdx), %rdi
	jnz L(StrncpyFillTailWithZero)
# endif
	ret

	.p2align 4
L(Exit32_63):
	VMOVU	(%rsi), %YMM2
	VMOVU	-31(%rsi, %rdx), %YMM3
	VMOVU	%YMM2, (%rdi)
	VMOVU	%YMM3, -31(%rdi, %rdx)
# ifdef USE_AS_STPCPY
	lea	(%rdi, %rdx), %rax
# endif
# if defined USE_AS_STRNCPY && !defined USE_AS_STRCAT
	sub	%rdx, %r8
	sub	$1, %r8
	lea	1(%rdi, %rdx), %rdi
	jnz	L(StrncpyFillTailWithZero)
# endif
	ret

# ifdef USE_AS_STRNCPY

	.p2align 4
L(StrncpyExit1):
	movzbl	(%rsi), %edx
	mov	%dl, (%rdi)
#  ifdef USE_AS_STPCPY
	lea	1(%rdi), %rax
#  endif
#  ifdef USE_AS_STRCAT
	movb	$0, 1(%rdi)
#  endif
	ret

	.p2align 4
L(StrncpyExit2):
	movzwl	(%rsi), %edx
	mov	%dx, (%rdi)
#  ifdef USE_AS_STPCPY
	lea	2(%rdi), %rax
#  endif
#  ifdef USE_AS_STRCAT
	movb	$0, 2(%rdi)
#  endif
	ret

	.p2align 4
L(StrncpyExit3_4):
	movzwl	(%rsi), %ecx
	movzwl	-2(%rsi, %r8), %edx
	mov	%cx, (%rdi)
	mov	%dx, -2(%rdi, %r8)
#  ifdef USE_AS_STPCPY
	lea	(%rdi, %r8), %rax
#  endif
#  ifdef USE_AS_STRCAT
	movb	$0, (%rdi, %r8)
#  endif
	ret

	.p2align 4
L(StrncpyExit5_8):
	mov	(%rsi), %ecx
	mov	-4(%rsi, %r8), %edx
	mov	%ecx, (%rdi)
	mov	%edx, -4(%rdi, %r8)
#  ifdef USE_AS_STPCPY
	lea	(%rdi, %r8), %rax
#  endif
#  ifdef USE_AS_STRCAT
	movb	$0, (%rdi, %r8)
#  endif
	ret

	.p2align 4
L(StrncpyExit9_16):
	mov	(%rsi), %rcx
	mov	-8(%rsi, %r8), %rdx
	mov	%rcx, (%rdi)
	mov	%rdx, -8(%rdi, %r8)
#  ifdef USE_AS_STPCPY
	lea	(%rdi, %r8), %rax
#  endif
#  ifdef USE_AS_STRCAT
	movb	$0, (%rdi, %r8)
#  endif
	ret

	.p2align 4
L(StrncpyExit17_32):
	VMOVU	(%rsi), %XMM2
	VMOVU	-16(%rsi, %r8), %XMM3
	VMOVU	%XMM2, (%rdi)
	VMOVU	%XMM3, -16(%rdi, %r8)
#  ifdef USE_AS_STPCPY
	lea	(%rdi, %r8), %rax
#  endif
#  ifdef USE_AS_STRCAT
	movb	$0, (%rdi, %r8)
#  endif
	ret

	.p2align 4
L(StrncpyExit33_64):
	/*  0/32, 31/16 */
	VMOVU	(%rsi), %YMM2
	VMOVU	-VEC_SIZE(%rsi, %r8), %YMM3
	VMOVU	%YMM2, (%rdi)
	VMOVU	%YMM3, -VEC_SIZE(%rdi, %r8)
#  ifdef USE_AS_STPCPY
	lea	(%rdi, %r8), %rax
#  endif
#  ifdef USE_AS_STRCAT
	movb	$0, (%rdi, %r8)
#  endif
	ret

	.p2align 4
L(StrncpyExit65):
	/* 0/32, 32/32, 64/1 */
	VMOVU	(%rsi), %YMM2
	VMOVU	32(%rsi), %YMM3
	mov	64(%rsi), %cl
	VMOVU	%YMM2, (%rdi)
	VMOVU	%YMM3, 32(%rdi)
	mov	%cl, 64(%rdi)
#  ifdef USE_AS_STPCPY
	lea	65(%rdi), %rax
#  endif
#  ifdef USE_AS_STRCAT
	movb	$0, 65(%rdi)
#  endif
	ret

#  ifndef USE_AS_STRCAT

	.p2align 4
L(Fill1):
	mov	%dl, (%rdi)
	ret

	.p2align 4
L(Fill2):
	mov	%dx, (%rdi)
	ret

	.p2align 4
L(Fill3_4):
	mov	%dx, (%rdi)
	mov     %dx, -2(%rdi, %r8)
	ret

	.p2align 4
L(Fill5_8):
	mov	%edx, (%rdi)
	mov     %edx, -4(%rdi, %r8)
	ret

	.p2align 4
L(Fill9_16):
	mov	%rdx, (%rdi)
	mov	%rdx, -8(%rdi, %r8)
	ret

	.p2align 4
L(Fill17_32):
	VMOVU	%XMMZERO, (%rdi)
	VMOVU	%XMMZERO, -16(%rdi, %r8)
	ret

	.p2align 4
L(CopyVecSizeUnalignedVec2):
	VMOVU	%YMM2, (%rdi, %rcx)

	.p2align 4
L(CopyVecSizeVecExit):
	bsf	%edx, %edx
	add	$(VEC_SIZE - 1), %r8
	add	%rcx, %rdi
#   ifdef USE_AS_STPCPY
	lea	(%rdi, %rdx), %rax
#   endif
	sub	%rdx, %r8
	lea	1(%rdi, %rdx), %rdi

	.p2align 4
L(StrncpyFillTailWithZero):
	xor	%edx, %edx
	sub	$VEC_SIZE, %r8
	jbe	L(StrncpyFillExit)

	VMOVU	%YMMZERO, (%rdi)
	add	$VEC_SIZE, %rdi

	mov	%rdi, %rsi
	and	$(VEC_SIZE - 1), %esi
	sub	%rsi, %rdi
	add	%rsi, %r8
	sub	$(VEC_SIZE * 4), %r8
	jb	L(StrncpyFillLessFourVecSize)

L(StrncpyFillLoopVmovdqa):
	VMOVA	%YMMZERO, (%rdi)
	VMOVA	%YMMZERO, VEC_SIZE(%rdi)
	VMOVA	%YMMZERO, (VEC_SIZE * 2)(%rdi)
	VMOVA	%YMMZERO, (VEC_SIZE * 3)(%rdi)
	add	$(VEC_SIZE * 4), %rdi
	sub	$(VEC_SIZE * 4), %r8
	jae	L(StrncpyFillLoopVmovdqa)

L(StrncpyFillLessFourVecSize):
	add	$(VEC_SIZE * 2), %r8
	jl	L(StrncpyFillLessTwoVecSize)
	VMOVA	%YMMZERO, (%rdi)
	VMOVA	%YMMZERO, VEC_SIZE(%rdi)
	add	$(VEC_SIZE * 2), %rdi
	sub	$VEC_SIZE, %r8
	jl	L(StrncpyFillExit)
	VMOVA	%YMMZERO, (%rdi)
	add	$VEC_SIZE, %rdi
	jmp	L(Fill)

	.p2align 4
L(StrncpyFillLessTwoVecSize):
	add	$VEC_SIZE, %r8
	jl	L(StrncpyFillExit)
	VMOVA	%YMMZERO, (%rdi)
	add	$VEC_SIZE, %rdi
	jmp	L(Fill)

	.p2align 4
L(StrncpyFillExit):
	add	$VEC_SIZE, %r8
L(Fill):
	cmp	$17, %r8d
	jae	L(Fill17_32)
	cmp	$9, %r8d
	jae	L(Fill9_16)
	cmp	$5, %r8d
	jae	L(Fill5_8)
	cmp	$3, %r8d
	jae	L(Fill3_4)
	cmp	$1, %r8d
	ja	L(Fill2)
	je	L(Fill1)
	ret

/* end of ifndef USE_AS_STRCAT */
#  endif

	.p2align 4
L(UnalignedLeaveCase2OrCase3):
	test	%rdx, %rdx
	jnz	L(UnalignedFourVecSizeLeaveCase2)
L(UnalignedFourVecSizeLeaveCase3):
	lea	(VEC_SIZE * 4)(%r8), %rcx
	and	$-VEC_SIZE, %rcx
	add	$(VEC_SIZE * 3), %r8
	jl	L(CopyVecSizeCase3)
	VMOVU	%YMM4, (%rdi)
	sub	$VEC_SIZE, %r8
	jb	L(CopyVecSizeCase3)
	VMOVU	%YMM5, VEC_SIZE(%rdi)
	sub	$VEC_SIZE, %r8
	jb	L(CopyVecSizeCase3)
	VMOVU	%YMM6, (VEC_SIZE * 2)(%rdi)
	sub	$VEC_SIZE, %r8
	jb	L(CopyVecSizeCase3)
	VMOVU	%YMM7, (VEC_SIZE * 3)(%rdi)
#  ifdef USE_AS_STPCPY
	lea	(VEC_SIZE * 4)(%rdi), %rax
#  endif
#  ifdef USE_AS_STRCAT
	movb	$0, (VEC_SIZE * 4)(%rdi)
#  endif
	ret

	.p2align 4
L(UnalignedFourVecSizeLeaveCase2):
	xor	%ecx, %ecx
	vpcmpb	$0, %YMM4, %YMMZERO, %k1
	kmovd	%k1, %edx
	add	$(VEC_SIZE * 3), %r8
	jle	L(CopyVecSizeCase2OrCase3)
	test	%edx, %edx
#  ifndef USE_AS_STRCAT
	jnz	L(CopyVecSizeUnalignedVec4)
#  else
	jnz	L(CopyVecSize)
#  endif
	vpcmpb	$0, %YMM5, %YMMZERO, %k2
	kmovd	%k2, %edx
	VMOVU	%YMM4, (%rdi)
	add	$VEC_SIZE, %rcx
	sub	$VEC_SIZE, %r8
	jbe	L(CopyVecSizeCase2OrCase3)
	test	%edx, %edx
#  ifndef USE_AS_STRCAT
	jnz	L(CopyVecSizeUnalignedVec5)
#  else
	jnz	L(CopyVecSize)
#  endif

	vpcmpb	$0, %YMM6, %YMMZERO, %k3
	kmovd	%k3, %edx
	VMOVU	%YMM5, VEC_SIZE(%rdi)
	add	$VEC_SIZE, %rcx
	sub	$VEC_SIZE, %r8
	jbe	L(CopyVecSizeCase2OrCase3)
	test	%edx, %edx
#  ifndef USE_AS_STRCAT
	jnz	L(CopyVecSizeUnalignedVec6)
#  else
	jnz	L(CopyVecSize)
#  endif

	vpcmpb	$0, %YMM7, %YMMZERO, %k4
	kmovd	%k4, %edx
	VMOVU	%YMM6, (VEC_SIZE * 2)(%rdi)
	lea	VEC_SIZE(%rdi, %rcx), %rdi
	lea	VEC_SIZE(%rsi, %rcx), %rsi
	bsf	%edx, %edx
	cmp	%r8d, %edx
	jb	L(CopyVecSizeExit)
L(StrncpyExit):
	cmp	$65, %r8d
	je	L(StrncpyExit65)
	cmp	$33, %r8d
	jae	L(StrncpyExit33_64)
	cmp	$17, %r8d
	jae	L(StrncpyExit17_32)
	cmp	$9, %r8d
	jae	L(StrncpyExit9_16)
	cmp	$5, %r8d
	jae	L(StrncpyExit5_8)
	cmp	$3, %r8d
	jae	L(StrncpyExit3_4)
	cmp	$1, %r8d
	ja	L(StrncpyExit2)
	je	L(StrncpyExit1)
#  ifdef USE_AS_STPCPY
	mov	%rdi, %rax
#  endif
#  ifdef USE_AS_STRCAT
	movb	$0, (%rdi)
#  endif
	ret

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
#endif
