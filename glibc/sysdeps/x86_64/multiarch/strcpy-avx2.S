/* strcpy with AVX2
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
#   define STRCPY  __strcpy_avx2
#  endif

# endif

/* Number of bytes in a vector register */
# ifndef VEC_SIZE
#  define VEC_SIZE	32
# endif

# ifndef VZEROUPPER
#  define VZEROUPPER	vzeroupper
# endif

# ifndef SECTION
#  define SECTION(p)	p##.avx
# endif

/* zero register */
#define xmmZ	xmm0
#define ymmZ	ymm0

/* mask register */
#define ymmM	ymm1

# ifndef USE_AS_STRCAT

	.section SECTION(.text),"ax",@progbits
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

	vpxor	%xmmZ, %xmmZ, %xmmZ

	and	$((VEC_SIZE * 4) - 1), %ecx
	cmp	$(VEC_SIZE * 2), %ecx
	jbe	L(SourceStringAlignmentLessTwoVecSize)

	and	$-VEC_SIZE, %rsi
	and	$(VEC_SIZE - 1), %ecx

	vpcmpeqb (%rsi), %ymmZ, %ymmM
	vpmovmskb %ymmM, %edx
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

	vpcmpeqb VEC_SIZE(%rsi), %ymmZ, %ymm2
	vpmovmskb %ymm2, %edx

# ifdef USE_AS_STRNCPY
	add	$VEC_SIZE, %r10
	cmp	%r10, %r8
	jbe	L(CopyTwoVecSizeCase2OrCase3)
# endif
	test	%edx, %edx
	jnz	L(CopyTwoVecSize)

	vmovdqu (%rsi, %rcx), %ymm2   /* copy VEC_SIZE bytes */
	vmovdqu %ymm2, (%rdi)

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
	vmovdqa (%rsi, %rcx), %ymm2
	vmovdqu %ymm2, (%rdi, %rcx)
	vmovdqa VEC_SIZE(%rsi, %rcx), %ymm2
	vpcmpeqb %ymm2, %ymmZ, %ymmM
	vpmovmskb %ymmM, %edx
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

	vmovdqu %ymm2, (%rdi, %rcx)
	vmovdqa VEC_SIZE(%rsi, %rcx), %ymm3
	vpcmpeqb %ymm3, %ymmZ, %ymmM
	vpmovmskb %ymmM, %edx
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

	vmovdqu %ymm3, (%rdi, %rcx)
	vmovdqa VEC_SIZE(%rsi, %rcx), %ymm4
	vpcmpeqb %ymm4, %ymmZ, %ymmM
	vpmovmskb %ymmM, %edx
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

	vmovdqu %ymm4, (%rdi, %rcx)
	vmovdqa VEC_SIZE(%rsi, %rcx), %ymm2
	vpcmpeqb %ymm2, %ymmZ, %ymmM
	vpmovmskb %ymmM, %edx
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

	vmovdqu %ymm2, (%rdi, %rcx)
	vmovdqa VEC_SIZE(%rsi, %rcx), %ymm2
	vpcmpeqb %ymm2, %ymmZ, %ymmM
	vpmovmskb %ymmM, %edx
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

	vmovdqa VEC_SIZE(%rsi, %rcx), %ymm3
	vmovdqu %ymm2, (%rdi, %rcx)
	vpcmpeqb %ymm3, %ymmZ, %ymmM
	vpmovmskb %ymmM, %edx
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

	vmovdqu %ymm3, (%rdi, %rcx)
	mov	%rsi, %rdx
	lea	VEC_SIZE(%rsi, %rcx), %rsi
	and	$-(VEC_SIZE * 4), %rsi
	sub	%rsi, %rdx
	sub	%rdx, %rdi
# ifdef USE_AS_STRNCPY
	lea	(VEC_SIZE * 8)(%r8, %rdx), %r8
# endif
L(UnalignedFourVecSizeLoop):
	vmovdqa (%rsi), %ymm4
	vmovdqa VEC_SIZE(%rsi), %ymm5
	vmovdqa (VEC_SIZE * 2)(%rsi), %ymm6
	vmovdqa (VEC_SIZE * 3)(%rsi), %ymm7
	vpminub %ymm5, %ymm4, %ymm2
	vpminub %ymm7, %ymm6, %ymm3
	vpminub %ymm2, %ymm3, %ymm3
	vpcmpeqb %ymmM, %ymm3, %ymm3
	vpmovmskb %ymm3, %edx
# ifdef USE_AS_STRNCPY
	sub	$(VEC_SIZE * 4), %r8
	jbe	L(UnalignedLeaveCase2OrCase3)
# endif
	test	%edx, %edx
	jnz	L(UnalignedFourVecSizeLeave)

L(UnalignedFourVecSizeLoop_start):
	add	$(VEC_SIZE * 4), %rdi
	add	$(VEC_SIZE * 4), %rsi
	vmovdqu %ymm4, -(VEC_SIZE * 4)(%rdi)
	vmovdqa (%rsi), %ymm4
	vmovdqu %ymm5, -(VEC_SIZE * 3)(%rdi)
	vmovdqa VEC_SIZE(%rsi), %ymm5
	vpminub %ymm5, %ymm4, %ymm2
	vmovdqu %ymm6, -(VEC_SIZE * 2)(%rdi)
	vmovdqa (VEC_SIZE * 2)(%rsi), %ymm6
	vmovdqu %ymm7, -VEC_SIZE(%rdi)
	vmovdqa (VEC_SIZE * 3)(%rsi), %ymm7
	vpminub %ymm7, %ymm6, %ymm3
	vpminub %ymm2, %ymm3, %ymm3
	vpcmpeqb %ymmM, %ymm3, %ymm3
	vpmovmskb %ymm3, %edx
# ifdef USE_AS_STRNCPY
	sub	$(VEC_SIZE * 4), %r8
	jbe	L(UnalignedLeaveCase2OrCase3)
# endif
	test	%edx, %edx
	jz	L(UnalignedFourVecSizeLoop_start)

L(UnalignedFourVecSizeLeave):
	vpcmpeqb %ymm4, %ymmZ, %ymmM
	vpmovmskb %ymmM, %edx
	test	%edx, %edx
	jnz	L(CopyVecSizeUnaligned_0)

	vpcmpeqb %ymm5, %ymmZ, %ymmM
	vpmovmskb %ymmM, %ecx
	test	%ecx, %ecx
	jnz	L(CopyVecSizeUnaligned_16)

	vpcmpeqb %ymm6, %ymmZ, %ymmM
	vpmovmskb %ymmM, %edx
	test	%edx, %edx
	jnz	L(CopyVecSizeUnaligned_32)

	vpcmpeqb %ymm7, %ymmZ, %ymmM
	vpmovmskb %ymmM, %ecx
	bsf	%ecx, %edx
	vmovdqu %ymm4, (%rdi)
	vmovdqu %ymm5, VEC_SIZE(%rdi)
	vmovdqu %ymm6, (VEC_SIZE * 2)(%rdi)
# if defined USE_AS_STRNCPY && !defined USE_AS_STRCAT
# ifdef USE_AS_STPCPY
	lea	(VEC_SIZE * 3)(%rdi, %rdx), %rax
# endif
	vmovdqu %ymm7, (VEC_SIZE * 3)(%rdi)
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
	vmovdqu (%rsi), %ymm3
	vmovdqu VEC_SIZE(%rsi), %ymm2
	vpcmpeqb %ymm3, %ymmZ, %ymmM
	vpmovmskb %ymmM, %edx

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

	vmovdqu %ymm3, (%rdi)
	vpcmpeqb %ymm2, %ymmZ, %ymmM
	vpmovmskb %ymmM, %edx

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
L(return_vzeroupper):
	ZERO_UPPER_VEC_REGISTERS_RETURN

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
	vmovdqu %ymm4, (%rdi)
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
	vmovdqu %ymm4, (%rdi)
# if defined USE_AS_STRNCPY && !defined USE_AS_STRCAT
# ifdef USE_AS_STPCPY
	lea	VEC_SIZE(%rdi, %rdx), %rax
# endif
	vmovdqu %ymm5, VEC_SIZE(%rdi)
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
	vmovdqu %ymm4, (%rdi)
	vmovdqu %ymm5, VEC_SIZE(%rdi)
# if defined USE_AS_STRNCPY && !defined USE_AS_STRCAT
# ifdef USE_AS_STPCPY
	lea	(VEC_SIZE * 2)(%rdi, %rdx), %rax
# endif
	vmovdqu %ymm6, (VEC_SIZE * 2)(%rdi)
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
	vmovdqu %ymm6, (%rdi, %rcx)
	jmp	L(CopyVecSizeVecExit)

	.p2align 4
L(CopyVecSizeUnalignedVec5):
	vmovdqu %ymm5, (%rdi, %rcx)
	jmp	L(CopyVecSizeVecExit)

	.p2align 4
L(CopyVecSizeUnalignedVec4):
	vmovdqu %ymm4, (%rdi, %rcx)
	jmp	L(CopyVecSizeVecExit)

	.p2align 4
L(CopyVecSizeUnalignedVec3):
	vmovdqu %ymm3, (%rdi, %rcx)
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
	VZEROUPPER_RETURN

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
	VZEROUPPER_RETURN

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
	VZEROUPPER_RETURN

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
	VZEROUPPER_RETURN

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
	VZEROUPPER_RETURN

	.p2align 4
L(Exit16_31):
	vmovdqu (%rsi), %xmm2
	vmovdqu -15(%rsi, %rdx), %xmm3
	vmovdqu %xmm2, (%rdi)
	vmovdqu %xmm3, -15(%rdi, %rdx)
# ifdef USE_AS_STPCPY
	lea	(%rdi, %rdx), %rax
# endif
# if defined USE_AS_STRNCPY && !defined USE_AS_STRCAT
	sub %rdx, %r8
	sub $1, %r8
	lea 1(%rdi, %rdx), %rdi
	jnz L(StrncpyFillTailWithZero)
# endif
	VZEROUPPER_RETURN

	.p2align 4
L(Exit32_63):
	vmovdqu (%rsi), %ymm2
	vmovdqu -31(%rsi, %rdx), %ymm3
	vmovdqu %ymm2, (%rdi)
	vmovdqu %ymm3, -31(%rdi, %rdx)
# ifdef USE_AS_STPCPY
	lea	(%rdi, %rdx), %rax
# endif
# if defined USE_AS_STRNCPY && !defined USE_AS_STRCAT
	sub	%rdx, %r8
	sub	$1, %r8
	lea	1(%rdi, %rdx), %rdi
	jnz	L(StrncpyFillTailWithZero)
# endif
	VZEROUPPER_RETURN

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
	VZEROUPPER_RETURN

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
	VZEROUPPER_RETURN

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
	VZEROUPPER_RETURN

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
	VZEROUPPER_RETURN

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
	VZEROUPPER_RETURN

	.p2align 4
L(StrncpyExit17_32):
	vmovdqu (%rsi), %xmm2
	vmovdqu -16(%rsi, %r8), %xmm3
	vmovdqu %xmm2, (%rdi)
	vmovdqu %xmm3, -16(%rdi, %r8)
#  ifdef USE_AS_STPCPY
	lea	(%rdi, %r8), %rax
#  endif
#  ifdef USE_AS_STRCAT
	movb	$0, (%rdi, %r8)
#  endif
	VZEROUPPER_RETURN

	.p2align 4
L(StrncpyExit33_64):
	/*  0/32, 31/16 */
	vmovdqu (%rsi), %ymm2
	vmovdqu -VEC_SIZE(%rsi, %r8), %ymm3
	vmovdqu %ymm2, (%rdi)
	vmovdqu %ymm3, -VEC_SIZE(%rdi, %r8)
#  ifdef USE_AS_STPCPY
	lea	(%rdi, %r8), %rax
#  endif
#  ifdef USE_AS_STRCAT
	movb	$0, (%rdi, %r8)
#  endif
	VZEROUPPER_RETURN

	.p2align 4
L(StrncpyExit65):
	/* 0/32, 32/32, 64/1 */
	vmovdqu (%rsi), %ymm2
	vmovdqu 32(%rsi), %ymm3
	mov	64(%rsi), %cl
	vmovdqu %ymm2, (%rdi)
	vmovdqu %ymm3, 32(%rdi)
	mov	%cl, 64(%rdi)
#  ifdef USE_AS_STPCPY
	lea	65(%rdi), %rax
#  endif
#  ifdef USE_AS_STRCAT
	movb	$0, 65(%rdi)
#  endif
	VZEROUPPER_RETURN

#  ifndef USE_AS_STRCAT

	.p2align 4
L(Fill1):
	mov	%dl, (%rdi)
	VZEROUPPER_RETURN

	.p2align 4
L(Fill2):
	mov	%dx, (%rdi)
	VZEROUPPER_RETURN

	.p2align 4
L(Fill3_4):
	mov	%dx, (%rdi)
	mov     %dx, -2(%rdi, %r8)
	VZEROUPPER_RETURN

	.p2align 4
L(Fill5_8):
	mov	%edx, (%rdi)
	mov     %edx, -4(%rdi, %r8)
	VZEROUPPER_RETURN

	.p2align 4
L(Fill9_16):
	mov	%rdx, (%rdi)
	mov	%rdx, -8(%rdi, %r8)
	VZEROUPPER_RETURN

	.p2align 4
L(Fill17_32):
	vmovdqu %xmmZ, (%rdi)
	vmovdqu %xmmZ, -16(%rdi, %r8)
	VZEROUPPER_RETURN

	.p2align 4
L(CopyVecSizeUnalignedVec2):
	vmovdqu %ymm2, (%rdi, %rcx)

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

	vmovdqu %ymmZ, (%rdi)
	add	$VEC_SIZE, %rdi

	mov	%rdi, %rsi
	and	$(VEC_SIZE - 1), %esi
	sub	%rsi, %rdi
	add	%rsi, %r8
	sub	$(VEC_SIZE * 4), %r8
	jb	L(StrncpyFillLessFourVecSize)

L(StrncpyFillLoopVmovdqa):
	vmovdqa %ymmZ, (%rdi)
	vmovdqa %ymmZ, VEC_SIZE(%rdi)
	vmovdqa %ymmZ, (VEC_SIZE * 2)(%rdi)
	vmovdqa %ymmZ, (VEC_SIZE * 3)(%rdi)
	add	$(VEC_SIZE * 4), %rdi
	sub	$(VEC_SIZE * 4), %r8
	jae	L(StrncpyFillLoopVmovdqa)

L(StrncpyFillLessFourVecSize):
	add	$(VEC_SIZE * 2), %r8
	jl	L(StrncpyFillLessTwoVecSize)
	vmovdqa %ymmZ, (%rdi)
	vmovdqa %ymmZ, VEC_SIZE(%rdi)
	add	$(VEC_SIZE * 2), %rdi
	sub	$VEC_SIZE, %r8
	jl	L(StrncpyFillExit)
	vmovdqa %ymmZ, (%rdi)
	add	$VEC_SIZE, %rdi
	jmp	L(Fill)

	.p2align 4
L(StrncpyFillLessTwoVecSize):
	add	$VEC_SIZE, %r8
	jl	L(StrncpyFillExit)
	vmovdqa %ymmZ, (%rdi)
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
	VZEROUPPER_RETURN

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
	vmovdqu %ymm4, (%rdi)
	sub	$VEC_SIZE, %r8
	jb	L(CopyVecSizeCase3)
	vmovdqu %ymm5, VEC_SIZE(%rdi)
	sub	$VEC_SIZE, %r8
	jb	L(CopyVecSizeCase3)
	vmovdqu %ymm6, (VEC_SIZE * 2)(%rdi)
	sub	$VEC_SIZE, %r8
	jb	L(CopyVecSizeCase3)
	vmovdqu %ymm7, (VEC_SIZE * 3)(%rdi)
#  ifdef USE_AS_STPCPY
	lea	(VEC_SIZE * 4)(%rdi), %rax
#  endif
#  ifdef USE_AS_STRCAT
	movb	$0, (VEC_SIZE * 4)(%rdi)
#  endif
	VZEROUPPER_RETURN

	.p2align 4
L(UnalignedFourVecSizeLeaveCase2):
	xor	%ecx, %ecx
	vpcmpeqb %ymm4, %ymmZ, %ymmM
	vpmovmskb %ymmM, %edx
	add	$(VEC_SIZE * 3), %r8
	jle	L(CopyVecSizeCase2OrCase3)
	test	%edx, %edx
#  ifndef USE_AS_STRCAT
	jnz	L(CopyVecSizeUnalignedVec4)
#  else
	jnz	L(CopyVecSize)
#  endif
	vpcmpeqb %ymm5, %ymmZ, %ymmM
	vpmovmskb %ymmM, %edx
	vmovdqu %ymm4, (%rdi)
	add	$VEC_SIZE, %rcx
	sub	$VEC_SIZE, %r8
	jbe	L(CopyVecSizeCase2OrCase3)
	test	%edx, %edx
#  ifndef USE_AS_STRCAT
	jnz	L(CopyVecSizeUnalignedVec5)
#  else
	jnz	L(CopyVecSize)
#  endif

	vpcmpeqb %ymm6, %ymmZ, %ymmM
	vpmovmskb %ymmM, %edx
	vmovdqu %ymm5, VEC_SIZE(%rdi)
	add	$VEC_SIZE, %rcx
	sub	$VEC_SIZE, %r8
	jbe	L(CopyVecSizeCase2OrCase3)
	test	%edx, %edx
#  ifndef USE_AS_STRCAT
	jnz	L(CopyVecSizeUnalignedVec6)
#  else
	jnz	L(CopyVecSize)
#  endif

	vpcmpeqb %ymm7, %ymmZ, %ymmM
	vpmovmskb %ymmM, %edx
	vmovdqu %ymm6, (VEC_SIZE * 2)(%rdi)
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
	VZEROUPPER_RETURN

	.p2align 4
L(ExitZero):
#  ifndef USE_AS_STRCAT
	mov	%rdi, %rax
#  endif
	VZEROUPPER_RETURN

# endif

# ifndef USE_AS_STRCAT
END (STRCPY)
# else
END (STRCAT)
# endif
#endif
