/* memrchr optimized with AVX2.
   Copyright (C) 2017-2021 Free Software Foundation, Inc.
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

# ifndef MEMRCHR
#  define MEMRCHR	__memrchr_avx2
# endif

# ifndef VZEROUPPER
#  define VZEROUPPER	vzeroupper
# endif

# ifndef SECTION
#  define SECTION(p)	p##.avx
# endif

# define VEC_SIZE 32

	.section SECTION(.text),"ax",@progbits
ENTRY (MEMRCHR)
	/* Broadcast CHAR to YMM0.  */
	vmovd	%esi, %xmm0
	vpbroadcastb %xmm0, %ymm0

	sub	$VEC_SIZE, %RDX_LP
	jbe	L(last_vec_or_less)

	add	%RDX_LP, %RDI_LP

	/* Check the last VEC_SIZE bytes.  */
	vpcmpeqb (%rdi), %ymm0, %ymm1
	vpmovmskb %ymm1, %eax
	testl	%eax, %eax
	jnz	L(last_vec_x0)

	subq	$(VEC_SIZE * 4), %rdi
	movl	%edi, %ecx
	andl	$(VEC_SIZE - 1), %ecx
	jz	L(aligned_more)

	/* Align data for aligned loads in the loop.  */
	addq	$VEC_SIZE, %rdi
	addq	$VEC_SIZE, %rdx
	andq	$-VEC_SIZE, %rdi
	subq	%rcx, %rdx

	.p2align 4
L(aligned_more):
	subq	$(VEC_SIZE * 4), %rdx
	jbe	L(last_4x_vec_or_less)

	/* Check the last 4 * VEC_SIZE.  Only one VEC_SIZE at a time
	   since data is only aligned to VEC_SIZE.  */
	vpcmpeqb (VEC_SIZE * 3)(%rdi), %ymm0, %ymm1
	vpmovmskb %ymm1, %eax
	testl	%eax, %eax
	jnz	L(last_vec_x3)

	vpcmpeqb (VEC_SIZE * 2)(%rdi), %ymm0, %ymm2
	vpmovmskb %ymm2, %eax
	testl	%eax, %eax
	jnz	L(last_vec_x2)

	vpcmpeqb VEC_SIZE(%rdi), %ymm0, %ymm3
	vpmovmskb %ymm3, %eax
	testl	%eax, %eax
	jnz	L(last_vec_x1)

	vpcmpeqb (%rdi), %ymm0, %ymm4
	vpmovmskb %ymm4, %eax
	testl	%eax, %eax
	jnz	L(last_vec_x0)

	/* Align data to 4 * VEC_SIZE for loop with fewer branches.
	   There are some overlaps with above if data isn't aligned
	   to 4 * VEC_SIZE.  */
	movl	%edi, %ecx
	andl	$(VEC_SIZE * 4 - 1), %ecx
	jz	L(loop_4x_vec)

	addq	$(VEC_SIZE * 4), %rdi
	addq	$(VEC_SIZE * 4), %rdx
	andq	$-(VEC_SIZE * 4), %rdi
	subq	%rcx, %rdx

	.p2align 4
L(loop_4x_vec):
	/* Compare 4 * VEC at a time forward.  */
	subq	$(VEC_SIZE * 4), %rdi
	subq	$(VEC_SIZE * 4), %rdx
	jbe	L(last_4x_vec_or_less)

	vmovdqa	(%rdi), %ymm1
	vmovdqa	VEC_SIZE(%rdi), %ymm2
	vmovdqa	(VEC_SIZE * 2)(%rdi), %ymm3
	vmovdqa	(VEC_SIZE * 3)(%rdi), %ymm4

	vpcmpeqb %ymm1, %ymm0, %ymm1
	vpcmpeqb %ymm2, %ymm0, %ymm2
	vpcmpeqb %ymm3, %ymm0, %ymm3
	vpcmpeqb %ymm4, %ymm0, %ymm4

	vpor	%ymm1, %ymm2, %ymm5
	vpor	%ymm3, %ymm4, %ymm6
	vpor	%ymm5, %ymm6, %ymm5

	vpmovmskb %ymm5, %eax
	testl	%eax, %eax
	jz	L(loop_4x_vec)

	/* There is a match.  */
	vpmovmskb %ymm4, %eax
	testl	%eax, %eax
	jnz	L(last_vec_x3)

	vpmovmskb %ymm3, %eax
	testl	%eax, %eax
	jnz	L(last_vec_x2)

	vpmovmskb %ymm2, %eax
	testl	%eax, %eax
	jnz	L(last_vec_x1)

	vpmovmskb %ymm1, %eax
	bsrl	%eax, %eax
	addq	%rdi, %rax
L(return_vzeroupper):
	ZERO_UPPER_VEC_REGISTERS_RETURN

	.p2align 4
L(last_4x_vec_or_less):
	addl	$(VEC_SIZE * 4), %edx
	cmpl	$(VEC_SIZE * 2), %edx
	jbe	L(last_2x_vec)

	vpcmpeqb (VEC_SIZE * 3)(%rdi), %ymm0, %ymm1
	vpmovmskb %ymm1, %eax
	testl	%eax, %eax
	jnz	L(last_vec_x3)

	vpcmpeqb (VEC_SIZE * 2)(%rdi), %ymm0, %ymm2
	vpmovmskb %ymm2, %eax
	testl	%eax, %eax
	jnz	L(last_vec_x2)

	vpcmpeqb VEC_SIZE(%rdi), %ymm0, %ymm3
	vpmovmskb %ymm3, %eax
	testl	%eax, %eax
	jnz	L(last_vec_x1_check)
	cmpl	$(VEC_SIZE * 3), %edx
	jbe	L(zero)

	vpcmpeqb (%rdi), %ymm0, %ymm4
	vpmovmskb %ymm4, %eax
	testl	%eax, %eax
	jz	L(zero)
	bsrl	%eax, %eax
	subq	$(VEC_SIZE * 4), %rdx
	addq	%rax, %rdx
	jl	L(zero)
	addq	%rdi, %rax
	VZEROUPPER_RETURN

	.p2align 4
L(last_2x_vec):
	vpcmpeqb (VEC_SIZE * 3)(%rdi), %ymm0, %ymm1
	vpmovmskb %ymm1, %eax
	testl	%eax, %eax
	jnz	L(last_vec_x3_check)
	cmpl	$VEC_SIZE, %edx
	jbe	L(zero)

	vpcmpeqb (VEC_SIZE * 2)(%rdi), %ymm0, %ymm1
	vpmovmskb %ymm1, %eax
	testl	%eax, %eax
	jz	L(zero)
	bsrl	%eax, %eax
	subq	$(VEC_SIZE * 2), %rdx
	addq	%rax, %rdx
	jl	L(zero)
	addl	$(VEC_SIZE * 2), %eax
	addq	%rdi, %rax
	VZEROUPPER_RETURN

	.p2align 4
L(last_vec_x0):
	bsrl	%eax, %eax
	addq	%rdi, %rax
	VZEROUPPER_RETURN

	.p2align 4
L(last_vec_x1):
	bsrl	%eax, %eax
	addl	$VEC_SIZE, %eax
	addq	%rdi, %rax
	VZEROUPPER_RETURN

	.p2align 4
L(last_vec_x2):
	bsrl	%eax, %eax
	addl	$(VEC_SIZE * 2), %eax
	addq	%rdi, %rax
	VZEROUPPER_RETURN

	.p2align 4
L(last_vec_x3):
	bsrl	%eax, %eax
	addl	$(VEC_SIZE * 3), %eax
	addq	%rdi, %rax
	ret

	.p2align 4
L(last_vec_x1_check):
	bsrl	%eax, %eax
	subq	$(VEC_SIZE * 3), %rdx
	addq	%rax, %rdx
	jl	L(zero)
	addl	$VEC_SIZE, %eax
	addq	%rdi, %rax
	VZEROUPPER_RETURN

	.p2align 4
L(last_vec_x3_check):
	bsrl	%eax, %eax
	subq	$VEC_SIZE, %rdx
	addq	%rax, %rdx
	jl	L(zero)
	addl	$(VEC_SIZE * 3), %eax
	addq	%rdi, %rax
	VZEROUPPER_RETURN

	.p2align 4
L(zero):
	xorl	%eax, %eax
	VZEROUPPER_RETURN

	.p2align 4
L(null):
	xorl	%eax, %eax
	ret

	.p2align 4
L(last_vec_or_less_aligned):
	movl	%edx, %ecx

	vpcmpeqb (%rdi), %ymm0, %ymm1

	movl	$1, %edx
	/* Support rdx << 32.  */
	salq	%cl, %rdx
	subq	$1, %rdx

	vpmovmskb %ymm1, %eax

	/* Remove the trailing bytes.  */
	andl	%edx, %eax
	testl	%eax, %eax
	jz	L(zero)

	bsrl	%eax, %eax
	addq	%rdi, %rax
	VZEROUPPER_RETURN

	.p2align 4
L(last_vec_or_less):
	addl	$VEC_SIZE, %edx

	/* Check for zero length.  */
	testl	%edx, %edx
	jz	L(null)

	movl	%edi, %ecx
	andl	$(VEC_SIZE - 1), %ecx
	jz	L(last_vec_or_less_aligned)

	movl	%ecx, %esi
	movl	%ecx, %r8d
	addl	%edx, %esi
	andq	$-VEC_SIZE, %rdi

	subl	$VEC_SIZE, %esi
	ja	L(last_vec_2x_aligned)

	/* Check the last VEC.  */
	vpcmpeqb (%rdi), %ymm0, %ymm1
	vpmovmskb %ymm1, %eax

	/* Remove the leading and trailing bytes.  */
	sarl	%cl, %eax
	movl	%edx, %ecx

	movl	$1, %edx
	sall	%cl, %edx
	subl	$1, %edx

	andl	%edx, %eax
	testl	%eax, %eax
	jz	L(zero)

	bsrl	%eax, %eax
	addq	%rdi, %rax
	addq	%r8, %rax
	VZEROUPPER_RETURN

	.p2align 4
L(last_vec_2x_aligned):
	movl	%esi, %ecx

	/* Check the last VEC.  */
	vpcmpeqb VEC_SIZE(%rdi), %ymm0, %ymm1

	movl	$1, %edx
	sall	%cl, %edx
	subl	$1, %edx

	vpmovmskb %ymm1, %eax

	/* Remove the trailing bytes.  */
	andl	%edx, %eax

	testl	%eax, %eax
	jnz	L(last_vec_x1)

	/* Check the second last VEC.  */
	vpcmpeqb (%rdi), %ymm0, %ymm1

	movl	%r8d, %ecx

	vpmovmskb %ymm1, %eax

	/* Remove the leading bytes.  Must use unsigned right shift for
	   bsrl below.  */
	shrl	%cl, %eax
	testl	%eax, %eax
	jz	L(zero)

	bsrl	%eax, %eax
	addq	%rdi, %rax
	addq	%r8, %rax
	VZEROUPPER_RETURN
END (MEMRCHR)
#endif
