/* strcat with AVX2
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
#  define STRCAT  __strcat_avx2
# endif

# define USE_AS_STRCAT

/* Number of bytes in a vector register */
# define VEC_SIZE	32

# ifndef SECTION
#  define SECTION(p)	p##.avx
# endif

	.section SECTION(.text),"ax",@progbits
ENTRY (STRCAT)
	mov	%rdi, %r9
# ifdef USE_AS_STRNCAT
	mov	%rdx, %r8
# endif

	xor	%eax, %eax
	mov	%edi, %ecx
	and	$((VEC_SIZE * 4) - 1), %ecx
	vpxor	%xmm6, %xmm6, %xmm6
	cmp	$(VEC_SIZE * 3), %ecx
	ja	L(fourth_vector_boundary)
	vpcmpeqb (%rdi), %ymm6, %ymm0
	vpmovmskb %ymm0, %edx
	test	%edx, %edx
	jnz	L(exit_null_on_first_vector)
	mov	%rdi, %rax
	and	$-VEC_SIZE, %rax
	jmp	L(align_vec_size_start)
L(fourth_vector_boundary):
	mov	%rdi, %rax
	and	$-VEC_SIZE, %rax
	vpcmpeqb	(%rax), %ymm6, %ymm0
	mov	$-1, %r10d
	sub	%rax, %rcx
	shl	%cl, %r10d
	vpmovmskb %ymm0, %edx
	and	%r10d, %edx
	jnz	L(exit)

L(align_vec_size_start):
	vpcmpeqb VEC_SIZE(%rax), %ymm6, %ymm0
	vpmovmskb %ymm0, %edx
	test	%edx, %edx
	jnz	L(exit_null_on_second_vector)

	vpcmpeqb (VEC_SIZE * 2)(%rax), %ymm6, %ymm1
	vpmovmskb %ymm1, %edx
	test	%edx, %edx
	jnz	L(exit_null_on_third_vector)

	vpcmpeqb (VEC_SIZE * 3)(%rax), %ymm6, %ymm2
	vpmovmskb %ymm2, %edx
	test	%edx, %edx
	jnz	L(exit_null_on_fourth_vector)

	vpcmpeqb (VEC_SIZE * 4)(%rax), %ymm6, %ymm3
	vpmovmskb %ymm3, %edx
	test	%edx, %edx
	jnz	L(exit_null_on_fifth_vector)

	vpcmpeqb (VEC_SIZE * 5)(%rax), %ymm6, %ymm0
	add	$(VEC_SIZE * 4), %rax
	vpmovmskb %ymm0, %edx
	test	%edx, %edx
	jnz	L(exit_null_on_second_vector)

	vpcmpeqb (VEC_SIZE * 2)(%rax), %ymm6, %ymm1
	vpmovmskb %ymm1, %edx
	test	%edx, %edx
	jnz	L(exit_null_on_third_vector)

	vpcmpeqb (VEC_SIZE * 3)(%rax), %ymm6, %ymm2
	vpmovmskb %ymm2, %edx
	test	%edx, %edx
	jnz	L(exit_null_on_fourth_vector)

	vpcmpeqb (VEC_SIZE * 4)(%rax), %ymm6, %ymm3
	vpmovmskb %ymm3, %edx
	test	%edx, %edx
	jnz	L(exit_null_on_fifth_vector)

	vpcmpeqb (VEC_SIZE * 5)(%rax), %ymm6, %ymm0
	add	$(VEC_SIZE * 4), %rax
	vpmovmskb %ymm0, %edx
	test	%edx, %edx
	jnz	L(exit_null_on_second_vector)

	vpcmpeqb (VEC_SIZE * 2)(%rax), %ymm6, %ymm1
	vpmovmskb %ymm1, %edx
	test	%edx, %edx
	jnz	L(exit_null_on_third_vector)

	vpcmpeqb (VEC_SIZE * 3)(%rax), %ymm6, %ymm2
	vpmovmskb %ymm2, %edx
	test	%edx, %edx
	jnz	L(exit_null_on_fourth_vector)

	vpcmpeqb (VEC_SIZE * 4)(%rax), %ymm6, %ymm3
	vpmovmskb %ymm3, %edx
	test	%edx, %edx
	jnz	L(exit_null_on_fifth_vector)

	vpcmpeqb (VEC_SIZE * 5)(%rax), %ymm6, %ymm0
	add	$(VEC_SIZE * 4), %rax
	vpmovmskb %ymm0, %edx
	test	%edx, %edx
	jnz	L(exit_null_on_second_vector)

	vpcmpeqb (VEC_SIZE * 2)(%rax), %ymm6, %ymm1
	vpmovmskb %ymm1, %edx
	test	%edx, %edx
	jnz	L(exit_null_on_third_vector)

	vpcmpeqb (VEC_SIZE * 3)(%rax), %ymm6, %ymm2
	vpmovmskb %ymm2, %edx
	test	%edx, %edx
	jnz	L(exit_null_on_fourth_vector)

	vpcmpeqb (VEC_SIZE * 4)(%rax), %ymm6, %ymm3
	vpmovmskb %ymm3, %edx
	test	%edx, %edx
	jnz	L(exit_null_on_fifth_vector)

	test	$((VEC_SIZE * 4) - 1), %rax
	jz	L(align_four_vec_loop)

	vpcmpeqb (VEC_SIZE * 5)(%rax), %ymm6, %ymm0
	add	$(VEC_SIZE * 5), %rax
	vpmovmskb %ymm0, %edx
	test	%edx, %edx
	jnz	L(exit)

	test	$((VEC_SIZE * 4) - 1), %rax
	jz	L(align_four_vec_loop)

	vpcmpeqb VEC_SIZE(%rax), %ymm6, %ymm1
	add	$VEC_SIZE, %rax
	vpmovmskb %ymm1, %edx
	test	%edx, %edx
	jnz	L(exit)

	test	$((VEC_SIZE * 4) - 1), %rax
	jz	L(align_four_vec_loop)

	vpcmpeqb VEC_SIZE(%rax), %ymm6, %ymm2
	add	$VEC_SIZE, %rax
	vpmovmskb %ymm2, %edx
	test	%edx, %edx
	jnz	L(exit)

	test	$((VEC_SIZE * 4) - 1), %rax
	jz	L(align_four_vec_loop)

	vpcmpeqb VEC_SIZE(%rax), %ymm6, %ymm3
	add	$VEC_SIZE, %rax
	vpmovmskb %ymm3, %edx
	test	%edx, %edx
	jnz	L(exit)

	add	$VEC_SIZE, %rax

	.p2align 4
L(align_four_vec_loop):
	vmovaps	(%rax),	%ymm4
	vpminub	VEC_SIZE(%rax),	%ymm4, %ymm4
	vmovaps	(VEC_SIZE * 2)(%rax),	%ymm5
	vpminub	(VEC_SIZE * 3)(%rax),	%ymm5, %ymm5
	add	$(VEC_SIZE * 4),	%rax
	vpminub	%ymm4,	%ymm5, %ymm5
	vpcmpeqb %ymm5,	%ymm6, %ymm5
	vpmovmskb %ymm5,	%edx
	test	%edx,	%edx
	jz	L(align_four_vec_loop)

	vpcmpeqb -(VEC_SIZE * 4)(%rax), %ymm6, %ymm0
	sub	$(VEC_SIZE * 5),	%rax
	vpmovmskb %ymm0, %edx
	test	%edx, %edx
	jnz	L(exit_null_on_second_vector)

	vpcmpeqb (VEC_SIZE * 2)(%rax), %ymm6, %ymm1
	vpmovmskb %ymm1, %edx
	test	%edx, %edx
	jnz	L(exit_null_on_third_vector)

	vpcmpeqb (VEC_SIZE * 3)(%rax), %ymm6, %ymm2
	vpmovmskb %ymm2, %edx
	test	%edx, %edx
	jnz	L(exit_null_on_fourth_vector)

	vpcmpeqb (VEC_SIZE * 4)(%rax), %ymm6, %ymm3
	vpmovmskb %ymm3, %edx
	sub	%rdi, %rax
	bsf	%rdx, %rdx
	add	%rdx, %rax
	add	$(VEC_SIZE * 4), %rax
	jmp	L(StartStrcpyPart)

	.p2align 4
L(exit):
	sub	%rdi, %rax
L(exit_null_on_first_vector):
	bsf	%rdx, %rdx
	add	%rdx, %rax
	jmp	L(StartStrcpyPart)

	.p2align 4
L(exit_null_on_second_vector):
	sub	%rdi, %rax
	bsf	%rdx, %rdx
	add	%rdx, %rax
	add	$VEC_SIZE, %rax
	jmp	L(StartStrcpyPart)

	.p2align 4
L(exit_null_on_third_vector):
	sub	%rdi, %rax
	bsf	%rdx, %rdx
	add	%rdx, %rax
	add	$(VEC_SIZE * 2), %rax
	jmp	L(StartStrcpyPart)

	.p2align 4
L(exit_null_on_fourth_vector):
	sub	%rdi, %rax
	bsf	%rdx, %rdx
	add	%rdx, %rax
	add	$(VEC_SIZE * 3), %rax
	jmp	L(StartStrcpyPart)

	.p2align 4
L(exit_null_on_fifth_vector):
	sub	%rdi, %rax
	bsf	%rdx, %rdx
	add	%rdx, %rax
	add	$(VEC_SIZE * 4), %rax

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

# include "strcpy-avx2.S"
#endif
