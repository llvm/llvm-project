/* strchr/strchrnul optimized with AVX2.
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

# ifndef STRCHR
#  define STRCHR	__strchr_avx2
# endif

# ifdef USE_AS_WCSCHR
#  define VPBROADCAST	vpbroadcastd
#  define VPCMPEQ	vpcmpeqd
#  define VPMINU	vpminud
#  define CHAR_REG	esi
# else
#  define VPBROADCAST	vpbroadcastb
#  define VPCMPEQ	vpcmpeqb
#  define VPMINU	vpminub
#  define CHAR_REG	sil
# endif

# ifndef VZEROUPPER
#  define VZEROUPPER	vzeroupper
# endif

# ifndef SECTION
#  define SECTION(p)	p##.avx
# endif

# define VEC_SIZE 32
# define PAGE_SIZE 4096

	.section SECTION(.text),"ax",@progbits
ENTRY (STRCHR)
	/* Broadcast CHAR to YMM0.	*/
	vmovd	%esi, %xmm0
	movl	%edi, %eax
	andl	$(PAGE_SIZE - 1), %eax
	VPBROADCAST	%xmm0, %ymm0
	vpxor	%xmm9, %xmm9, %xmm9

	/* Check if we cross page boundary with one vector load.  */
	cmpl	$(PAGE_SIZE - VEC_SIZE), %eax
	ja	L(cross_page_boundary)

	/* Check the first VEC_SIZE bytes.	Search for both CHAR and the
	   null byte.  */
	vmovdqu	(%rdi), %ymm8
	VPCMPEQ	%ymm8, %ymm0, %ymm1
	VPCMPEQ	%ymm8, %ymm9, %ymm2
	vpor	%ymm1, %ymm2, %ymm1
	vpmovmskb %ymm1, %eax
	testl	%eax, %eax
	jz	L(aligned_more)
	tzcntl	%eax, %eax
# ifndef USE_AS_STRCHRNUL
	/* Found CHAR or the null byte.	 */
	cmp	(%rdi, %rax), %CHAR_REG
	jne	L(zero)
# endif
	addq	%rdi, %rax
	VZEROUPPER_RETURN

	/* .p2align 5 helps keep performance more consistent if ENTRY()
	   alignment % 32 was either 16 or 0. As well this makes the
	   alignment % 32 of the loop_4x_vec fixed which makes tuning it
	   easier.  */
	.p2align 5
L(first_vec_x4):
	tzcntl	%eax, %eax
	addq	$(VEC_SIZE * 3 + 1), %rdi
# ifndef USE_AS_STRCHRNUL
	/* Found CHAR or the null byte.	 */
	cmp	(%rdi, %rax), %CHAR_REG
	jne	L(zero)
# endif
	addq	%rdi, %rax
	VZEROUPPER_RETURN

# ifndef USE_AS_STRCHRNUL
L(zero):
	xorl	%eax, %eax
	VZEROUPPER_RETURN
# endif


	.p2align 4
L(first_vec_x1):
	tzcntl	%eax, %eax
	incq	%rdi
# ifndef USE_AS_STRCHRNUL
	/* Found CHAR or the null byte.	 */
	cmp	(%rdi, %rax), %CHAR_REG
	jne	L(zero)
# endif
	addq	%rdi, %rax
	VZEROUPPER_RETURN

	.p2align 4
L(first_vec_x2):
	tzcntl	%eax, %eax
	addq	$(VEC_SIZE + 1), %rdi
# ifndef USE_AS_STRCHRNUL
	/* Found CHAR or the null byte.	 */
	cmp	(%rdi, %rax), %CHAR_REG
	jne	L(zero)
# endif
	addq	%rdi, %rax
	VZEROUPPER_RETURN

	.p2align 4
L(first_vec_x3):
	tzcntl	%eax, %eax
	addq	$(VEC_SIZE * 2 + 1), %rdi
# ifndef USE_AS_STRCHRNUL
	/* Found CHAR or the null byte.	 */
	cmp	(%rdi, %rax), %CHAR_REG
	jne	L(zero)
# endif
	addq	%rdi, %rax
	VZEROUPPER_RETURN

	.p2align 4
L(aligned_more):
	/* Align data to VEC_SIZE - 1. This is the same number of
	   instructions as using andq -VEC_SIZE but saves 4 bytes of code
	   on x4 check.  */
	orq	$(VEC_SIZE - 1), %rdi
L(cross_page_continue):
	/* Check the next 4 * VEC_SIZE.  Only one VEC_SIZE at a time
	   since data is only aligned to VEC_SIZE.  */
	vmovdqa	1(%rdi), %ymm8
	VPCMPEQ	%ymm8, %ymm0, %ymm1
	VPCMPEQ	%ymm8, %ymm9, %ymm2
	vpor	%ymm1, %ymm2, %ymm1
	vpmovmskb %ymm1, %eax
	testl	%eax, %eax
	jnz	L(first_vec_x1)

	vmovdqa	(VEC_SIZE + 1)(%rdi), %ymm8
	VPCMPEQ	%ymm8, %ymm0, %ymm1
	VPCMPEQ	%ymm8, %ymm9, %ymm2
	vpor	%ymm1, %ymm2, %ymm1
	vpmovmskb %ymm1, %eax
	testl	%eax, %eax
	jnz	L(first_vec_x2)

	vmovdqa	(VEC_SIZE * 2 + 1)(%rdi), %ymm8
	VPCMPEQ	%ymm8, %ymm0, %ymm1
	VPCMPEQ	%ymm8, %ymm9, %ymm2
	vpor	%ymm1, %ymm2, %ymm1
	vpmovmskb %ymm1, %eax
	testl	%eax, %eax
	jnz	L(first_vec_x3)

	vmovdqa	(VEC_SIZE * 3 + 1)(%rdi), %ymm8
	VPCMPEQ	%ymm8, %ymm0, %ymm1
	VPCMPEQ	%ymm8, %ymm9, %ymm2
	vpor	%ymm1, %ymm2, %ymm1
	vpmovmskb %ymm1, %eax
	testl	%eax, %eax
	jnz	L(first_vec_x4)
	/* Align data to VEC_SIZE * 4 - 1.	*/
	addq	$(VEC_SIZE * 4 + 1), %rdi
	andq	$-(VEC_SIZE * 4), %rdi
	.p2align 4
L(loop_4x_vec):
	/* Compare 4 * VEC at a time forward.  */
	vmovdqa	(%rdi), %ymm5
	vmovdqa	(VEC_SIZE)(%rdi), %ymm6
	vmovdqa	(VEC_SIZE * 2)(%rdi), %ymm7
	vmovdqa	(VEC_SIZE * 3)(%rdi), %ymm8

	/* Leaves only CHARS matching esi as 0.	 */
	vpxor	%ymm5, %ymm0, %ymm1
	vpxor	%ymm6, %ymm0, %ymm2
	vpxor	%ymm7, %ymm0, %ymm3
	vpxor	%ymm8, %ymm0, %ymm4

	VPMINU	%ymm1, %ymm5, %ymm1
	VPMINU	%ymm2, %ymm6, %ymm2
	VPMINU	%ymm3, %ymm7, %ymm3
	VPMINU	%ymm4, %ymm8, %ymm4

	VPMINU	%ymm1, %ymm2, %ymm5
	VPMINU	%ymm3, %ymm4, %ymm6

	VPMINU	%ymm5, %ymm6, %ymm6

	VPCMPEQ	%ymm6, %ymm9, %ymm6
	vpmovmskb %ymm6, %ecx
	subq	$-(VEC_SIZE * 4), %rdi
	testl	%ecx, %ecx
	jz	L(loop_4x_vec)


	VPCMPEQ	%ymm1, %ymm9, %ymm1
	vpmovmskb %ymm1, %eax
	testl	%eax, %eax
	jnz	L(last_vec_x0)


	VPCMPEQ	%ymm5, %ymm9, %ymm2
	vpmovmskb %ymm2, %eax
	testl	%eax, %eax
	jnz	L(last_vec_x1)

	VPCMPEQ	%ymm3, %ymm9, %ymm3
	vpmovmskb %ymm3, %eax
	/* rcx has combined result from all 4 VEC. It will only be used
	   if the first 3 other VEC all did not contain a match.  */
	salq	$32, %rcx
	orq	%rcx, %rax
	tzcntq	%rax, %rax
	subq	$(VEC_SIZE * 2), %rdi
# ifndef USE_AS_STRCHRNUL
	/* Found CHAR or the null byte.	 */
	cmp	(%rdi, %rax), %CHAR_REG
	jne	L(zero_end)
# endif
	addq	%rdi, %rax
	VZEROUPPER_RETURN


	.p2align 4
L(last_vec_x0):
	tzcntl	%eax, %eax
	addq	$-(VEC_SIZE * 4), %rdi
# ifndef USE_AS_STRCHRNUL
	/* Found CHAR or the null byte.	 */
	cmp	(%rdi, %rax), %CHAR_REG
	jne	L(zero_end)
# endif
	addq	%rdi, %rax
	VZEROUPPER_RETURN

# ifndef USE_AS_STRCHRNUL
L(zero_end):
	xorl	%eax, %eax
	VZEROUPPER_RETURN
# endif

	.p2align 4
L(last_vec_x1):
	tzcntl	%eax, %eax
	subq	$(VEC_SIZE * 3), %rdi
# ifndef USE_AS_STRCHRNUL
	/* Found CHAR or the null byte.	 */
	cmp	(%rdi, %rax), %CHAR_REG
	jne	L(zero_end)
# endif
	addq	%rdi, %rax
	VZEROUPPER_RETURN


	/* Cold case for crossing page with first load.	 */
	.p2align 4
L(cross_page_boundary):
	movq	%rdi, %rdx
	/* Align rdi to VEC_SIZE - 1.  */
	orq	$(VEC_SIZE - 1), %rdi
	vmovdqa	-(VEC_SIZE - 1)(%rdi), %ymm8
	VPCMPEQ	%ymm8, %ymm0, %ymm1
	VPCMPEQ	%ymm8, %ymm9, %ymm2
	vpor	%ymm1, %ymm2, %ymm1
	vpmovmskb %ymm1, %eax
	/* Remove the leading bytes. sarxl only uses bits [5:0] of COUNT
	   so no need to manually mod edx.  */
	sarxl	%edx, %eax, %eax
	testl	%eax, %eax
	jz	L(cross_page_continue)
	tzcntl	%eax, %eax
# ifndef USE_AS_STRCHRNUL
	xorl	%ecx, %ecx
	/* Found CHAR or the null byte.	 */
	cmp	(%rdx, %rax), %CHAR_REG
	leaq	(%rdx, %rax), %rax
	cmovne	%rcx, %rax
# else
	addq	%rdx, %rax
# endif
L(return_vzeroupper):
	ZERO_UPPER_VEC_REGISTERS_RETURN

END (STRCHR)
# endif
