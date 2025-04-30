/* strrchr/wcsrchr optimized with AVX2.
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

# ifndef STRRCHR
#  define STRRCHR	__strrchr_avx2
# endif

# ifdef USE_AS_WCSRCHR
#  define VPBROADCAST	vpbroadcastd
#  define VPCMPEQ	vpcmpeqd
# else
#  define VPBROADCAST	vpbroadcastb
#  define VPCMPEQ	vpcmpeqb
# endif

# ifndef VZEROUPPER
#  define VZEROUPPER	vzeroupper
# endif

# ifndef SECTION
#  define SECTION(p)	p##.avx
# endif

# define VEC_SIZE	32

	.section SECTION(.text),"ax",@progbits
ENTRY (STRRCHR)
	movd	%esi, %xmm4
	movl	%edi, %ecx
	/* Broadcast CHAR to YMM4.  */
	VPBROADCAST %xmm4, %ymm4
	vpxor	%xmm0, %xmm0, %xmm0

	/* Check if we may cross page boundary with one vector load.  */
	andl	$(2 * VEC_SIZE - 1), %ecx
	cmpl	$VEC_SIZE, %ecx
	ja	L(cros_page_boundary)

	vmovdqu	(%rdi), %ymm1
	VPCMPEQ	%ymm1, %ymm0, %ymm2
	VPCMPEQ	%ymm1, %ymm4, %ymm3
	vpmovmskb %ymm2, %ecx
	vpmovmskb %ymm3, %eax
	addq	$VEC_SIZE, %rdi

	testl	%eax, %eax
	jnz	L(first_vec)

	testl	%ecx, %ecx
	jnz	L(return_null)

	andq	$-VEC_SIZE, %rdi
	xorl	%edx, %edx
	jmp	L(aligned_loop)

	.p2align 4
L(first_vec):
	/* Check if there is a nul CHAR.  */
	testl	%ecx, %ecx
	jnz	L(char_and_nul_in_first_vec)

	/* Remember the match and keep searching.  */
	movl	%eax, %edx
	movq	%rdi, %rsi
	andq	$-VEC_SIZE, %rdi
	jmp	L(aligned_loop)

	.p2align 4
L(cros_page_boundary):
	andl	$(VEC_SIZE - 1), %ecx
	andq	$-VEC_SIZE, %rdi
	vmovdqa	(%rdi), %ymm1
	VPCMPEQ	%ymm1, %ymm0, %ymm2
	VPCMPEQ	%ymm1, %ymm4, %ymm3
	vpmovmskb %ymm2, %edx
	vpmovmskb %ymm3, %eax
	shrl	%cl, %edx
	shrl	%cl, %eax
	addq	$VEC_SIZE, %rdi

	/* Check if there is a CHAR.  */
	testl	%eax, %eax
	jnz	L(found_char)

	testl	%edx, %edx
	jnz	L(return_null)

	jmp	L(aligned_loop)

	.p2align 4
L(found_char):
	testl	%edx, %edx
	jnz	L(char_and_nul)

	/* Remember the match and keep searching.  */
	movl	%eax, %edx
	leaq	(%rdi, %rcx), %rsi

	.p2align 4
L(aligned_loop):
	vmovdqa	(%rdi), %ymm1
	VPCMPEQ	%ymm1, %ymm0, %ymm2
	addq	$VEC_SIZE, %rdi
	VPCMPEQ	%ymm1, %ymm4, %ymm3
	vpmovmskb %ymm2, %ecx
	vpmovmskb %ymm3, %eax
	orl	%eax, %ecx
	jnz	L(char_nor_null)

	vmovdqa	(%rdi), %ymm1
	VPCMPEQ	%ymm1, %ymm0, %ymm2
	add	$VEC_SIZE, %rdi
	VPCMPEQ	%ymm1, %ymm4, %ymm3
	vpmovmskb %ymm2, %ecx
	vpmovmskb %ymm3, %eax
	orl	%eax, %ecx
	jnz	L(char_nor_null)

	vmovdqa	(%rdi), %ymm1
	VPCMPEQ	%ymm1, %ymm0, %ymm2
	addq	$VEC_SIZE, %rdi
	VPCMPEQ	%ymm1, %ymm4, %ymm3
	vpmovmskb %ymm2, %ecx
	vpmovmskb %ymm3, %eax
	orl	%eax, %ecx
	jnz	L(char_nor_null)

	vmovdqa	(%rdi), %ymm1
	VPCMPEQ	%ymm1, %ymm0, %ymm2
	addq	$VEC_SIZE, %rdi
	VPCMPEQ	%ymm1, %ymm4, %ymm3
	vpmovmskb %ymm2, %ecx
	vpmovmskb %ymm3, %eax
	orl	%eax, %ecx
	jz	L(aligned_loop)

	.p2align 4
L(char_nor_null):
	/* Find a CHAR or a nul CHAR in a loop.  */
	testl	%eax, %eax
	jnz	L(match)
L(return_value):
	testl	%edx, %edx
	jz	L(return_null)
	movl	%edx, %eax
	movq	%rsi, %rdi

# ifdef USE_AS_WCSRCHR
	/* Keep the first bit for each matching CHAR for bsr.  */
	andl	$0x11111111, %eax
# endif
	bsrl	%eax, %eax
	leaq	-VEC_SIZE(%rdi, %rax), %rax
L(return_vzeroupper):
	ZERO_UPPER_VEC_REGISTERS_RETURN

	.p2align 4
L(match):
	/* Find a CHAR.  Check if there is a nul CHAR.  */
	vpmovmskb %ymm2, %ecx
	testl	%ecx, %ecx
	jnz	L(find_nul)

	/* Remember the match and keep searching.  */
	movl	%eax, %edx
	movq	%rdi, %rsi
	jmp	L(aligned_loop)

	.p2align 4
L(find_nul):
# ifdef USE_AS_WCSRCHR
	/* Keep the first bit for each matching CHAR for bsr.  */
	andl	$0x11111111, %ecx
	andl	$0x11111111, %eax
# endif
	/* Mask out any matching bits after the nul CHAR.  */
	movl	%ecx, %r8d
	subl	$1, %r8d
	xorl	%ecx, %r8d
	andl	%r8d, %eax
	testl	%eax, %eax
	/* If there is no CHAR here, return the remembered one.  */
	jz	L(return_value)
	bsrl	%eax, %eax
	leaq	-VEC_SIZE(%rdi, %rax), %rax
	VZEROUPPER_RETURN

	.p2align 4
L(char_and_nul):
	/* Find both a CHAR and a nul CHAR.  */
	addq	%rcx, %rdi
	movl	%edx, %ecx
L(char_and_nul_in_first_vec):
# ifdef USE_AS_WCSRCHR
	/* Keep the first bit for each matching CHAR for bsr.  */
	andl	$0x11111111, %ecx
	andl	$0x11111111, %eax
# endif
	/* Mask out any matching bits after the nul CHAR.  */
	movl	%ecx, %r8d
	subl	$1, %r8d
	xorl	%ecx, %r8d
	andl	%r8d, %eax
	testl	%eax, %eax
	/* Return null pointer if the nul CHAR comes first.  */
	jz	L(return_null)
	bsrl	%eax, %eax
	leaq	-VEC_SIZE(%rdi, %rax), %rax
	VZEROUPPER_RETURN

	.p2align 4
L(return_null):
	xorl	%eax, %eax
	VZEROUPPER_RETURN

END (STRRCHR)
#endif
