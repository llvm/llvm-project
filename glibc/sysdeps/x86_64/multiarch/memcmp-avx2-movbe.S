/* memcmp/wmemcmp optimized with AVX2.
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

/* memcmp/wmemcmp is implemented as:
   1. Use ymm vector compares when possible. The only case where
      vector compares is not possible for when size < VEC_SIZE
      and loading from either s1 or s2 would cause a page cross.
   2. For size from 2 to 7 bytes on page cross, load as big endian
      with movbe and bswap to avoid branches.
   3. Use xmm vector compare when size >= 4 bytes for memcmp or
      size >= 8 bytes for wmemcmp.
   4. Optimistically compare up to first 4 * VEC_SIZE one at a
      to check for early mismatches. Only do this if its guranteed the
      work is not wasted.
   5. If size is 8 * VEC_SIZE or less, unroll the loop.
   6. Compare 4 * VEC_SIZE at a time with the aligned first memory
      area.
   7. Use 2 vector compares when size is 2 * VEC_SIZE or less.
   8. Use 4 vector compares when size is 4 * VEC_SIZE or less.
   9. Use 8 vector compares when size is 8 * VEC_SIZE or less.  */


# include <sysdep.h>

# ifndef MEMCMP
#  define MEMCMP	__memcmp_avx2_movbe
# endif

# ifdef USE_AS_WMEMCMP
#  define CHAR_SIZE	4
#  define VPCMPEQ	vpcmpeqd
# else
#  define CHAR_SIZE	1
#  define VPCMPEQ	vpcmpeqb
# endif

# ifndef VZEROUPPER
#  define VZEROUPPER	vzeroupper
# endif

# ifndef SECTION
#  define SECTION(p)	p##.avx
# endif

# define VEC_SIZE 32
# define PAGE_SIZE	4096

/* Warning!
           wmemcmp has to use SIGNED comparison for elements.
           memcmp has to use UNSIGNED comparison for elemnts.
*/

	.section SECTION(.text),"ax",@progbits
ENTRY (MEMCMP)
# ifdef USE_AS_WMEMCMP
	shl	$2, %RDX_LP
# elif defined __ILP32__
	/* Clear the upper 32 bits.  */
	movl	%edx, %edx
# endif
	cmp	$VEC_SIZE, %RDX_LP
	jb	L(less_vec)

	/* From VEC to 2 * VEC.  No branch when size == VEC_SIZE.  */
	vmovdqu	(%rsi), %ymm1
	VPCMPEQ	(%rdi), %ymm1, %ymm1
	vpmovmskb %ymm1, %eax
	/* NB: eax must be destination register if going to
	   L(return_vec_[0,2]). For L(return_vec_3 destination register
	   must be ecx.  */
	incl	%eax
	jnz	L(return_vec_0)

	cmpq	$(VEC_SIZE * 2), %rdx
	jbe	L(last_1x_vec)

	/* Check second VEC no matter what.  */
	vmovdqu	VEC_SIZE(%rsi), %ymm2
	VPCMPEQ	VEC_SIZE(%rdi), %ymm2, %ymm2
	vpmovmskb %ymm2, %eax
	/* If all 4 VEC where equal eax will be all 1s so incl will
	   overflow and set zero flag.  */
	incl	%eax
	jnz	L(return_vec_1)

	/* Less than 4 * VEC.  */
	cmpq	$(VEC_SIZE * 4), %rdx
	jbe	L(last_2x_vec)

	/* Check third and fourth VEC no matter what.  */
	vmovdqu	(VEC_SIZE * 2)(%rsi), %ymm3
	VPCMPEQ	(VEC_SIZE * 2)(%rdi), %ymm3, %ymm3
	vpmovmskb %ymm3, %eax
	incl	%eax
	jnz	L(return_vec_2)
	vmovdqu	(VEC_SIZE * 3)(%rsi), %ymm4
	VPCMPEQ	(VEC_SIZE * 3)(%rdi), %ymm4, %ymm4
	vpmovmskb %ymm4, %ecx
	incl	%ecx
	jnz	L(return_vec_3)

	/* Go to 4x VEC loop.  */
	cmpq	$(VEC_SIZE * 8), %rdx
	ja	L(more_8x_vec)

	/* Handle remainder of size = 4 * VEC + 1 to 8 * VEC without any
	   branches.  */

	/* Load first two VEC from s2 before adjusting addresses.  */
	vmovdqu	-(VEC_SIZE * 4)(%rsi, %rdx), %ymm1
	vmovdqu	-(VEC_SIZE * 3)(%rsi, %rdx), %ymm2
	leaq	-(4 * VEC_SIZE)(%rdi, %rdx), %rdi
	leaq	-(4 * VEC_SIZE)(%rsi, %rdx), %rsi

	/* Wait to load from s1 until addressed adjust due to
	   unlamination of microfusion with complex address mode.  */
	VPCMPEQ	(%rdi), %ymm1, %ymm1
	VPCMPEQ	(VEC_SIZE)(%rdi), %ymm2, %ymm2

	vmovdqu	(VEC_SIZE * 2)(%rsi), %ymm3
	VPCMPEQ	(VEC_SIZE * 2)(%rdi), %ymm3, %ymm3
	vmovdqu	(VEC_SIZE * 3)(%rsi), %ymm4
	VPCMPEQ	(VEC_SIZE * 3)(%rdi), %ymm4, %ymm4

	/* Reduce VEC0 - VEC4.  */
	vpand	%ymm1, %ymm2, %ymm5
	vpand	%ymm3, %ymm4, %ymm6
	vpand	%ymm5, %ymm6, %ymm7
	vpmovmskb %ymm7, %ecx
	incl	%ecx
	jnz	L(return_vec_0_1_2_3)
	/* NB: eax must be zero to reach here.  */
	VZEROUPPER_RETURN

	.p2align 4
L(return_vec_0):
	tzcntl	%eax, %eax
# ifdef USE_AS_WMEMCMP
	movl	(%rdi, %rax), %ecx
	xorl	%edx, %edx
	cmpl	(%rsi, %rax), %ecx
	/* NB: no partial register stall here because xorl zero idiom
	   above.  */
	setg	%dl
	leal	-1(%rdx, %rdx), %eax
# else
	movzbl	(%rsi, %rax), %ecx
	movzbl	(%rdi, %rax), %eax
	subl	%ecx, %eax
# endif
L(return_vzeroupper):
	ZERO_UPPER_VEC_REGISTERS_RETURN

	.p2align 4
L(return_vec_1):
	tzcntl	%eax, %eax
# ifdef USE_AS_WMEMCMP
	movl	VEC_SIZE(%rdi, %rax), %ecx
	xorl	%edx, %edx
	cmpl	VEC_SIZE(%rsi, %rax), %ecx
	setg	%dl
	leal	-1(%rdx, %rdx), %eax
# else
	movzbl	VEC_SIZE(%rsi, %rax), %ecx
	movzbl	VEC_SIZE(%rdi, %rax), %eax
	subl	%ecx, %eax
# endif
	VZEROUPPER_RETURN

	.p2align 4
L(return_vec_2):
	tzcntl	%eax, %eax
# ifdef USE_AS_WMEMCMP
	movl	(VEC_SIZE * 2)(%rdi, %rax), %ecx
	xorl	%edx, %edx
	cmpl	(VEC_SIZE * 2)(%rsi, %rax), %ecx
	setg	%dl
	leal	-1(%rdx, %rdx), %eax
# else
	movzbl	(VEC_SIZE * 2)(%rsi, %rax), %ecx
	movzbl	(VEC_SIZE * 2)(%rdi, %rax), %eax
	subl	%ecx, %eax
# endif
	VZEROUPPER_RETURN

	/* NB: p2align 5 here to ensure 4x loop is 32 byte aligned.  */
	.p2align 5
L(8x_return_vec_0_1_2_3):
	/* Returning from L(more_8x_vec) requires restoring rsi.  */
	addq	%rdi, %rsi
L(return_vec_0_1_2_3):
	vpmovmskb %ymm1, %eax
	incl	%eax
	jnz	L(return_vec_0)

	vpmovmskb %ymm2, %eax
	incl	%eax
	jnz	L(return_vec_1)

	vpmovmskb %ymm3, %eax
	incl	%eax
	jnz	L(return_vec_2)
L(return_vec_3):
	tzcntl	%ecx, %ecx
# ifdef USE_AS_WMEMCMP
	movl	(VEC_SIZE * 3)(%rdi, %rcx), %eax
	xorl	%edx, %edx
	cmpl	(VEC_SIZE * 3)(%rsi, %rcx), %eax
	setg	%dl
	leal	-1(%rdx, %rdx), %eax
# else
	movzbl	(VEC_SIZE * 3)(%rdi, %rcx), %eax
	movzbl	(VEC_SIZE * 3)(%rsi, %rcx), %ecx
	subl	%ecx, %eax
# endif
	VZEROUPPER_RETURN

	.p2align 4
L(more_8x_vec):
	/* Set end of s1 in rdx.  */
	leaq	-(VEC_SIZE * 4)(%rdi, %rdx), %rdx
	/* rsi stores s2 - s1. This allows loop to only update one
	   pointer.  */
	subq	%rdi, %rsi
	/* Align s1 pointer.  */
	andq	$-VEC_SIZE, %rdi
	/* Adjust because first 4x vec where check already.  */
	subq	$-(VEC_SIZE * 4), %rdi
	.p2align 4
L(loop_4x_vec):
	/* rsi has s2 - s1 so get correct address by adding s1 (in rdi).
	 */
	vmovdqu	(%rsi, %rdi), %ymm1
	VPCMPEQ	(%rdi), %ymm1, %ymm1

	vmovdqu	VEC_SIZE(%rsi, %rdi), %ymm2
	VPCMPEQ	VEC_SIZE(%rdi), %ymm2, %ymm2

	vmovdqu	(VEC_SIZE * 2)(%rsi, %rdi), %ymm3
	VPCMPEQ	(VEC_SIZE * 2)(%rdi), %ymm3, %ymm3

	vmovdqu	(VEC_SIZE * 3)(%rsi, %rdi), %ymm4
	VPCMPEQ	(VEC_SIZE * 3)(%rdi), %ymm4, %ymm4

	vpand	%ymm1, %ymm2, %ymm5
	vpand	%ymm3, %ymm4, %ymm6
	vpand	%ymm5, %ymm6, %ymm7
	vpmovmskb %ymm7, %ecx
	incl	%ecx
	jnz	L(8x_return_vec_0_1_2_3)
	subq	$-(VEC_SIZE * 4), %rdi
	/* Check if s1 pointer at end.  */
	cmpq	%rdx, %rdi
	jb	L(loop_4x_vec)

	subq	%rdx, %rdi
	/* rdi has 4 * VEC_SIZE - remaining length.  */
	cmpl	$(VEC_SIZE * 3), %edi
	jae	L(8x_last_1x_vec)
	/* Load regardless of branch.  */
	vmovdqu	(VEC_SIZE * 2)(%rsi, %rdx), %ymm3
	cmpl	$(VEC_SIZE * 2), %edi
	jae	L(8x_last_2x_vec)

	/* Check last 4 VEC.  */
	vmovdqu	(%rsi, %rdx), %ymm1
	VPCMPEQ	(%rdx), %ymm1, %ymm1

	vmovdqu	VEC_SIZE(%rsi, %rdx), %ymm2
	VPCMPEQ	VEC_SIZE(%rdx), %ymm2, %ymm2

	VPCMPEQ	(VEC_SIZE * 2)(%rdx), %ymm3, %ymm3

	vmovdqu	(VEC_SIZE * 3)(%rsi, %rdx), %ymm4
	VPCMPEQ	(VEC_SIZE * 3)(%rdx), %ymm4, %ymm4

	vpand	%ymm1, %ymm2, %ymm5
	vpand	%ymm3, %ymm4, %ymm6
	vpand	%ymm5, %ymm6, %ymm7
	vpmovmskb %ymm7, %ecx
	/* Restore s1 pointer to rdi.  */
	movq	%rdx, %rdi
	incl	%ecx
	jnz	L(8x_return_vec_0_1_2_3)
	/* NB: eax must be zero to reach here.  */
	VZEROUPPER_RETURN

	/* Only entry is from L(more_8x_vec).  */
	.p2align 4
L(8x_last_2x_vec):
	/* Check second to last VEC. rdx store end pointer of s1 and
	   ymm3 has already been loaded with second to last VEC from s2.
	 */
	VPCMPEQ	(VEC_SIZE * 2)(%rdx), %ymm3, %ymm3
	vpmovmskb %ymm3, %eax
	incl	%eax
	jnz	L(8x_return_vec_2)
	/* Check last VEC.  */
	.p2align 4
L(8x_last_1x_vec):
	vmovdqu	(VEC_SIZE * 3)(%rsi, %rdx), %ymm4
	VPCMPEQ	(VEC_SIZE * 3)(%rdx), %ymm4, %ymm4
	vpmovmskb %ymm4, %eax
	incl	%eax
	jnz	L(8x_return_vec_3)
	VZEROUPPER_RETURN

	.p2align 4
L(last_2x_vec):
	/* Check second to last VEC.  */
	vmovdqu	-(VEC_SIZE * 2)(%rsi, %rdx), %ymm1
	VPCMPEQ	-(VEC_SIZE * 2)(%rdi, %rdx), %ymm1, %ymm1
	vpmovmskb %ymm1, %eax
	incl	%eax
	jnz	L(return_vec_1_end)
	/* Check last VEC.  */
L(last_1x_vec):
	vmovdqu	-(VEC_SIZE * 1)(%rsi, %rdx), %ymm1
	VPCMPEQ	-(VEC_SIZE * 1)(%rdi, %rdx), %ymm1, %ymm1
	vpmovmskb %ymm1, %eax
	incl	%eax
	jnz	L(return_vec_0_end)
	VZEROUPPER_RETURN

	.p2align 4
L(8x_return_vec_2):
	subq	$VEC_SIZE, %rdx
L(8x_return_vec_3):
	tzcntl	%eax, %eax
	addq	%rdx, %rax
# ifdef USE_AS_WMEMCMP
	movl	(VEC_SIZE * 3)(%rax), %ecx
	xorl	%edx, %edx
	cmpl	(VEC_SIZE * 3)(%rsi, %rax), %ecx
	setg	%dl
	leal	-1(%rdx, %rdx), %eax
# else
	movzbl	(VEC_SIZE * 3)(%rsi, %rax), %ecx
	movzbl	(VEC_SIZE * 3)(%rax), %eax
	subl	%ecx, %eax
# endif
	VZEROUPPER_RETURN

	.p2align 4
L(return_vec_1_end):
	tzcntl	%eax, %eax
	addl	%edx, %eax
# ifdef USE_AS_WMEMCMP
	movl	-(VEC_SIZE * 2)(%rdi, %rax), %ecx
	xorl	%edx, %edx
	cmpl	-(VEC_SIZE * 2)(%rsi, %rax), %ecx
	setg	%dl
	leal	-1(%rdx, %rdx), %eax
# else
	movzbl	-(VEC_SIZE * 2)(%rsi, %rax), %ecx
	movzbl	-(VEC_SIZE * 2)(%rdi, %rax), %eax
	subl	%ecx, %eax
# endif
	VZEROUPPER_RETURN

	.p2align 4
L(return_vec_0_end):
	tzcntl	%eax, %eax
	addl	%edx, %eax
# ifdef USE_AS_WMEMCMP
	movl	-VEC_SIZE(%rdi, %rax), %ecx
	xorl	%edx, %edx
	cmpl	-VEC_SIZE(%rsi, %rax), %ecx
	setg	%dl
	leal	-1(%rdx, %rdx), %eax
# else
	movzbl	-VEC_SIZE(%rsi, %rax), %ecx
	movzbl	-VEC_SIZE(%rdi, %rax), %eax
	subl	%ecx, %eax
# endif
	VZEROUPPER_RETURN

	.p2align 4
L(less_vec):
	/* Check if one or less CHAR. This is necessary for size = 0 but
	   is also faster for size = CHAR_SIZE.  */
	cmpl	$CHAR_SIZE, %edx
	jbe	L(one_or_less)

	/* Check if loading one VEC from either s1 or s2 could cause a
	   page cross. This can have false positives but is by far the
	   fastest method.  */
	movl	%edi, %eax
	orl	%esi, %eax
	andl	$(PAGE_SIZE - 1), %eax
	cmpl	$(PAGE_SIZE - VEC_SIZE), %eax
	jg	L(page_cross_less_vec)

	/* No page cross possible.  */
	vmovdqu	(%rsi), %ymm2
	VPCMPEQ	(%rdi), %ymm2, %ymm2
	vpmovmskb %ymm2, %eax
	incl	%eax
	/* Result will be zero if s1 and s2 match. Otherwise first set
	   bit will be first mismatch.  */
	bzhil	%edx, %eax, %edx
	jnz	L(return_vec_0)
	xorl	%eax, %eax
	VZEROUPPER_RETURN

	.p2align 4
L(page_cross_less_vec):
	/* if USE_AS_WMEMCMP it can only be 0, 4, 8, 12, 16, 20, 24, 28
	   bytes.  */
	cmpl	$16, %edx
	jae	L(between_16_31)
# ifndef USE_AS_WMEMCMP
	cmpl	$8, %edx
	jae	L(between_8_15)
	cmpl	$4, %edx
	jae	L(between_4_7)

	/* Load as big endian to avoid branches.  */
	movzwl	(%rdi), %eax
	movzwl	(%rsi), %ecx
	shll	$8, %eax
	shll	$8, %ecx
	bswap	%eax
	bswap	%ecx
	movzbl	-1(%rdi, %rdx), %edi
	movzbl	-1(%rsi, %rdx), %esi
	orl	%edi, %eax
	orl	%esi, %ecx
	/* Subtraction is okay because the upper 8 bits are zero.  */
	subl	%ecx, %eax
	/* No ymm register was touched.  */
	ret

	.p2align 4
L(one_or_less):
	jb	L(zero)
	movzbl	(%rsi), %ecx
	movzbl	(%rdi), %eax
	subl	%ecx, %eax
	/* No ymm register was touched.  */
	ret

	.p2align 4
L(between_8_15):
# endif
	/* If USE_AS_WMEMCMP fall through into 8-15 byte case.  */
	vmovq	(%rdi), %xmm1
	vmovq	(%rsi), %xmm2
	VPCMPEQ	%xmm1, %xmm2, %xmm2
	vpmovmskb %xmm2, %eax
	subl	$0xffff, %eax
	jnz	L(return_vec_0)
	/* Use overlapping loads to avoid branches.  */
	leaq	-8(%rdi, %rdx), %rdi
	leaq	-8(%rsi, %rdx), %rsi
	vmovq	(%rdi), %xmm1
	vmovq	(%rsi), %xmm2
	VPCMPEQ	%xmm1, %xmm2, %xmm2
	vpmovmskb %xmm2, %eax
	subl	$0xffff, %eax
	jnz	L(return_vec_0)
	/* No ymm register was touched.  */
	ret

	.p2align 4
L(zero):
	xorl	%eax, %eax
	ret

	.p2align 4
L(between_16_31):
	/* From 16 to 31 bytes.  No branch when size == 16.  */
	vmovdqu	(%rsi), %xmm2
	VPCMPEQ	(%rdi), %xmm2, %xmm2
	vpmovmskb %xmm2, %eax
	subl	$0xffff, %eax
	jnz	L(return_vec_0)

	/* Use overlapping loads to avoid branches.  */

	vmovdqu	-16(%rsi, %rdx), %xmm2
	leaq	-16(%rdi, %rdx), %rdi
	leaq	-16(%rsi, %rdx), %rsi
	VPCMPEQ	(%rdi), %xmm2, %xmm2
	vpmovmskb %xmm2, %eax
	subl	$0xffff, %eax
	jnz	L(return_vec_0)
	/* No ymm register was touched.  */
	ret

# ifdef USE_AS_WMEMCMP
	.p2align 4
L(one_or_less):
	jb	L(zero)
	movl	(%rdi), %ecx
	xorl	%edx, %edx
	cmpl	(%rsi), %ecx
	je	L(zero)
	setg	%dl
	leal	-1(%rdx, %rdx), %eax
	/* No ymm register was touched.  */
	ret
# else

	.p2align 4
L(between_4_7):
	/* Load as big endian with overlapping movbe to avoid branches.
	 */
	movbe	(%rdi), %eax
	movbe	(%rsi), %ecx
	shlq	$32, %rax
	shlq	$32, %rcx
	movbe	-4(%rdi, %rdx), %edi
	movbe	-4(%rsi, %rdx), %esi
	orq	%rdi, %rax
	orq	%rsi, %rcx
	subq	%rcx, %rax
	jz	L(zero_4_7)
	sbbl	%eax, %eax
	orl	$1, %eax
L(zero_4_7):
	/* No ymm register was touched.  */
	ret
# endif

END (MEMCMP)
#endif
