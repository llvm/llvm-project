/* memcmp/wmemcmp optimized with 256-bit EVEX instructions.
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

/* memcmp/wmemcmp is implemented as:
   1. Use ymm vector compares when possible. The only case where
      vector compares is not possible for when size < CHAR_PER_VEC
      and loading from either s1 or s2 would cause a page cross.
   2. For size from 2 to 7 bytes on page cross, load as big endian
      with movbe and bswap to avoid branches.
   3. Use xmm vector compare when size >= 4 bytes for memcmp or
      size >= 8 bytes for wmemcmp.
   4. Optimistically compare up to first 4 * CHAR_PER_VEC one at a
      to check for early mismatches. Only do this if its guranteed the
      work is not wasted.
   5. If size is 8 * VEC_SIZE or less, unroll the loop.
   6. Compare 4 * VEC_SIZE at a time with the aligned first memory
      area.
   7. Use 2 vector compares when size is 2 * CHAR_PER_VEC or less.
   8. Use 4 vector compares when size is 4 * CHAR_PER_VEC or less.
   9. Use 8 vector compares when size is 8 * CHAR_PER_VEC or less.  */

# include <sysdep.h>

# ifndef MEMCMP
#  define MEMCMP	__memcmp_evex_movbe
# endif

# define VMOVU		vmovdqu64

# ifdef USE_AS_WMEMCMP
#  define CHAR_SIZE	4
#  define VPCMP	vpcmpd
# else
#  define CHAR_SIZE	1
#  define VPCMP	vpcmpub
# endif

# define VEC_SIZE	32
# define PAGE_SIZE	4096
# define CHAR_PER_VEC	(VEC_SIZE / CHAR_SIZE)

# define XMM0		xmm16
# define XMM1		xmm17
# define XMM2		xmm18
# define YMM0		ymm16
# define XMM1		xmm17
# define XMM2		xmm18
# define YMM1		ymm17
# define YMM2		ymm18
# define YMM3		ymm19
# define YMM4		ymm20
# define YMM5		ymm21
# define YMM6		ymm22

/* Warning!
           wmemcmp has to use SIGNED comparison for elements.
           memcmp has to use UNSIGNED comparison for elemnts.
*/

	.section .text.evex,"ax",@progbits
ENTRY (MEMCMP)
# ifdef __ILP32__
	/* Clear the upper 32 bits.  */
	movl	%edx, %edx
# endif
	cmp	$CHAR_PER_VEC, %RDX_LP
	jb	L(less_vec)

	/* From VEC to 2 * VEC.  No branch when size == VEC_SIZE.  */
	VMOVU	(%rsi), %YMM1
	/* Use compare not equals to directly check for mismatch.  */
	VPCMP	$4, (%rdi), %YMM1, %k1
	kmovd	%k1, %eax
	/* NB: eax must be destination register if going to
	   L(return_vec_[0,2]). For L(return_vec_3 destination register
	   must be ecx.  */
	testl	%eax, %eax
	jnz	L(return_vec_0)

	cmpq	$(CHAR_PER_VEC * 2), %rdx
	jbe	L(last_1x_vec)

	/* Check second VEC no matter what.  */
	VMOVU	VEC_SIZE(%rsi), %YMM2
	VPCMP	$4, VEC_SIZE(%rdi), %YMM2, %k1
	kmovd	%k1, %eax
	testl	%eax, %eax
	jnz	L(return_vec_1)

	/* Less than 4 * VEC.  */
	cmpq	$(CHAR_PER_VEC * 4), %rdx
	jbe	L(last_2x_vec)

	/* Check third and fourth VEC no matter what.  */
	VMOVU	(VEC_SIZE * 2)(%rsi), %YMM3
	VPCMP	$4, (VEC_SIZE * 2)(%rdi), %YMM3, %k1
	kmovd	%k1, %eax
	testl	%eax, %eax
	jnz	L(return_vec_2)

	VMOVU	(VEC_SIZE * 3)(%rsi), %YMM4
	VPCMP	$4, (VEC_SIZE * 3)(%rdi), %YMM4, %k1
	kmovd	%k1, %ecx
	testl	%ecx, %ecx
	jnz	L(return_vec_3)

	/* Zero YMM0. 4x VEC reduction is done with vpxor + vtern so
	   compare with zero to get a mask is needed.  */
	vpxorq	%XMM0, %XMM0, %XMM0

	/* Go to 4x VEC loop.  */
	cmpq	$(CHAR_PER_VEC * 8), %rdx
	ja	L(more_8x_vec)

	/* Handle remainder of size = 4 * VEC + 1 to 8 * VEC without any
	   branches.  */

	/* Load first two VEC from s2 before adjusting addresses.  */
	VMOVU	-(VEC_SIZE * 4)(%rsi, %rdx, CHAR_SIZE), %YMM1
	VMOVU	-(VEC_SIZE * 3)(%rsi, %rdx, CHAR_SIZE), %YMM2
	leaq	-(4 * VEC_SIZE)(%rdi, %rdx, CHAR_SIZE), %rdi
	leaq	-(4 * VEC_SIZE)(%rsi, %rdx, CHAR_SIZE), %rsi

	/* Wait to load from s1 until addressed adjust due to
	   unlamination of microfusion with complex address mode.  */

	/* vpxor will be all 0s if s1 and s2 are equal. Otherwise it
	   will have some 1s.  */
	vpxorq	(%rdi), %YMM1, %YMM1
	vpxorq	(VEC_SIZE)(%rdi), %YMM2, %YMM2

	VMOVU	(VEC_SIZE * 2)(%rsi), %YMM3
	vpxorq	(VEC_SIZE * 2)(%rdi), %YMM3, %YMM3
	/* Or together YMM1, YMM2, and YMM3 into YMM3.  */
	vpternlogd $0xfe, %YMM1, %YMM2, %YMM3

	VMOVU	(VEC_SIZE * 3)(%rsi), %YMM4
	/* Ternary logic to xor (VEC_SIZE * 3)(%rdi) with YMM4 while
	   oring with YMM3. Result is stored in YMM4.  */
	vpternlogd $0xde, (VEC_SIZE * 3)(%rdi), %YMM3, %YMM4
	/* Compare YMM4 with 0. If any 1s s1 and s2 don't match.  */
	VPCMP	$4, %YMM4, %YMM0, %k1
	kmovd	%k1, %ecx
	testl	%ecx, %ecx
	jnz	L(return_vec_0_1_2_3)
	/* NB: eax must be zero to reach here.  */
	ret

	/* NB: aligning 32 here allows for the rest of the jump targets
	   to be tuned for 32 byte alignment. Most important this ensures
	   the L(more_8x_vec) loop is 32 byte aligned.  */
	.p2align 5
L(less_vec):
	/* Check if one or less CHAR. This is necessary for size = 0 but
	   is also faster for size = CHAR_SIZE.  */
	cmpl	$1, %edx
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
	VMOVU	(%rsi), %YMM2
	VPCMP	$4, (%rdi), %YMM2, %k1
	kmovd	%k1, %eax
	/* Create mask in ecx for potentially in bound matches.  */
	bzhil	%edx, %eax, %eax
	jnz	L(return_vec_0)
	ret

	.p2align 4
L(return_vec_0):
	tzcntl	%eax, %eax
# ifdef USE_AS_WMEMCMP
	movl	(%rdi, %rax, CHAR_SIZE), %ecx
	xorl	%edx, %edx
	cmpl	(%rsi, %rax, CHAR_SIZE), %ecx
	/* NB: no partial register stall here because xorl zero idiom
	   above.  */
	setg	%dl
	leal	-1(%rdx, %rdx), %eax
# else
	movzbl	(%rsi, %rax), %ecx
	movzbl	(%rdi, %rax), %eax
	subl	%ecx, %eax
# endif
	ret

	/* NB: No p2align necessary. Alignment  % 16 is naturally 1
	   which is good enough for a target not in a loop.  */
L(return_vec_1):
	tzcntl	%eax, %eax
# ifdef USE_AS_WMEMCMP
	movl	VEC_SIZE(%rdi, %rax, CHAR_SIZE), %ecx
	xorl	%edx, %edx
	cmpl	VEC_SIZE(%rsi, %rax, CHAR_SIZE), %ecx
	setg	%dl
	leal	-1(%rdx, %rdx), %eax
# else
	movzbl	VEC_SIZE(%rsi, %rax), %ecx
	movzbl	VEC_SIZE(%rdi, %rax), %eax
	subl	%ecx, %eax
# endif
	ret

	/* NB: No p2align necessary. Alignment  % 16 is naturally 2
	   which is good enough for a target not in a loop.  */
L(return_vec_2):
	tzcntl	%eax, %eax
# ifdef USE_AS_WMEMCMP
	movl	(VEC_SIZE * 2)(%rdi, %rax, CHAR_SIZE), %ecx
	xorl	%edx, %edx
	cmpl	(VEC_SIZE * 2)(%rsi, %rax, CHAR_SIZE), %ecx
	setg	%dl
	leal	-1(%rdx, %rdx), %eax
# else
	movzbl	(VEC_SIZE * 2)(%rsi, %rax), %ecx
	movzbl	(VEC_SIZE * 2)(%rdi, %rax), %eax
	subl	%ecx, %eax
# endif
	ret

	.p2align 4
L(8x_return_vec_0_1_2_3):
	/* Returning from L(more_8x_vec) requires restoring rsi.  */
	addq	%rdi, %rsi
L(return_vec_0_1_2_3):
	VPCMP	$4, %YMM1, %YMM0, %k0
	kmovd	%k0, %eax
	testl	%eax, %eax
	jnz	L(return_vec_0)

	VPCMP	$4, %YMM2, %YMM0, %k0
	kmovd	%k0, %eax
	testl	%eax, %eax
	jnz	L(return_vec_1)

	VPCMP	$4, %YMM3, %YMM0, %k0
	kmovd	%k0, %eax
	testl	%eax, %eax
	jnz	L(return_vec_2)
L(return_vec_3):
	tzcntl	%ecx, %ecx
# ifdef USE_AS_WMEMCMP
	movl	(VEC_SIZE * 3)(%rdi, %rcx, CHAR_SIZE), %eax
	xorl	%edx, %edx
	cmpl	(VEC_SIZE * 3)(%rsi, %rcx, CHAR_SIZE), %eax
	setg	%dl
	leal	-1(%rdx, %rdx), %eax
# else
	movzbl	(VEC_SIZE * 3)(%rdi, %rcx), %eax
	movzbl	(VEC_SIZE * 3)(%rsi, %rcx), %ecx
	subl	%ecx, %eax
# endif
	ret

	.p2align 4
L(more_8x_vec):
	/* Set end of s1 in rdx.  */
	leaq	-(VEC_SIZE * 4)(%rdi, %rdx, CHAR_SIZE), %rdx
	/* rsi stores s2 - s1. This allows loop to only update one
	   pointer.  */
	subq	%rdi, %rsi
	/* Align s1 pointer.  */
	andq	$-VEC_SIZE, %rdi
	/* Adjust because first 4x vec where check already.  */
	subq	$-(VEC_SIZE * 4), %rdi
	.p2align 4
L(loop_4x_vec):
	VMOVU	(%rsi, %rdi), %YMM1
	vpxorq	(%rdi), %YMM1, %YMM1

	VMOVU	VEC_SIZE(%rsi, %rdi), %YMM2
	vpxorq	VEC_SIZE(%rdi), %YMM2, %YMM2

	VMOVU	(VEC_SIZE * 2)(%rsi, %rdi), %YMM3
	vpxorq	(VEC_SIZE * 2)(%rdi), %YMM3, %YMM3
	vpternlogd $0xfe, %YMM1, %YMM2, %YMM3

	VMOVU	(VEC_SIZE * 3)(%rsi, %rdi), %YMM4
	vpternlogd $0xde, (VEC_SIZE * 3)(%rdi), %YMM3, %YMM4
	VPCMP	$4, %YMM4, %YMM0, %k1
	kmovd	%k1, %ecx
	testl	%ecx, %ecx
	jnz	L(8x_return_vec_0_1_2_3)
	subq	$-(VEC_SIZE * 4), %rdi
	cmpq	%rdx, %rdi
	jb	L(loop_4x_vec)

	subq	%rdx, %rdi
	/* rdi has 4 * VEC_SIZE - remaining length.  */
	cmpl	$(VEC_SIZE * 3), %edi
	jae	L(8x_last_1x_vec)
	/* Load regardless of branch.  */
	VMOVU	(VEC_SIZE * 2)(%rsi, %rdx), %YMM3
	cmpl	$(VEC_SIZE * 2), %edi
	jae	L(8x_last_2x_vec)

	VMOVU	(%rsi, %rdx), %YMM1
	vpxorq	(%rdx), %YMM1, %YMM1

	VMOVU	VEC_SIZE(%rsi, %rdx), %YMM2
	vpxorq	VEC_SIZE(%rdx), %YMM2, %YMM2

	vpxorq	(VEC_SIZE * 2)(%rdx), %YMM3, %YMM3
	vpternlogd $0xfe, %YMM1, %YMM2, %YMM3

	VMOVU	(VEC_SIZE * 3)(%rsi, %rdx), %YMM4
	vpternlogd $0xde, (VEC_SIZE * 3)(%rdx), %YMM3, %YMM4
	VPCMP	$4, %YMM4, %YMM0, %k1
	kmovd	%k1, %ecx
	/* Restore s1 pointer to rdi.  */
	movq	%rdx, %rdi
	testl	%ecx, %ecx
	jnz	L(8x_return_vec_0_1_2_3)
	/* NB: eax must be zero to reach here.  */
	ret

	/* Only entry is from L(more_8x_vec).  */
	.p2align 4
L(8x_last_2x_vec):
	VPCMP	$4, (VEC_SIZE * 2)(%rdx), %YMM3, %k1
	kmovd	%k1, %eax
	testl	%eax, %eax
	jnz	L(8x_return_vec_2)
	/* Naturally aligned to 16 bytes.  */
L(8x_last_1x_vec):
	VMOVU	(VEC_SIZE * 3)(%rsi, %rdx), %YMM1
	VPCMP	$4, (VEC_SIZE * 3)(%rdx), %YMM1, %k1
	kmovd	%k1, %eax
	testl	%eax, %eax
	jnz	L(8x_return_vec_3)
	ret

	.p2align 4
L(last_2x_vec):
	/* Check second to last VEC.  */
	VMOVU	-(VEC_SIZE * 2)(%rsi, %rdx, CHAR_SIZE), %YMM1
	VPCMP	$4, -(VEC_SIZE * 2)(%rdi, %rdx, CHAR_SIZE), %YMM1, %k1
	kmovd	%k1, %eax
	testl	%eax, %eax
	jnz	L(return_vec_1_end)

	/* Check last VEC.  */
	.p2align 4
L(last_1x_vec):
	VMOVU	-(VEC_SIZE * 1)(%rsi, %rdx, CHAR_SIZE), %YMM1
	VPCMP	$4, -(VEC_SIZE * 1)(%rdi, %rdx, CHAR_SIZE), %YMM1, %k1
	kmovd	%k1, %eax
	testl	%eax, %eax
	jnz	L(return_vec_0_end)
	ret

	.p2align 4
L(8x_return_vec_2):
	subq	$VEC_SIZE, %rdx
L(8x_return_vec_3):
	tzcntl	%eax, %eax
# ifdef USE_AS_WMEMCMP
	leaq	(%rdx, %rax, CHAR_SIZE), %rax
	movl	(VEC_SIZE * 3)(%rax), %ecx
	xorl	%edx, %edx
	cmpl	(VEC_SIZE * 3)(%rsi, %rax), %ecx
	setg	%dl
	leal	-1(%rdx, %rdx), %eax
# else
	addq	%rdx, %rax
	movzbl	(VEC_SIZE * 3)(%rsi, %rax), %ecx
	movzbl	(VEC_SIZE * 3)(%rax), %eax
	subl	%ecx, %eax
# endif
	ret

	.p2align 4
L(return_vec_0_end):
	tzcntl	%eax, %eax
	addl	%edx, %eax
# ifdef USE_AS_WMEMCMP
	movl	-VEC_SIZE(%rdi, %rax, CHAR_SIZE), %ecx
	xorl	%edx, %edx
	cmpl	-VEC_SIZE(%rsi, %rax, CHAR_SIZE), %ecx
	setg	%dl
	leal	-1(%rdx, %rdx), %eax
# else
	movzbl	-VEC_SIZE(%rsi, %rax), %ecx
	movzbl	-VEC_SIZE(%rdi, %rax), %eax
	subl	%ecx, %eax
# endif
	ret

	.p2align 4
L(return_vec_1_end):
	tzcntl	%eax, %eax
	addl	%edx, %eax
# ifdef USE_AS_WMEMCMP
	movl	-(VEC_SIZE * 2)(%rdi, %rax, CHAR_SIZE), %ecx
	xorl	%edx, %edx
	cmpl	-(VEC_SIZE * 2)(%rsi, %rax, CHAR_SIZE), %ecx
	setg	%dl
	leal	-1(%rdx, %rdx), %eax
# else
	movzbl	-(VEC_SIZE * 2)(%rsi, %rax), %ecx
	movzbl	-(VEC_SIZE * 2)(%rdi, %rax), %eax
	subl	%ecx, %eax
# endif
	ret


	.p2align 4
L(page_cross_less_vec):
	/* if USE_AS_WMEMCMP it can only be 0, 4, 8, 12, 16, 20, 24, 28
	   bytes.  */
	cmpl	$(16 / CHAR_SIZE), %edx
	jae	L(between_16_31)
# ifndef USE_AS_WMEMCMP
	cmpl	$8, %edx
	jae	L(between_8_15)
	cmpl	$4, %edx
	jae	L(between_4_7)
L(between_2_3):
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
	ret
	.p2align 4
L(one_or_less):
	jb	L(zero)
	movzbl	(%rsi), %ecx
	movzbl	(%rdi), %eax
	subl	%ecx, %eax
	ret

	.p2align 4
L(between_8_15):
# endif
	/* If USE_AS_WMEMCMP fall through into 8-15 byte case.  */
	vmovq	(%rdi), %XMM1
	vmovq	(%rsi), %XMM2
	VPCMP	$4, %XMM1, %XMM2, %k1
	kmovd	%k1, %eax
	testl	%eax, %eax
	jnz	L(return_vec_0)
	/* Use overlapping loads to avoid branches.  */
	leaq	-8(%rdi, %rdx, CHAR_SIZE), %rdi
	leaq	-8(%rsi, %rdx, CHAR_SIZE), %rsi
	vmovq	(%rdi), %XMM1
	vmovq	(%rsi), %XMM2
	VPCMP	$4, %XMM1, %XMM2, %k1
	kmovd	%k1, %eax
	testl	%eax, %eax
	jnz	L(return_vec_0)
	ret

	.p2align 4
L(zero):
	xorl	%eax, %eax
	ret

	.p2align 4
L(between_16_31):
	/* From 16 to 31 bytes.  No branch when size == 16.  */
	VMOVU	(%rsi), %XMM2
	VPCMP	$4, (%rdi), %XMM2, %k1
	kmovd	%k1, %eax
	testl	%eax, %eax
	jnz	L(return_vec_0)

	/* Use overlapping loads to avoid branches.  */

	VMOVU	-16(%rsi, %rdx, CHAR_SIZE), %XMM2
	leaq	-16(%rdi, %rdx, CHAR_SIZE), %rdi
	leaq	-16(%rsi, %rdx, CHAR_SIZE), %rsi
	VPCMP	$4, (%rdi), %XMM2, %k1
	kmovd	%k1, %eax
	testl	%eax, %eax
	jnz	L(return_vec_0)
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
	ret
# endif

END (MEMCMP)
#endif
