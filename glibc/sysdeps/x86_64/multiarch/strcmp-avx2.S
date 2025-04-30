/* strcmp/wcscmp/strncmp/wcsncmp optimized with AVX2.
   Copyright (C) 2018-2021 Free Software Foundation, Inc.
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

# ifndef STRCMP
#  define STRCMP	__strcmp_avx2
# endif

# define PAGE_SIZE	4096

/* VEC_SIZE = Number of bytes in a ymm register */
# define VEC_SIZE	32

/* Shift for dividing by (VEC_SIZE * 4).  */
# define DIVIDE_BY_VEC_4_SHIFT	7
# if (VEC_SIZE * 4) != (1 << DIVIDE_BY_VEC_4_SHIFT)
#  error (VEC_SIZE * 4) != (1 << DIVIDE_BY_VEC_4_SHIFT)
# endif

# ifdef USE_AS_WCSCMP
/* Compare packed dwords.  */
#  define VPCMPEQ	vpcmpeqd
/* Compare packed dwords and store minimum.  */
#  define VPMINU	vpminud
/* 1 dword char == 4 bytes.  */
#  define SIZE_OF_CHAR	4
# else
/* Compare packed bytes.  */
#  define VPCMPEQ	vpcmpeqb
/* Compare packed bytes and store minimum.  */
#  define VPMINU	vpminub
/* 1 byte char == 1 byte.  */
#  define SIZE_OF_CHAR	1
# endif

# ifndef VZEROUPPER
#  define VZEROUPPER	vzeroupper
# endif

# ifndef SECTION
#  define SECTION(p)	p##.avx
# endif

/* Warning!
           wcscmp/wcsncmp have to use SIGNED comparison for elements.
           strcmp/strncmp have to use UNSIGNED comparison for elements.
*/

/* The main idea of the string comparison (byte or dword) using AVX2
   consists of comparing (VPCMPEQ) two ymm vectors. The latter can be on
   either packed bytes or dwords depending on USE_AS_WCSCMP. In order
   to check the null char, algorithm keeps the matched bytes/dwords,
   requiring two more AVX2 instructions (VPMINU and VPCMPEQ). In general,
   the costs of comparing VEC_SIZE bytes (32-bytes) are two VPCMPEQ and
   one VPMINU instructions, together with movdqu and testl instructions.
   Main loop (away from from page boundary) compares 4 vectors are a time,
   effectively comparing 4 x VEC_SIZE bytes (128 bytes) on each loop.

   The routine strncmp/wcsncmp (enabled by defining USE_AS_STRNCMP) logic
   is the same as strcmp, except that an a maximum offset is tracked.  If
   the maximum offset is reached before a difference is found, zero is
   returned.  */

	.section SECTION(.text),"ax",@progbits
ENTRY (STRCMP)
# ifdef USE_AS_STRNCMP
	/* Check for simple cases (0 or 1) in offset.  */
	cmp	$1, %RDX_LP
	je	L(char0)
	jb	L(zero)
#  ifdef USE_AS_WCSCMP
	/* Convert units: from wide to byte char.  */
	shl	$2, %RDX_LP
#  endif
	/* Register %r11 tracks the maximum offset.  */
	mov	%RDX_LP, %R11_LP
# endif
	movl	%edi, %eax
	xorl	%edx, %edx
	/* Make %xmm7 (%ymm7) all zeros in this function.  */
	vpxor	%xmm7, %xmm7, %xmm7
	orl	%esi, %eax
	andl	$(PAGE_SIZE - 1), %eax
	cmpl	$(PAGE_SIZE - (VEC_SIZE * 4)), %eax
	jg	L(cross_page)
	/* Start comparing 4 vectors.  */
	vmovdqu	(%rdi), %ymm1
	VPCMPEQ	(%rsi), %ymm1, %ymm0
	VPMINU	%ymm1, %ymm0, %ymm0
	VPCMPEQ	%ymm7, %ymm0, %ymm0
	vpmovmskb %ymm0, %ecx
	testl	%ecx, %ecx
	je	L(next_3_vectors)
	tzcntl	%ecx, %edx
# ifdef USE_AS_STRNCMP
	/* Return 0 if the mismatched index (%rdx) is after the maximum
	   offset (%r11).   */
	cmpq	%r11, %rdx
	jae	L(zero)
# endif
# ifdef USE_AS_WCSCMP
	xorl	%eax, %eax
	movl	(%rdi, %rdx), %ecx
	cmpl	(%rsi, %rdx), %ecx
	je	L(return)
L(wcscmp_return):
	setl	%al
	negl	%eax
	orl	$1, %eax
L(return):
# else
	movzbl	(%rdi, %rdx), %eax
	movzbl	(%rsi, %rdx), %edx
	subl	%edx, %eax
# endif
L(return_vzeroupper):
	ZERO_UPPER_VEC_REGISTERS_RETURN

	.p2align 4
L(return_vec_size):
	tzcntl	%ecx, %edx
# ifdef USE_AS_STRNCMP
	/* Return 0 if the mismatched index (%rdx + VEC_SIZE) is after
	   the maximum offset (%r11).  */
	addq	$VEC_SIZE, %rdx
	cmpq	%r11, %rdx
	jae	L(zero)
#  ifdef USE_AS_WCSCMP
	xorl	%eax, %eax
	movl	(%rdi, %rdx), %ecx
	cmpl	(%rsi, %rdx), %ecx
	jne	L(wcscmp_return)
#  else
	movzbl	(%rdi, %rdx), %eax
	movzbl	(%rsi, %rdx), %edx
	subl	%edx, %eax
#  endif
# else
#  ifdef USE_AS_WCSCMP
	xorl	%eax, %eax
	movl	VEC_SIZE(%rdi, %rdx), %ecx
	cmpl	VEC_SIZE(%rsi, %rdx), %ecx
	jne	L(wcscmp_return)
#  else
	movzbl	VEC_SIZE(%rdi, %rdx), %eax
	movzbl	VEC_SIZE(%rsi, %rdx), %edx
	subl	%edx, %eax
#  endif
# endif
	VZEROUPPER_RETURN

	.p2align 4
L(return_2_vec_size):
	tzcntl	%ecx, %edx
# ifdef USE_AS_STRNCMP
	/* Return 0 if the mismatched index (%rdx + 2 * VEC_SIZE) is
	   after the maximum offset (%r11).  */
	addq	$(VEC_SIZE * 2), %rdx
	cmpq	%r11, %rdx
	jae	L(zero)
#  ifdef USE_AS_WCSCMP
	xorl	%eax, %eax
	movl	(%rdi, %rdx), %ecx
	cmpl	(%rsi, %rdx), %ecx
	jne	L(wcscmp_return)
#  else
	movzbl	(%rdi, %rdx), %eax
	movzbl	(%rsi, %rdx), %edx
	subl	%edx, %eax
#  endif
# else
#  ifdef USE_AS_WCSCMP
	xorl	%eax, %eax
	movl	(VEC_SIZE * 2)(%rdi, %rdx), %ecx
	cmpl	(VEC_SIZE * 2)(%rsi, %rdx), %ecx
	jne	L(wcscmp_return)
#  else
	movzbl	(VEC_SIZE * 2)(%rdi, %rdx), %eax
	movzbl	(VEC_SIZE * 2)(%rsi, %rdx), %edx
	subl	%edx, %eax
#  endif
# endif
	VZEROUPPER_RETURN

	.p2align 4
L(return_3_vec_size):
	tzcntl	%ecx, %edx
# ifdef USE_AS_STRNCMP
	/* Return 0 if the mismatched index (%rdx + 3 * VEC_SIZE) is
	   after the maximum offset (%r11).  */
	addq	$(VEC_SIZE * 3), %rdx
	cmpq	%r11, %rdx
	jae	L(zero)
#  ifdef USE_AS_WCSCMP
	xorl	%eax, %eax
	movl	(%rdi, %rdx), %ecx
	cmpl	(%rsi, %rdx), %ecx
	jne	L(wcscmp_return)
#  else
	movzbl	(%rdi, %rdx), %eax
	movzbl	(%rsi, %rdx), %edx
	subl	%edx, %eax
#  endif
# else
#  ifdef USE_AS_WCSCMP
	xorl	%eax, %eax
	movl	(VEC_SIZE * 3)(%rdi, %rdx), %ecx
	cmpl	(VEC_SIZE * 3)(%rsi, %rdx), %ecx
	jne	L(wcscmp_return)
#  else
	movzbl	(VEC_SIZE * 3)(%rdi, %rdx), %eax
	movzbl	(VEC_SIZE * 3)(%rsi, %rdx), %edx
	subl	%edx, %eax
#  endif
# endif
	VZEROUPPER_RETURN

	.p2align 4
L(next_3_vectors):
	vmovdqu	VEC_SIZE(%rdi), %ymm6
	VPCMPEQ	VEC_SIZE(%rsi), %ymm6, %ymm3
	VPMINU	%ymm6, %ymm3, %ymm3
	VPCMPEQ	%ymm7, %ymm3, %ymm3
	vpmovmskb %ymm3, %ecx
	testl	%ecx, %ecx
	jne	L(return_vec_size)
	vmovdqu	(VEC_SIZE * 2)(%rdi), %ymm5
	vmovdqu	(VEC_SIZE * 3)(%rdi), %ymm4
	vmovdqu	(VEC_SIZE * 3)(%rsi), %ymm0
	VPCMPEQ	(VEC_SIZE * 2)(%rsi), %ymm5, %ymm2
	VPMINU	%ymm5, %ymm2, %ymm2
	VPCMPEQ	%ymm4, %ymm0, %ymm0
	VPCMPEQ	%ymm7, %ymm2, %ymm2
	vpmovmskb %ymm2, %ecx
	testl	%ecx, %ecx
	jne	L(return_2_vec_size)
	VPMINU	%ymm4, %ymm0, %ymm0
	VPCMPEQ	%ymm7, %ymm0, %ymm0
	vpmovmskb %ymm0, %ecx
	testl	%ecx, %ecx
	jne	L(return_3_vec_size)
L(main_loop_header):
	leaq	(VEC_SIZE * 4)(%rdi), %rdx
	movl	$PAGE_SIZE, %ecx
	/* Align load via RAX.  */
	andq	$-(VEC_SIZE * 4), %rdx
	subq	%rdi, %rdx
	leaq	(%rdi, %rdx), %rax
# ifdef USE_AS_STRNCMP
	/* Starting from this point, the maximum offset, or simply the
	   'offset', DECREASES by the same amount when base pointers are
	   moved forward.  Return 0 when:
	     1) On match: offset <= the matched vector index.
	     2) On mistmach, offset is before the mistmatched index.
	 */
	subq	%rdx, %r11
	jbe	L(zero)
# endif
	addq	%rsi, %rdx
	movq	%rdx, %rsi
	andl	$(PAGE_SIZE - 1), %esi
	/* Number of bytes before page crossing.  */
	subq	%rsi, %rcx
	/* Number of VEC_SIZE * 4 blocks before page crossing.  */
	shrq	$DIVIDE_BY_VEC_4_SHIFT, %rcx
	/* ESI: Number of VEC_SIZE * 4 blocks before page crossing.   */
	movl	%ecx, %esi
	jmp	L(loop_start)

	.p2align 4
L(loop):
# ifdef USE_AS_STRNCMP
	/* Base pointers are moved forward by 4 * VEC_SIZE.  Decrease
	   the maximum offset (%r11) by the same amount.  */
	subq	$(VEC_SIZE * 4), %r11
	jbe	L(zero)
# endif
	addq	$(VEC_SIZE * 4), %rax
	addq	$(VEC_SIZE * 4), %rdx
L(loop_start):
	testl	%esi, %esi
	leal	-1(%esi), %esi
	je	L(loop_cross_page)
L(back_to_loop):
	/* Main loop, comparing 4 vectors are a time.  */
	vmovdqa	(%rax), %ymm0
	vmovdqa	VEC_SIZE(%rax), %ymm3
	VPCMPEQ	(%rdx), %ymm0, %ymm4
	VPCMPEQ	VEC_SIZE(%rdx), %ymm3, %ymm1
	VPMINU	%ymm0, %ymm4, %ymm4
	VPMINU	%ymm3, %ymm1, %ymm1
	vmovdqa	(VEC_SIZE * 2)(%rax), %ymm2
	VPMINU	%ymm1, %ymm4, %ymm0
	vmovdqa	(VEC_SIZE * 3)(%rax), %ymm3
	VPCMPEQ	(VEC_SIZE * 2)(%rdx), %ymm2, %ymm5
	VPCMPEQ	(VEC_SIZE * 3)(%rdx), %ymm3, %ymm6
	VPMINU	%ymm2, %ymm5, %ymm5
	VPMINU	%ymm3, %ymm6, %ymm6
	VPMINU	%ymm5, %ymm0, %ymm0
	VPMINU	%ymm6, %ymm0, %ymm0
	VPCMPEQ	%ymm7, %ymm0, %ymm0

	/* Test each mask (32 bits) individually because for VEC_SIZE
	   == 32 is not possible to OR the four masks and keep all bits
	   in a 64-bit integer register, differing from SSE2 strcmp
	   where ORing is possible.  */
	vpmovmskb %ymm0, %ecx
	testl	%ecx, %ecx
	je	L(loop)
	VPCMPEQ	%ymm7, %ymm4, %ymm0
	vpmovmskb %ymm0, %edi
	testl	%edi, %edi
	je	L(test_vec)
	tzcntl	%edi, %ecx
# ifdef USE_AS_STRNCMP
	cmpq	%rcx, %r11
	jbe	L(zero)
#  ifdef USE_AS_WCSCMP
	movq	%rax, %rsi
	xorl	%eax, %eax
	movl	(%rsi, %rcx), %edi
	cmpl	(%rdx, %rcx), %edi
	jne	L(wcscmp_return)
#  else
	movzbl	(%rax, %rcx), %eax
	movzbl	(%rdx, %rcx), %edx
	subl	%edx, %eax
#  endif
# else
#  ifdef USE_AS_WCSCMP
	movq	%rax, %rsi
	xorl	%eax, %eax
	movl	(%rsi, %rcx), %edi
	cmpl	(%rdx, %rcx), %edi
	jne	L(wcscmp_return)
#  else
	movzbl	(%rax, %rcx), %eax
	movzbl	(%rdx, %rcx), %edx
	subl	%edx, %eax
#  endif
# endif
	VZEROUPPER_RETURN

	.p2align 4
L(test_vec):
# ifdef USE_AS_STRNCMP
	/* The first vector matched.  Return 0 if the maximum offset
	   (%r11) <= VEC_SIZE.  */
	cmpq	$VEC_SIZE, %r11
	jbe	L(zero)
# endif
	VPCMPEQ	%ymm7, %ymm1, %ymm1
	vpmovmskb %ymm1, %ecx
	testl	%ecx, %ecx
	je	L(test_2_vec)
	tzcntl	%ecx, %edi
# ifdef USE_AS_STRNCMP
	addq	$VEC_SIZE, %rdi
	cmpq	%rdi, %r11
	jbe	L(zero)
#  ifdef USE_AS_WCSCMP
	movq	%rax, %rsi
	xorl	%eax, %eax
	movl	(%rsi, %rdi), %ecx
	cmpl	(%rdx, %rdi), %ecx
	jne	L(wcscmp_return)
#  else
	movzbl	(%rax, %rdi), %eax
	movzbl	(%rdx, %rdi), %edx
	subl	%edx, %eax
#  endif
# else
#  ifdef USE_AS_WCSCMP
	movq	%rax, %rsi
	xorl	%eax, %eax
	movl	VEC_SIZE(%rsi, %rdi), %ecx
	cmpl	VEC_SIZE(%rdx, %rdi), %ecx
	jne	L(wcscmp_return)
#  else
	movzbl	VEC_SIZE(%rax, %rdi), %eax
	movzbl	VEC_SIZE(%rdx, %rdi), %edx
	subl	%edx, %eax
#  endif
# endif
	VZEROUPPER_RETURN

	.p2align 4
L(test_2_vec):
# ifdef USE_AS_STRNCMP
	/* The first 2 vectors matched.  Return 0 if the maximum offset
	   (%r11) <= 2 * VEC_SIZE.  */
	cmpq	$(VEC_SIZE * 2), %r11
	jbe	L(zero)
# endif
	VPCMPEQ	%ymm7, %ymm5, %ymm5
	vpmovmskb %ymm5, %ecx
	testl	%ecx, %ecx
	je	L(test_3_vec)
	tzcntl	%ecx, %edi
# ifdef USE_AS_STRNCMP
	addq	$(VEC_SIZE * 2), %rdi
	cmpq	%rdi, %r11
	jbe	L(zero)
#  ifdef USE_AS_WCSCMP
	movq	%rax, %rsi
	xorl	%eax, %eax
	movl	(%rsi, %rdi), %ecx
	cmpl	(%rdx, %rdi), %ecx
	jne	L(wcscmp_return)
#  else
	movzbl	(%rax, %rdi), %eax
	movzbl	(%rdx, %rdi), %edx
	subl	%edx, %eax
#  endif
# else
#  ifdef USE_AS_WCSCMP
	movq	%rax, %rsi
	xorl	%eax, %eax
	movl	(VEC_SIZE * 2)(%rsi, %rdi), %ecx
	cmpl	(VEC_SIZE * 2)(%rdx, %rdi), %ecx
	jne	L(wcscmp_return)
#  else
	movzbl	(VEC_SIZE * 2)(%rax, %rdi), %eax
	movzbl	(VEC_SIZE * 2)(%rdx, %rdi), %edx
	subl	%edx, %eax
#  endif
# endif
	VZEROUPPER_RETURN

	.p2align 4
L(test_3_vec):
# ifdef USE_AS_STRNCMP
	/* The first 3 vectors matched.  Return 0 if the maximum offset
	   (%r11) <= 3 * VEC_SIZE.  */
	cmpq	$(VEC_SIZE * 3), %r11
	jbe	L(zero)
# endif
	VPCMPEQ	%ymm7, %ymm6, %ymm6
	vpmovmskb %ymm6, %esi
	tzcntl	%esi, %ecx
# ifdef USE_AS_STRNCMP
	addq	$(VEC_SIZE * 3), %rcx
	cmpq	%rcx, %r11
	jbe	L(zero)
#  ifdef USE_AS_WCSCMP
	movq	%rax, %rsi
	xorl	%eax, %eax
	movl	(%rsi, %rcx), %esi
	cmpl	(%rdx, %rcx), %esi
	jne	L(wcscmp_return)
#  else
	movzbl	(%rax, %rcx), %eax
	movzbl	(%rdx, %rcx), %edx
	subl	%edx, %eax
#  endif
# else
#  ifdef USE_AS_WCSCMP
	movq	%rax, %rsi
	xorl	%eax, %eax
	movl	(VEC_SIZE * 3)(%rsi, %rcx), %esi
	cmpl	(VEC_SIZE * 3)(%rdx, %rcx), %esi
	jne	L(wcscmp_return)
#  else
	movzbl	(VEC_SIZE * 3)(%rax, %rcx), %eax
	movzbl	(VEC_SIZE * 3)(%rdx, %rcx), %edx
	subl	%edx, %eax
#  endif
# endif
	VZEROUPPER_RETURN

	.p2align 4
L(loop_cross_page):
	xorl	%r10d, %r10d
	movq	%rdx, %rcx
	/* Align load via RDX.  We load the extra ECX bytes which should
	   be ignored.  */
	andl	$((VEC_SIZE * 4) - 1), %ecx
	/* R10 is -RCX.  */
	subq	%rcx, %r10

	/* This works only if VEC_SIZE * 2 == 64. */
# if (VEC_SIZE * 2) != 64
#  error (VEC_SIZE * 2) != 64
# endif

	/* Check if the first VEC_SIZE * 2 bytes should be ignored.  */
	cmpl	$(VEC_SIZE * 2), %ecx
	jge	L(loop_cross_page_2_vec)

	vmovdqu	(%rax, %r10), %ymm2
	vmovdqu	VEC_SIZE(%rax, %r10), %ymm3
	VPCMPEQ	(%rdx, %r10), %ymm2, %ymm0
	VPCMPEQ	VEC_SIZE(%rdx, %r10), %ymm3, %ymm1
	VPMINU	%ymm2, %ymm0, %ymm0
	VPMINU	%ymm3, %ymm1, %ymm1
	VPCMPEQ	%ymm7, %ymm0, %ymm0
	VPCMPEQ	%ymm7, %ymm1, %ymm1

	vpmovmskb %ymm0, %edi
	vpmovmskb %ymm1, %esi

	salq	$32, %rsi
	xorq	%rsi, %rdi

	/* Since ECX < VEC_SIZE * 2, simply skip the first ECX bytes.  */
	shrq	%cl, %rdi

	testq	%rdi, %rdi
	je	L(loop_cross_page_2_vec)
	tzcntq	%rdi, %rcx
# ifdef USE_AS_STRNCMP
	cmpq	%rcx, %r11
	jbe	L(zero)
#  ifdef USE_AS_WCSCMP
	movq	%rax, %rsi
	xorl	%eax, %eax
	movl	(%rsi, %rcx), %edi
	cmpl	(%rdx, %rcx), %edi
	jne	L(wcscmp_return)
#  else
	movzbl	(%rax, %rcx), %eax
	movzbl	(%rdx, %rcx), %edx
	subl	%edx, %eax
#  endif
# else
#  ifdef USE_AS_WCSCMP
	movq	%rax, %rsi
	xorl	%eax, %eax
	movl	(%rsi, %rcx), %edi
	cmpl	(%rdx, %rcx), %edi
	jne	L(wcscmp_return)
#  else
	movzbl	(%rax, %rcx), %eax
	movzbl	(%rdx, %rcx), %edx
	subl	%edx, %eax
#  endif
# endif
	VZEROUPPER_RETURN

	.p2align 4
L(loop_cross_page_2_vec):
	/* The first VEC_SIZE * 2 bytes match or are ignored.  */
	vmovdqu	(VEC_SIZE * 2)(%rax, %r10), %ymm2
	vmovdqu	(VEC_SIZE * 3)(%rax, %r10), %ymm3
	VPCMPEQ	(VEC_SIZE * 2)(%rdx, %r10), %ymm2, %ymm5
	VPMINU	%ymm2, %ymm5, %ymm5
	VPCMPEQ	(VEC_SIZE * 3)(%rdx, %r10), %ymm3, %ymm6
	VPCMPEQ	%ymm7, %ymm5, %ymm5
	VPMINU	%ymm3, %ymm6, %ymm6
	VPCMPEQ	%ymm7, %ymm6, %ymm6

	vpmovmskb %ymm5, %edi
	vpmovmskb %ymm6, %esi

	salq	$32, %rsi
	xorq	%rsi, %rdi

	xorl	%r8d, %r8d
	/* If ECX > VEC_SIZE * 2, skip ECX - (VEC_SIZE * 2) bytes.  */
	subl	$(VEC_SIZE * 2), %ecx
	jle	1f
	/* Skip ECX bytes.  */
	shrq	%cl, %rdi
	/* R8 has number of bytes skipped.  */
	movl	%ecx, %r8d
1:
	/* Before jumping back to the loop, set ESI to the number of
	   VEC_SIZE * 4 blocks before page crossing.  */
	movl	$(PAGE_SIZE / (VEC_SIZE * 4) - 1), %esi

	testq	%rdi, %rdi
# ifdef USE_AS_STRNCMP
	/* At this point, if %rdi value is 0, it already tested
	   VEC_SIZE*4+%r10 byte starting from %rax. This label
	   checks whether strncmp maximum offset reached or not.  */
	je	L(string_nbyte_offset_check)
# else
	je	L(back_to_loop)
# endif
	tzcntq	%rdi, %rcx
	addq	%r10, %rcx
	/* Adjust for number of bytes skipped.  */
	addq	%r8, %rcx
# ifdef USE_AS_STRNCMP
	addq	$(VEC_SIZE * 2), %rcx
	subq	%rcx, %r11
	jbe	L(zero)
#  ifdef USE_AS_WCSCMP
	movq	%rax, %rsi
	xorl	%eax, %eax
	movl	(%rsi, %rcx), %edi
	cmpl	(%rdx, %rcx), %edi
	jne	L(wcscmp_return)
#  else
	movzbl	(%rax, %rcx), %eax
	movzbl	(%rdx, %rcx), %edx
	subl	%edx, %eax
#  endif
# else
#  ifdef USE_AS_WCSCMP
	movq	%rax, %rsi
	xorl	%eax, %eax
	movl	(VEC_SIZE * 2)(%rsi, %rcx), %edi
	cmpl	(VEC_SIZE * 2)(%rdx, %rcx), %edi
	jne	L(wcscmp_return)
#  else
	movzbl	(VEC_SIZE * 2)(%rax, %rcx), %eax
	movzbl	(VEC_SIZE * 2)(%rdx, %rcx), %edx
	subl	%edx, %eax
#  endif
# endif
	VZEROUPPER_RETURN

# ifdef USE_AS_STRNCMP
L(string_nbyte_offset_check):
	leaq	(VEC_SIZE * 4)(%r10), %r10
	cmpq	%r10, %r11
	jbe	L(zero)
	jmp	L(back_to_loop)
# endif

	.p2align 4
L(cross_page_loop):
	/* Check one byte/dword at a time.  */
# ifdef USE_AS_WCSCMP
	cmpl	%ecx, %eax
# else
	subl	%ecx, %eax
# endif
	jne	L(different)
	addl	$SIZE_OF_CHAR, %edx
	cmpl	$(VEC_SIZE * 4), %edx
	je	L(main_loop_header)
# ifdef USE_AS_STRNCMP
	cmpq	%r11, %rdx
	jae	L(zero)
# endif
# ifdef USE_AS_WCSCMP
	movl	(%rdi, %rdx), %eax
	movl	(%rsi, %rdx), %ecx
# else
	movzbl	(%rdi, %rdx), %eax
	movzbl	(%rsi, %rdx), %ecx
# endif
	/* Check null char.  */
	testl	%eax, %eax
	jne	L(cross_page_loop)
	/* Since %eax == 0, subtract is OK for both SIGNED and UNSIGNED
	   comparisons.  */
	subl	%ecx, %eax
# ifndef USE_AS_WCSCMP
L(different):
# endif
	VZEROUPPER_RETURN

# ifdef USE_AS_WCSCMP
	.p2align 4
L(different):
	/* Use movl to avoid modifying EFLAGS.  */
	movl	$0, %eax
	setl	%al
	negl	%eax
	orl	$1, %eax
	VZEROUPPER_RETURN
# endif

# ifdef USE_AS_STRNCMP
	.p2align 4
L(zero):
	xorl	%eax, %eax
	VZEROUPPER_RETURN

	.p2align 4
L(char0):
#  ifdef USE_AS_WCSCMP
	xorl	%eax, %eax
	movl	(%rdi), %ecx
	cmpl	(%rsi), %ecx
	jne	L(wcscmp_return)
#  else
	movzbl	(%rsi), %ecx
	movzbl	(%rdi), %eax
	subl	%ecx, %eax
#  endif
	VZEROUPPER_RETURN
# endif

	.p2align 4
L(last_vector):
	addq	%rdx, %rdi
	addq	%rdx, %rsi
# ifdef USE_AS_STRNCMP
	subq	%rdx, %r11
# endif
	tzcntl	%ecx, %edx
# ifdef USE_AS_STRNCMP
	cmpq	%r11, %rdx
	jae	L(zero)
# endif
# ifdef USE_AS_WCSCMP
	xorl	%eax, %eax
	movl	(%rdi, %rdx), %ecx
	cmpl	(%rsi, %rdx), %ecx
	jne	L(wcscmp_return)
# else
	movzbl	(%rdi, %rdx), %eax
	movzbl	(%rsi, %rdx), %edx
	subl	%edx, %eax
# endif
	VZEROUPPER_RETURN

	/* Comparing on page boundary region requires special treatment:
	   It must done one vector at the time, starting with the wider
	   ymm vector if possible, if not, with xmm. If fetching 16 bytes
	   (xmm) still passes the boundary, byte comparison must be done.
	 */
	.p2align 4
L(cross_page):
	/* Try one ymm vector at a time.  */
	cmpl	$(PAGE_SIZE - VEC_SIZE), %eax
	jg	L(cross_page_1_vector)
L(loop_1_vector):
	vmovdqu	(%rdi, %rdx), %ymm1
	VPCMPEQ	(%rsi, %rdx), %ymm1, %ymm0
	VPMINU	%ymm1, %ymm0, %ymm0
	VPCMPEQ	%ymm7, %ymm0, %ymm0
	vpmovmskb %ymm0, %ecx
	testl	%ecx, %ecx
	jne	L(last_vector)

	addl	$VEC_SIZE, %edx

	addl	$VEC_SIZE, %eax
# ifdef USE_AS_STRNCMP
	/* Return 0 if the current offset (%rdx) >= the maximum offset
	   (%r11).  */
	cmpq	%r11, %rdx
	jae	L(zero)
# endif
	cmpl	$(PAGE_SIZE - VEC_SIZE), %eax
	jle	L(loop_1_vector)
L(cross_page_1_vector):
	/* Less than 32 bytes to check, try one xmm vector.  */
	cmpl	$(PAGE_SIZE - 16), %eax
	jg	L(cross_page_1_xmm)
	vmovdqu	(%rdi, %rdx), %xmm1
	VPCMPEQ	(%rsi, %rdx), %xmm1, %xmm0
	VPMINU	%xmm1, %xmm0, %xmm0
	VPCMPEQ	%xmm7, %xmm0, %xmm0
	vpmovmskb %xmm0, %ecx
	testl	%ecx, %ecx
	jne	L(last_vector)

	addl	$16, %edx
# ifndef USE_AS_WCSCMP
	addl	$16, %eax
# endif
# ifdef USE_AS_STRNCMP
	/* Return 0 if the current offset (%rdx) >= the maximum offset
	   (%r11).  */
	cmpq	%r11, %rdx
	jae	L(zero)
# endif

L(cross_page_1_xmm):
# ifndef USE_AS_WCSCMP
	/* Less than 16 bytes to check, try 8 byte vector.  NB: No need
	   for wcscmp nor wcsncmp since wide char is 4 bytes.   */
	cmpl	$(PAGE_SIZE - 8), %eax
	jg	L(cross_page_8bytes)
	vmovq	(%rdi, %rdx), %xmm1
	vmovq	(%rsi, %rdx), %xmm0
	VPCMPEQ	%xmm0, %xmm1, %xmm0
	VPMINU	%xmm1, %xmm0, %xmm0
	VPCMPEQ	%xmm7, %xmm0, %xmm0
	vpmovmskb %xmm0, %ecx
	/* Only last 8 bits are valid.  */
	andl	$0xff, %ecx
	testl	%ecx, %ecx
	jne	L(last_vector)

	addl	$8, %edx
	addl	$8, %eax
#  ifdef USE_AS_STRNCMP
	/* Return 0 if the current offset (%rdx) >= the maximum offset
	   (%r11).  */
	cmpq	%r11, %rdx
	jae	L(zero)
#  endif

L(cross_page_8bytes):
	/* Less than 8 bytes to check, try 4 byte vector.  */
	cmpl	$(PAGE_SIZE - 4), %eax
	jg	L(cross_page_4bytes)
	vmovd	(%rdi, %rdx), %xmm1
	vmovd	(%rsi, %rdx), %xmm0
	VPCMPEQ	%xmm0, %xmm1, %xmm0
	VPMINU	%xmm1, %xmm0, %xmm0
	VPCMPEQ	%xmm7, %xmm0, %xmm0
	vpmovmskb %xmm0, %ecx
	/* Only last 4 bits are valid.  */
	andl	$0xf, %ecx
	testl	%ecx, %ecx
	jne	L(last_vector)

	addl	$4, %edx
#  ifdef USE_AS_STRNCMP
	/* Return 0 if the current offset (%rdx) >= the maximum offset
	   (%r11).  */
	cmpq	%r11, %rdx
	jae	L(zero)
#  endif

L(cross_page_4bytes):
# endif
	/* Less than 4 bytes to check, try one byte/dword at a time.  */
# ifdef USE_AS_STRNCMP
	cmpq	%r11, %rdx
	jae	L(zero)
# endif
# ifdef USE_AS_WCSCMP
	movl	(%rdi, %rdx), %eax
	movl	(%rsi, %rdx), %ecx
# else
	movzbl	(%rdi, %rdx), %eax
	movzbl	(%rsi, %rdx), %ecx
# endif
	testl	%eax, %eax
	jne	L(cross_page_loop)
	subl	%ecx, %eax
	VZEROUPPER_RETURN
END (STRCMP)
#endif
