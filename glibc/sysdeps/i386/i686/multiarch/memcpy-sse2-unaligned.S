/* memcpy optimized with SSE2 unaligned memory access instructions.
   Copyright (C) 2014-2021 Free Software Foundation, Inc.
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

#if IS_IN (libc) \
    && (defined SHARED \
	|| defined USE_AS_MEMMOVE \
	|| !defined USE_MULTIARCH)

# include <sysdep.h>
# include "asm-syntax.h"

# ifndef MEMCPY
#  define MEMCPY	__memcpy_sse2_unaligned
#  define MEMCPY_CHK	__memcpy_chk_sse2_unaligned
# endif

# ifdef USE_AS_BCOPY
#  define SRC		PARMS
#  define DEST		SRC+4
#  define LEN		DEST+4
# else
#  define DEST		PARMS
#  define SRC		DEST+4
#  define LEN		SRC+4
# endif

# define CFI_PUSH(REG)		\
  cfi_adjust_cfa_offset (4);		\
  cfi_rel_offset (REG, 0)

# define CFI_POP(REG)		\
  cfi_adjust_cfa_offset (-4);		\
  cfi_restore (REG)

# define PUSH(REG)	pushl REG; CFI_PUSH (REG)
# define POP(REG)	popl REG; CFI_POP (REG)

# define PARMS		8		/* Preserve EBX.  */
# define ENTRANCE	PUSH (%ebx);
# define RETURN_END	POP (%ebx); ret
# define RETURN	RETURN_END; CFI_PUSH (%ebx)

	.section .text.sse2,"ax",@progbits
# if !defined USE_AS_BCOPY && defined SHARED
ENTRY (MEMCPY_CHK)
	movl	12(%esp), %eax
	cmpl	%eax, 16(%esp)
	jb	HIDDEN_JUMPTARGET (__chk_fail)
END (MEMCPY_CHK)
# endif

ENTRY (MEMCPY)
	ENTRANCE
	movl	LEN(%esp), %ecx
	movl	SRC(%esp), %eax
	movl	DEST(%esp), %edx
	cmp	%edx, %eax

# ifdef USE_AS_MEMMOVE
	ja	L(check_forward)

L(mm_len_0_or_more_backward):
/* Now do checks for lengths. We do [0..16], [16..32], [32..64], [64..128]
	separately.  */
	cmp	$16, %ecx
	jbe	L(mm_len_0_16_bytes_backward)

	cmpl	$32, %ecx
	ja	L(mm_len_32_or_more_backward)

/* Copy [0..32] and return.  */
	movdqu	(%eax), %xmm0
	movdqu	-16(%eax, %ecx), %xmm1
	movdqu	%xmm0, (%edx)
	movdqu	%xmm1, -16(%edx, %ecx)
	jmp	L(return)

L(mm_len_32_or_more_backward):
	cmpl	$64, %ecx
	ja	L(mm_len_64_or_more_backward)

/* Copy [0..64] and return.  */
	movdqu	(%eax), %xmm0
	movdqu	16(%eax), %xmm1
	movdqu	-16(%eax, %ecx), %xmm2
	movdqu	-32(%eax, %ecx), %xmm3
	movdqu	%xmm0, (%edx)
	movdqu	%xmm1, 16(%edx)
	movdqu	%xmm2, -16(%edx, %ecx)
	movdqu	%xmm3, -32(%edx, %ecx)
	jmp	L(return)

L(mm_len_64_or_more_backward):
	cmpl	$128, %ecx
	ja	L(mm_len_128_or_more_backward)

/* Copy [0..128] and return.  */
	movdqu	(%eax), %xmm0
	movdqu	16(%eax), %xmm1
	movdqu	32(%eax), %xmm2
	movdqu	48(%eax), %xmm3
	movdqu	-64(%eax, %ecx), %xmm4
	movdqu	-48(%eax, %ecx), %xmm5
	movdqu	-32(%eax, %ecx), %xmm6
	movdqu	-16(%eax, %ecx), %xmm7
	movdqu	%xmm0, (%edx)
	movdqu	%xmm1, 16(%edx)
	movdqu	%xmm2, 32(%edx)
	movdqu	%xmm3, 48(%edx)
	movdqu	%xmm4, -64(%edx, %ecx)
	movdqu	%xmm5, -48(%edx, %ecx)
	movdqu	%xmm6, -32(%edx, %ecx)
	movdqu	%xmm7, -16(%edx, %ecx)
	jmp	L(return)

L(mm_len_128_or_more_backward):
	add	%ecx, %eax
	cmp	%edx, %eax
	movl	SRC(%esp), %eax
	jbe	L(forward)
	PUSH (%esi)
	PUSH (%edi)
	PUSH (%ebx)

/* Aligning the address of destination. */
	movdqu	(%eax), %xmm4
	movdqu	16(%eax), %xmm5
	movdqu	32(%eax), %xmm6
	movdqu	48(%eax), %xmm7
	leal	(%edx, %ecx), %esi
	movdqu	-16(%eax, %ecx), %xmm0
	subl	$16, %esp
	movdqu	%xmm0, (%esp)
	mov	%ecx, %edi
	movl	%esi, %ecx
	andl	$-16, %ecx
	leal	(%ecx), %ebx
	subl	%edx, %ebx
	leal	(%eax, %ebx), %eax
	shrl	$6, %ebx

# ifdef SHARED_CACHE_SIZE_HALF
	cmp	$SHARED_CACHE_SIZE_HALF, %edi
# else
#  ifdef PIC
	PUSH (%ebx)
	SETUP_PIC_REG (bx)
	add	$_GLOBAL_OFFSET_TABLE_, %ebx
	cmp	__x86_shared_cache_size_half@GOTOFF(%ebx), %edi
	POP (%ebx)
#  else
	cmp	__x86_shared_cache_size_half, %edi
#  endif
# endif
	jae	L(mm_large_page_loop_backward)

	.p2align 4
L(mm_main_loop_backward):

	prefetcht0 -128(%eax)

	movdqu	-64(%eax), %xmm0
	movdqu	-48(%eax), %xmm1
	movdqu	-32(%eax), %xmm2
	movdqu	-16(%eax), %xmm3
	movaps	%xmm0, -64(%ecx)
	subl	$64, %eax
	movaps	%xmm1, -48(%ecx)
	movaps	%xmm2, -32(%ecx)
	movaps	%xmm3, -16(%ecx)
	subl	$64, %ecx
	sub	$1, %ebx
	jnz	L(mm_main_loop_backward)
	movdqu	(%esp), %xmm0
	addl	$16, %esp
	movdqu	%xmm0, -16(%esi)
	movdqu	%xmm4, (%edx)
	movdqu	%xmm5, 16(%edx)
	movdqu	%xmm6, 32(%edx)
	movdqu	%xmm7, 48(%edx)
	POP (%ebx)
	jmp	L(mm_return_pop_all)

/* Copy [0..16] and return.  */
L(mm_len_0_16_bytes_backward):
	testb	$24, %cl
	jnz	L(mm_len_9_16_bytes_backward)
	testb	$4, %cl
	.p2align 4,,5
	jnz	L(mm_len_5_8_bytes_backward)
	testl	%ecx, %ecx
	.p2align 4,,2
	je	L(return)
	testb	$2, %cl
	.p2align 4,,1
	jne	L(mm_len_3_4_bytes_backward)
	movzbl	-1(%eax,%ecx), %ebx
	movzbl	(%eax), %eax
	movb	%bl, -1(%edx,%ecx)
	movb	%al, (%edx)
	jmp	L(return)

L(mm_len_3_4_bytes_backward):
	movzwl	-2(%eax,%ecx), %ebx
	movzwl	(%eax), %eax
	movw	%bx, -2(%edx,%ecx)
	movw	%ax, (%edx)
	jmp	L(return)

L(mm_len_9_16_bytes_backward):
	PUSH (%esi)
	movl	-4(%eax,%ecx), %ebx
	movl	-8(%eax,%ecx), %esi
	movl	%ebx, -4(%edx,%ecx)
	movl	%esi, -8(%edx,%ecx)
	subl	$8, %ecx
	POP (%esi)
	jmp	L(mm_len_0_16_bytes_backward)

L(mm_len_5_8_bytes_backward):
	movl	(%eax), %ebx
	movl	-4(%eax,%ecx), %eax
	movl	%ebx, (%edx)
	movl	%eax, -4(%edx,%ecx)
	jmp	L(return)

/* Big length copy backward part.  */
	.p2align 4
L(mm_large_page_loop_backward):
	movdqu	-64(%eax), %xmm0
	movdqu	-48(%eax), %xmm1
	movdqu	-32(%eax), %xmm2
	movdqu	-16(%eax), %xmm3
	movntdq	%xmm0, -64(%ecx)
	subl	$64, %eax
	movntdq	%xmm1, -48(%ecx)
	movntdq	%xmm2, -32(%ecx)
	movntdq	%xmm3, -16(%ecx)
	subl	$64, %ecx
	sub	$1, %ebx
	jnz	L(mm_large_page_loop_backward)
	sfence
	movdqu	(%esp), %xmm0
	addl	$16, %esp
	movdqu	%xmm0, -16(%esi)
	movdqu	%xmm4, (%edx)
	movdqu	%xmm5, 16(%edx)
	movdqu	%xmm6, 32(%edx)
	movdqu	%xmm7, 48(%edx)
	POP (%ebx)
	jmp	L(mm_return_pop_all)

L(check_forward):
	add	%edx, %ecx
	cmp	%eax, %ecx
	movl	LEN(%esp), %ecx
	jbe	L(forward)

/* Now do checks for lengths. We do [0..16], [0..32], [0..64], [0..128]
	separately.  */
	cmp	$16, %ecx
	jbe	L(mm_len_0_16_bytes_forward)

	cmpl	$32, %ecx
	ja	L(mm_len_32_or_more_forward)

/* Copy [0..32] and return.  */
	movdqu	(%eax), %xmm0
	movdqu	-16(%eax, %ecx), %xmm1
	movdqu	%xmm0, (%edx)
	movdqu	%xmm1, -16(%edx, %ecx)
	jmp	L(return)

L(mm_len_32_or_more_forward):
	cmpl	$64, %ecx
	ja	L(mm_len_64_or_more_forward)

/* Copy [0..64] and return.  */
	movdqu	(%eax), %xmm0
	movdqu	16(%eax), %xmm1
	movdqu	-16(%eax, %ecx), %xmm2
	movdqu	-32(%eax, %ecx), %xmm3
	movdqu	%xmm0, (%edx)
	movdqu	%xmm1, 16(%edx)
	movdqu	%xmm2, -16(%edx, %ecx)
	movdqu	%xmm3, -32(%edx, %ecx)
	jmp	L(return)

L(mm_len_64_or_more_forward):
	cmpl	$128, %ecx
	ja	L(mm_len_128_or_more_forward)

/* Copy [0..128] and return.  */
	movdqu	(%eax), %xmm0
	movdqu	16(%eax), %xmm1
	movdqu	32(%eax), %xmm2
	movdqu	48(%eax), %xmm3
	movdqu	-64(%eax, %ecx), %xmm4
	movdqu	-48(%eax, %ecx), %xmm5
	movdqu	-32(%eax, %ecx), %xmm6
	movdqu	-16(%eax, %ecx), %xmm7
	movdqu	%xmm0, (%edx)
	movdqu	%xmm1, 16(%edx)
	movdqu	%xmm2, 32(%edx)
	movdqu	%xmm3, 48(%edx)
	movdqu	%xmm4, -64(%edx, %ecx)
	movdqu	%xmm5, -48(%edx, %ecx)
	movdqu	%xmm6, -32(%edx, %ecx)
	movdqu	%xmm7, -16(%edx, %ecx)
	jmp	L(return)

L(mm_len_128_or_more_forward):
	PUSH (%esi)
	PUSH (%edi)
	PUSH (%ebx)

/* Aligning the address of destination. */
	movdqu	-16(%eax, %ecx), %xmm4
	movdqu	-32(%eax, %ecx), %xmm5
	movdqu	-48(%eax, %ecx), %xmm6
	movdqu	-64(%eax, %ecx), %xmm7
	leal	(%edx, %ecx), %esi
	movdqu	(%eax), %xmm0
	subl	$16, %esp
	movdqu	%xmm0, (%esp)
	mov	%ecx, %edi
	leal	16(%edx), %ecx
	andl	$-16, %ecx
	movl	%ecx, %ebx
	subl	%edx, %ebx
	addl	%ebx, %eax
	movl	%esi, %ebx
	subl	%ecx, %ebx
	shrl	$6, %ebx

# ifdef SHARED_CACHE_SIZE_HALF
	cmp	$SHARED_CACHE_SIZE_HALF, %edi
# else
#  ifdef PIC
	PUSH (%ebx)
	SETUP_PIC_REG(bx)
	add	$_GLOBAL_OFFSET_TABLE_, %ebx
	cmp	__x86_shared_cache_size_half@GOTOFF(%ebx), %edi
	POP (%ebx)
#  else
	cmp	__x86_shared_cache_size_half, %edi
#  endif
# endif
	jae	L(mm_large_page_loop_forward)

	.p2align 4
L(mm_main_loop_forward):

	prefetcht0 128(%eax)

	movdqu	(%eax), %xmm0
	movdqu	16(%eax), %xmm1
	movdqu	32(%eax), %xmm2
	movdqu	48(%eax), %xmm3
	movdqa	%xmm0, (%ecx)
	addl	$64, %eax
	movaps	%xmm1, 16(%ecx)
	movaps	%xmm2, 32(%ecx)
	movaps	%xmm3, 48(%ecx)
	addl	$64, %ecx
	sub	$1, %ebx
	jnz	L(mm_main_loop_forward)
	movdqu	(%esp), %xmm0
	addl	$16, %esp
	movdqu	%xmm0, (%edx)
	movdqu	%xmm4, -16(%esi)
	movdqu	%xmm5, -32(%esi)
	movdqu	%xmm6, -48(%esi)
	movdqu	%xmm7, -64(%esi)
	POP (%ebx)
	jmp	L(mm_return_pop_all)

L(mm_len_0_16_bytes_forward):
	testb	$24, %cl
	jne	L(mm_len_9_16_bytes_forward)
	testb	$4, %cl
	.p2align 4,,5
	jne	L(mm_len_5_8_bytes_forward)
	testl	%ecx, %ecx
	.p2align 4,,2
	je	L(return)
	testb	$2, %cl
	.p2align 4,,1
	jne	L(mm_len_2_4_bytes_forward)
	movzbl	-1(%eax,%ecx), %ebx
	movzbl	(%eax), %eax
	movb	%bl, -1(%edx,%ecx)
	movb	%al, (%edx)
	jmp	L(return)

L(mm_len_2_4_bytes_forward):
	movzwl	-2(%eax,%ecx), %ebx
	movzwl	(%eax), %eax
	movw	%bx, -2(%edx,%ecx)
	movw	%ax, (%edx)
	jmp	L(return)

L(mm_len_5_8_bytes_forward):
	movl	(%eax), %ebx
	movl	-4(%eax,%ecx), %eax
	movl	%ebx, (%edx)
	movl	%eax, -4(%edx,%ecx)
	jmp	L(return)

L(mm_len_9_16_bytes_forward):
	movq	(%eax), %xmm0
	movq	-8(%eax, %ecx), %xmm1
	movq	%xmm0, (%edx)
	movq	%xmm1, -8(%edx, %ecx)
	jmp	L(return)

L(mm_return_pop_all):
	movl	%edx, %eax
	POP (%edi)
	POP (%esi)
	RETURN

/* Big length copy forward part.  */
	.p2align 4
L(mm_large_page_loop_forward):
	movdqu	(%eax), %xmm0
	movdqu	16(%eax), %xmm1
	movdqu	32(%eax), %xmm2
	movdqu	48(%eax), %xmm3
	movntdq	%xmm0, (%ecx)
	addl	$64, %eax
	movntdq	%xmm1, 16(%ecx)
	movntdq	%xmm2, 32(%ecx)
	movntdq	%xmm3, 48(%ecx)
	addl	$64, %ecx
	sub	$1, %ebx
	jnz	L(mm_large_page_loop_forward)
	sfence
	movdqu	(%esp), %xmm0
	addl	$16, %esp
	movdqu	%xmm0, (%edx)
	movdqu	%xmm4, -16(%esi)
	movdqu	%xmm5, -32(%esi)
	movdqu	%xmm6, -48(%esi)
	movdqu	%xmm7, -64(%esi)
	POP (%ebx)
	jmp	L(mm_return_pop_all)
# endif

L(forward):
	cmp	$16, %ecx
	jbe	L(len_0_16_bytes)

# ifdef SHARED_CACHE_SIZE_HALF
	cmp	$SHARED_CACHE_SIZE_HALF, %ecx
# else
#  ifdef PIC
	SETUP_PIC_REG(bx)
	add	$_GLOBAL_OFFSET_TABLE_, %ebx
	cmp	__x86_shared_cache_size_half@GOTOFF(%ebx), %ecx
#  else
	cmp	__x86_shared_cache_size_half, %ecx
#  endif
# endif
	jae     L(large_page)

	movdqu	(%eax), %xmm0
	movdqu	-16(%eax, %ecx), %xmm1
	cmpl    $32, %ecx
	movdqu	%xmm0, (%edx)
	movdqu	%xmm1, -16(%edx, %ecx)
	jbe	L(return)

	movdqu	16(%eax), %xmm0
	movdqu	-32(%eax, %ecx), %xmm1
	cmpl    $64, %ecx
	movdqu	%xmm0, 16(%edx)
	movdqu	%xmm1, -32(%edx, %ecx)
	jbe	L(return)

	movdqu	32(%eax), %xmm0
	movdqu	48(%eax), %xmm1
	movdqu	-48(%eax, %ecx), %xmm2
	movdqu	-64(%eax, %ecx), %xmm3
	cmpl    $128, %ecx
	movdqu	%xmm0, 32(%edx)
	movdqu	%xmm1, 48(%edx)
	movdqu	%xmm2, -48(%edx, %ecx)
	movdqu	%xmm3, -64(%edx, %ecx)
	jbe	L(return)

/* Now the main loop: we align the address of the destination.  */
	leal	64(%edx), %ebx
	andl	$-64, %ebx

	addl	%edx, %ecx
	andl	$-64, %ecx

	subl	%edx, %eax

/* We should stop two iterations before the termination
	(in order not to misprefetch).  */
	subl	$64, %ecx
	cmpl	%ebx, %ecx
	je	L(main_loop_just_one_iteration)

	subl	$64, %ecx
	cmpl	%ebx, %ecx
	je	L(main_loop_last_two_iterations)

	.p2align 4
L(main_loop_cache):

	prefetcht0 128(%ebx, %eax)

	movdqu	(%ebx, %eax), %xmm0
	movdqu	16(%ebx, %eax), %xmm1
	movdqu	32(%ebx, %eax), %xmm2
	movdqu	48(%ebx, %eax), %xmm3
	movdqa	%xmm0, (%ebx)
	movaps	%xmm1, 16(%ebx)
	movaps	%xmm2, 32(%ebx)
	movaps	%xmm3, 48(%ebx)
	lea	64(%ebx), %ebx
	cmpl	%ebx, %ecx
	jne	L(main_loop_cache)

L(main_loop_last_two_iterations):
	movdqu	(%ebx, %eax), %xmm0
	movdqu	16(%ebx, %eax), %xmm1
	movdqu	32(%ebx, %eax), %xmm2
	movdqu	48(%ebx, %eax), %xmm3
	movdqu	64(%ebx, %eax), %xmm4
	movdqu	80(%ebx, %eax), %xmm5
	movdqu	96(%ebx, %eax), %xmm6
	movdqu	112(%ebx, %eax), %xmm7
	movdqa	%xmm0, (%ebx)
	movaps	%xmm1, 16(%ebx)
	movaps	%xmm2, 32(%ebx)
	movaps	%xmm3, 48(%ebx)
	movaps	%xmm4, 64(%ebx)
	movaps	%xmm5, 80(%ebx)
	movaps	%xmm6, 96(%ebx)
	movaps	%xmm7, 112(%ebx)
	jmp	L(return)

L(main_loop_just_one_iteration):
	movdqu	(%ebx, %eax), %xmm0
	movdqu	16(%ebx, %eax), %xmm1
	movdqu	32(%ebx, %eax), %xmm2
	movdqu	48(%ebx, %eax), %xmm3
	movdqa	%xmm0, (%ebx)
	movaps	%xmm1, 16(%ebx)
	movaps	%xmm2, 32(%ebx)
	movaps	%xmm3, 48(%ebx)
	jmp	L(return)

L(large_page):
	movdqu	(%eax), %xmm0
	movdqu	16(%eax), %xmm1
	movdqu	32(%eax), %xmm2
	movdqu	48(%eax), %xmm3
	movdqu	-64(%eax, %ecx), %xmm4
	movdqu	-48(%eax, %ecx), %xmm5
	movdqu	-32(%eax, %ecx), %xmm6
	movdqu	-16(%eax, %ecx), %xmm7
	movdqu	%xmm0, (%edx)
	movdqu	%xmm1, 16(%edx)
	movdqu	%xmm2, 32(%edx)
	movdqu	%xmm3, 48(%edx)
	movdqu	%xmm4, -64(%edx, %ecx)
	movdqu	%xmm5, -48(%edx, %ecx)
	movdqu	%xmm6, -32(%edx, %ecx)
	movdqu	%xmm7, -16(%edx, %ecx)

	movdqu	64(%eax), %xmm0
	movdqu	80(%eax), %xmm1
	movdqu	96(%eax), %xmm2
	movdqu	112(%eax), %xmm3
	movdqu	-128(%eax, %ecx), %xmm4
	movdqu	-112(%eax, %ecx), %xmm5
	movdqu	-96(%eax, %ecx), %xmm6
	movdqu	-80(%eax, %ecx), %xmm7
	movdqu	%xmm0, 64(%edx)
	movdqu	%xmm1, 80(%edx)
	movdqu	%xmm2, 96(%edx)
	movdqu	%xmm3, 112(%edx)
	movdqu	%xmm4, -128(%edx, %ecx)
	movdqu	%xmm5, -112(%edx, %ecx)
	movdqu	%xmm6, -96(%edx, %ecx)
	movdqu	%xmm7, -80(%edx, %ecx)

/* Now the main loop with non temporal stores. We align
	the address of the destination.  */
	leal	128(%edx), %ebx
	andl	$-128, %ebx

	addl	%edx, %ecx
	andl	$-128, %ecx

	subl	%edx, %eax

	.p2align 4
L(main_loop_large_page):
	movdqu	(%ebx, %eax), %xmm0
	movdqu	16(%ebx, %eax), %xmm1
	movdqu	32(%ebx, %eax), %xmm2
	movdqu	48(%ebx, %eax), %xmm3
	movdqu	64(%ebx, %eax), %xmm4
	movdqu	80(%ebx, %eax), %xmm5
	movdqu	96(%ebx, %eax), %xmm6
	movdqu	112(%ebx, %eax), %xmm7
	movntdq	%xmm0, (%ebx)
	movntdq	%xmm1, 16(%ebx)
	movntdq	%xmm2, 32(%ebx)
	movntdq	%xmm3, 48(%ebx)
	movntdq	%xmm4, 64(%ebx)
	movntdq	%xmm5, 80(%ebx)
	movntdq	%xmm6, 96(%ebx)
	movntdq	%xmm7, 112(%ebx)
	lea	128(%ebx), %ebx
	cmpl	%ebx, %ecx
	jne	L(main_loop_large_page)
	sfence
	jmp	L(return)

L(len_0_16_bytes):
	testb	$24, %cl
	jne	L(len_9_16_bytes)
	testb	$4, %cl
	.p2align 4,,5
	jne	L(len_5_8_bytes)
	testl	%ecx, %ecx
	.p2align 4,,2
	je	L(return)
	movzbl	(%eax), %ebx
	testb	$2, %cl
	movb	%bl, (%edx)
	je	L(return)
	movzwl	-2(%eax,%ecx), %ebx
	movw	%bx, -2(%edx,%ecx)
	jmp	L(return)

L(len_9_16_bytes):
	movq	(%eax), %xmm0
	movq	-8(%eax, %ecx), %xmm1
	movq	%xmm0, (%edx)
	movq	%xmm1, -8(%edx, %ecx)
	jmp	L(return)

L(len_5_8_bytes):
	movl	(%eax), %ebx
	movl	%ebx, (%edx)
	movl	-4(%eax,%ecx), %ebx
	movl	%ebx, -4(%edx,%ecx)

L(return):
	movl	%edx, %eax
# if !defined USE_AS_BCOPY && defined USE_AS_MEMPCPY
	movl	LEN(%esp), %ecx
	add	%ecx, %eax
# endif
	RETURN

END (MEMCPY)
#endif
