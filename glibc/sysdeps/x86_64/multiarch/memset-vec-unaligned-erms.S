/* memset/bzero with unaligned store and rep stosb
   Copyright (C) 2016-2021 Free Software Foundation, Inc.
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

/* memset is implemented as:
   1. Use overlapping store to avoid branch.
   2. If size is less than VEC, use integer register stores.
   3. If size is from VEC_SIZE to 2 * VEC_SIZE, use 2 VEC stores.
   4. If size is from 2 * VEC_SIZE to 4 * VEC_SIZE, use 4 VEC stores.
   5. If size is more to 4 * VEC_SIZE, align to 4 * VEC_SIZE with
      4 VEC stores and store 4 * VEC at a time until done.  */

#include <sysdep.h>

#ifndef MEMSET_CHK_SYMBOL
# define MEMSET_CHK_SYMBOL(p,s)		MEMSET_SYMBOL(p, s)
#endif

#ifndef WMEMSET_CHK_SYMBOL
# define WMEMSET_CHK_SYMBOL(p,s)	WMEMSET_SYMBOL(p, s)
#endif

#ifndef XMM0
# define XMM0				xmm0
#endif

#ifndef YMM0
# define YMM0				ymm0
#endif

#ifndef VZEROUPPER
# if VEC_SIZE > 16
#  define VZEROUPPER			vzeroupper
#  define VZEROUPPER_SHORT_RETURN	vzeroupper; ret
# else
#  define VZEROUPPER
# endif
#endif

#ifndef VZEROUPPER_SHORT_RETURN
# define VZEROUPPER_SHORT_RETURN	rep; ret
#endif

#ifndef MOVQ
# if VEC_SIZE > 16
#  define MOVQ				vmovq
# else
#  define MOVQ				movq
# endif
#endif

#define PAGE_SIZE 4096

#ifndef SECTION
# error SECTION is not defined!
#endif

	.section SECTION(.text),"ax",@progbits
#if VEC_SIZE == 16 && IS_IN (libc)
ENTRY (__bzero)
	mov	%RDI_LP, %RAX_LP /* Set return value.  */
	mov	%RSI_LP, %RDX_LP /* Set n.  */
	pxor	%XMM0, %XMM0
	jmp	L(entry_from_bzero)
END (__bzero)
weak_alias (__bzero, bzero)
#endif

#if IS_IN (libc)
# if defined SHARED
ENTRY_CHK (WMEMSET_CHK_SYMBOL (__wmemset_chk, unaligned))
	cmp	%RDX_LP, %RCX_LP
	jb	HIDDEN_JUMPTARGET (__chk_fail)
END_CHK (WMEMSET_CHK_SYMBOL (__wmemset_chk, unaligned))
# endif

ENTRY (WMEMSET_SYMBOL (__wmemset, unaligned))
	shl	$2, %RDX_LP
	WMEMSET_VDUP_TO_VEC0_AND_SET_RETURN (%esi, %rdi)
	jmp	L(entry_from_bzero)
END (WMEMSET_SYMBOL (__wmemset, unaligned))
#endif

#if defined SHARED && IS_IN (libc)
ENTRY_CHK (MEMSET_CHK_SYMBOL (__memset_chk, unaligned))
	cmp	%RDX_LP, %RCX_LP
	jb	HIDDEN_JUMPTARGET (__chk_fail)
END_CHK (MEMSET_CHK_SYMBOL (__memset_chk, unaligned))
#endif

ENTRY (MEMSET_SYMBOL (__memset, unaligned))
	MEMSET_VDUP_TO_VEC0_AND_SET_RETURN (%esi, %rdi)
# ifdef __ILP32__
	/* Clear the upper 32 bits.  */
	mov	%edx, %edx
# endif
L(entry_from_bzero):
	cmpq	$VEC_SIZE, %rdx
	jb	L(less_vec)
	cmpq	$(VEC_SIZE * 2), %rdx
	ja	L(more_2x_vec)
	/* From VEC and to 2 * VEC.  No branch when size == VEC_SIZE.  */
	VMOVU	%VEC(0), -VEC_SIZE(%rdi,%rdx)
	VMOVU	%VEC(0), (%rdi)
	VZEROUPPER_RETURN
#if defined USE_MULTIARCH && IS_IN (libc)
END (MEMSET_SYMBOL (__memset, unaligned))

# if VEC_SIZE == 16
ENTRY (__memset_chk_erms)
	cmp	%RDX_LP, %RCX_LP
	jb	HIDDEN_JUMPTARGET (__chk_fail)
END (__memset_chk_erms)

/* Only used to measure performance of REP STOSB.  */
ENTRY (__memset_erms)
	/* Skip zero length.  */
	test	%RDX_LP, %RDX_LP
	jnz	 L(stosb)
	movq	%rdi, %rax
	ret
# else
/* Provide a hidden symbol to debugger.  */
	.hidden	MEMSET_SYMBOL (__memset, erms)
ENTRY (MEMSET_SYMBOL (__memset, erms))
# endif
L(stosb):
	mov	%RDX_LP, %RCX_LP
	movzbl	%sil, %eax
	mov	%RDI_LP, %RDX_LP
	rep stosb
	mov	%RDX_LP, %RAX_LP
	VZEROUPPER_RETURN
# if VEC_SIZE == 16
END (__memset_erms)
# else
END (MEMSET_SYMBOL (__memset, erms))
# endif

# if defined SHARED && IS_IN (libc)
ENTRY_CHK (MEMSET_CHK_SYMBOL (__memset_chk, unaligned_erms))
	cmp	%RDX_LP, %RCX_LP
	jb	HIDDEN_JUMPTARGET (__chk_fail)
END_CHK (MEMSET_CHK_SYMBOL (__memset_chk, unaligned_erms))
# endif

ENTRY (MEMSET_SYMBOL (__memset, unaligned_erms))
	MEMSET_VDUP_TO_VEC0_AND_SET_RETURN (%esi, %rdi)
# ifdef __ILP32__
	/* Clear the upper 32 bits.  */
	mov	%edx, %edx
# endif
	cmp	$VEC_SIZE, %RDX_LP
	jb	L(less_vec)
	cmp	$(VEC_SIZE * 2), %RDX_LP
	ja	L(stosb_more_2x_vec)
	/* From VEC and to 2 * VEC.  No branch when size == VEC_SIZE.  */
	VMOVU	%VEC(0), -VEC_SIZE(%rdi,%rdx)
	VMOVU	%VEC(0), (%rdi)
	VZEROUPPER_RETURN

	.p2align 4
L(stosb_more_2x_vec):
	cmp	__x86_rep_stosb_threshold(%rip), %RDX_LP
	ja	L(stosb)
#else
	.p2align 4
#endif
L(more_2x_vec):
	/* Stores to first 2x VEC before cmp as any path forward will
	   require it.  */
	VMOVU	%VEC(0), (%rdi)
	VMOVU	%VEC(0), VEC_SIZE(%rdi)
	cmpq	$(VEC_SIZE * 4), %rdx
	ja	L(loop_start)
	VMOVU	%VEC(0), -(VEC_SIZE * 2)(%rdi,%rdx)
	VMOVU	%VEC(0), -VEC_SIZE(%rdi,%rdx)
L(return):
#if VEC_SIZE > 16
	ZERO_UPPER_VEC_REGISTERS_RETURN
#else
	ret
#endif

L(loop_start):
	VMOVU	%VEC(0), (VEC_SIZE * 2)(%rdi)
	VMOVU	%VEC(0), (VEC_SIZE * 3)(%rdi)
	cmpq	$(VEC_SIZE * 8), %rdx
	jbe	L(loop_end)
	andq	$-(VEC_SIZE * 2), %rdi
	subq	$-(VEC_SIZE * 4), %rdi
	leaq	-(VEC_SIZE * 4)(%rax, %rdx), %rcx
	.p2align 4
L(loop):
	VMOVA	%VEC(0), (%rdi)
	VMOVA	%VEC(0), VEC_SIZE(%rdi)
	VMOVA	%VEC(0), (VEC_SIZE * 2)(%rdi)
	VMOVA	%VEC(0), (VEC_SIZE * 3)(%rdi)
	subq	$-(VEC_SIZE * 4), %rdi
	cmpq	%rcx, %rdi
	jb	L(loop)
L(loop_end):
	/* NB: rax is set as ptr in MEMSET_VDUP_TO_VEC0_AND_SET_RETURN.
	       rdx as length is also unchanged.  */
	VMOVU	%VEC(0), -(VEC_SIZE * 4)(%rax, %rdx)
	VMOVU	%VEC(0), -(VEC_SIZE * 3)(%rax, %rdx)
	VMOVU	%VEC(0), -(VEC_SIZE * 2)(%rax, %rdx)
	VMOVU	%VEC(0), -VEC_SIZE(%rax, %rdx)
	VZEROUPPER_SHORT_RETURN

	.p2align 4
L(less_vec):
	/* Less than 1 VEC.  */
# if VEC_SIZE != 16 && VEC_SIZE != 32 && VEC_SIZE != 64
#  error Unsupported VEC_SIZE!
# endif
# ifdef USE_LESS_VEC_MASK_STORE
	/* Clear high bits from edi. Only keeping bits relevant to page
	   cross check. Note that we are using rax which is set in
	   MEMSET_VDUP_TO_VEC0_AND_SET_RETURN as ptr from here on out.
	 */
	andl	$(PAGE_SIZE - 1), %edi
	/* Check if VEC_SIZE store cross page. Mask stores suffer serious
	   performance degradation when it has to fault supress.  */
	cmpl	$(PAGE_SIZE - VEC_SIZE), %edi
	ja	L(cross_page)
# if VEC_SIZE > 32
	movq	$-1, %rcx
	bzhiq	%rdx, %rcx, %rcx
	kmovq	%rcx, %k1
# else
	movl	$-1, %ecx
	bzhil	%edx, %ecx, %ecx
	kmovd	%ecx, %k1
# endif
	vmovdqu8	%VEC(0), (%rax) {%k1}
	VZEROUPPER_RETURN

	.p2align 4
L(cross_page):
# endif
# if VEC_SIZE > 32
	cmpb	$32, %dl
	jae	L(between_32_63)
# endif
# if VEC_SIZE > 16
	cmpb	$16, %dl
	jae	L(between_16_31)
# endif
	MOVQ	%XMM0, %rcx
	cmpb	$8, %dl
	jae	L(between_8_15)
	cmpb	$4, %dl
	jae	L(between_4_7)
	cmpb	$1, %dl
	ja	L(between_2_3)
	jb	1f
	movb	%cl, (%rax)
1:
	VZEROUPPER_RETURN
# if VEC_SIZE > 32
	/* From 32 to 63.  No branch when size == 32.  */
L(between_32_63):
	VMOVU	%YMM0, -32(%rax,%rdx)
	VMOVU	%YMM0, (%rax)
	VZEROUPPER_RETURN
# endif
# if VEC_SIZE > 16
	/* From 16 to 31.  No branch when size == 16.  */
L(between_16_31):
	VMOVU	%XMM0, -16(%rax,%rdx)
	VMOVU	%XMM0, (%rax)
	VZEROUPPER_RETURN
# endif
	/* From 8 to 15.  No branch when size == 8.  */
L(between_8_15):
	movq	%rcx, -8(%rax,%rdx)
	movq	%rcx, (%rax)
	VZEROUPPER_RETURN
L(between_4_7):
	/* From 4 to 7.  No branch when size == 4.  */
	movl	%ecx, -4(%rax,%rdx)
	movl	%ecx, (%rax)
	VZEROUPPER_RETURN
L(between_2_3):
	/* From 2 to 3.  No branch when size == 2.  */
	movw	%cx, -2(%rax,%rdx)
	movw	%cx, (%rax)
	VZEROUPPER_RETURN
END (MEMSET_SYMBOL (__memset, unaligned_erms))
