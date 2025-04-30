/* memmove/memcpy/mempcpy with unaligned load/store and rep movsb
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

/* memmove/memcpy/mempcpy is implemented as:
   1. Use overlapping load and store to avoid branch.
   2. Load all sources into registers and store them together to avoid
      possible address overlap between source and destination.
   3. If size is 8 * VEC_SIZE or less, load all sources into registers
      and store them together.
   4. If address of destination > address of source, backward copy
      4 * VEC_SIZE at a time with unaligned load and aligned store.
      Load the first 4 * VEC and last VEC before the loop and store
      them after the loop to support overlapping addresses.
   5. Otherwise, forward copy 4 * VEC_SIZE at a time with unaligned
      load and aligned store.  Load the last 4 * VEC and first VEC
      before the loop and store them after the loop to support
      overlapping addresses.
   6. On machines with ERMS feature, if size greater than equal or to
      __x86_rep_movsb_threshold and less than
      __x86_rep_movsb_stop_threshold, then REP MOVSB will be used.
   7. If size >= __x86_shared_non_temporal_threshold and there is no
      overlap between destination and source, use non-temporal store
      instead of aligned store copying from either 2 or 4 pages at
      once.
   8. For point 7) if size < 16 * __x86_shared_non_temporal_threshold
      and source and destination do not page alias, copy from 2 pages
      at once using non-temporal stores. Page aliasing in this case is
      considered true if destination's page alignment - sources' page
      alignment is less than 8 * VEC_SIZE.
   9. If size >= 16 * __x86_shared_non_temporal_threshold or source
      and destination do page alias copy from 4 pages at once using
      non-temporal stores.  */

#include <sysdep.h>

#ifndef MEMCPY_SYMBOL
# define MEMCPY_SYMBOL(p,s)		MEMMOVE_SYMBOL(p, s)
#endif

#ifndef MEMPCPY_SYMBOL
# define MEMPCPY_SYMBOL(p,s)		MEMMOVE_SYMBOL(p, s)
#endif

#ifndef MEMMOVE_CHK_SYMBOL
# define MEMMOVE_CHK_SYMBOL(p,s)	MEMMOVE_SYMBOL(p, s)
#endif

#ifndef XMM0
# define XMM0				xmm0
#endif

#ifndef YMM0
# define YMM0				ymm0
#endif

#ifndef VZEROUPPER
# if VEC_SIZE > 16
#  define VZEROUPPER vzeroupper
# else
#  define VZEROUPPER
# endif
#endif

#ifndef PAGE_SIZE
# define PAGE_SIZE 4096
#endif

#if PAGE_SIZE != 4096
# error Unsupported PAGE_SIZE
#endif

#ifndef LOG_PAGE_SIZE
# define LOG_PAGE_SIZE 12
#endif

#if PAGE_SIZE != (1 << LOG_PAGE_SIZE)
# error Invalid LOG_PAGE_SIZE
#endif

/* Byte per page for large_memcpy inner loop.  */
#if VEC_SIZE == 64
# define LARGE_LOAD_SIZE (VEC_SIZE * 2)
#else
# define LARGE_LOAD_SIZE (VEC_SIZE * 4)
#endif

/* Amount to shift rdx by to compare for memcpy_large_4x.  */
#ifndef LOG_4X_MEMCPY_THRESH
# define LOG_4X_MEMCPY_THRESH 4
#endif

/* Avoid short distance rep movsb only with non-SSE vector.  */
#ifndef AVOID_SHORT_DISTANCE_REP_MOVSB
# define AVOID_SHORT_DISTANCE_REP_MOVSB (VEC_SIZE > 16)
#else
# define AVOID_SHORT_DISTANCE_REP_MOVSB 0
#endif

#ifndef PREFETCH
# define PREFETCH(addr) prefetcht0 addr
#endif

/* Assume 64-byte prefetch size.  */
#ifndef PREFETCH_SIZE
# define PREFETCH_SIZE 64
#endif

#define PREFETCHED_LOAD_SIZE (VEC_SIZE * 4)

#if PREFETCH_SIZE == 64
# if PREFETCHED_LOAD_SIZE == PREFETCH_SIZE
#  define PREFETCH_ONE_SET(dir, base, offset) \
	PREFETCH ((offset)base)
# elif PREFETCHED_LOAD_SIZE == 2 * PREFETCH_SIZE
#  define PREFETCH_ONE_SET(dir, base, offset) \
	PREFETCH ((offset)base); \
	PREFETCH ((offset + dir * PREFETCH_SIZE)base)
# elif PREFETCHED_LOAD_SIZE == 4 * PREFETCH_SIZE
#  define PREFETCH_ONE_SET(dir, base, offset) \
	PREFETCH ((offset)base); \
	PREFETCH ((offset + dir * PREFETCH_SIZE)base); \
	PREFETCH ((offset + dir * PREFETCH_SIZE * 2)base); \
	PREFETCH ((offset + dir * PREFETCH_SIZE * 3)base)
# else
#   error Unsupported PREFETCHED_LOAD_SIZE!
# endif
#else
# error Unsupported PREFETCH_SIZE!
#endif

#if LARGE_LOAD_SIZE == (VEC_SIZE * 2)
# define LOAD_ONE_SET(base, offset, vec0, vec1, ...) \
	VMOVU	(offset)base, vec0; \
	VMOVU	((offset) + VEC_SIZE)base, vec1;
# define STORE_ONE_SET(base, offset, vec0, vec1, ...) \
	VMOVNT  vec0, (offset)base; \
	VMOVNT  vec1, ((offset) + VEC_SIZE)base;
#elif LARGE_LOAD_SIZE == (VEC_SIZE * 4)
# define LOAD_ONE_SET(base, offset, vec0, vec1, vec2, vec3) \
	VMOVU	(offset)base, vec0; \
	VMOVU	((offset) + VEC_SIZE)base, vec1; \
	VMOVU	((offset) + VEC_SIZE * 2)base, vec2; \
	VMOVU	((offset) + VEC_SIZE * 3)base, vec3;
# define STORE_ONE_SET(base, offset, vec0, vec1, vec2, vec3) \
	VMOVNT	vec0, (offset)base; \
	VMOVNT	vec1, ((offset) + VEC_SIZE)base; \
	VMOVNT	vec2, ((offset) + VEC_SIZE * 2)base; \
	VMOVNT	vec3, ((offset) + VEC_SIZE * 3)base;
#else
# error Invalid LARGE_LOAD_SIZE
#endif

#ifndef SECTION
# error SECTION is not defined!
#endif

	.section SECTION(.text),"ax",@progbits
#if defined SHARED && IS_IN (libc)
ENTRY (MEMMOVE_CHK_SYMBOL (__mempcpy_chk, unaligned))
	cmp	%RDX_LP, %RCX_LP
	jb	HIDDEN_JUMPTARGET (__chk_fail)
END (MEMMOVE_CHK_SYMBOL (__mempcpy_chk, unaligned))
#endif

ENTRY (MEMPCPY_SYMBOL (__mempcpy, unaligned))
	mov	%RDI_LP, %RAX_LP
	add	%RDX_LP, %RAX_LP
	jmp	L(start)
END (MEMPCPY_SYMBOL (__mempcpy, unaligned))

#if defined SHARED && IS_IN (libc)
ENTRY (MEMMOVE_CHK_SYMBOL (__memmove_chk, unaligned))
	cmp	%RDX_LP, %RCX_LP
	jb	HIDDEN_JUMPTARGET (__chk_fail)
END (MEMMOVE_CHK_SYMBOL (__memmove_chk, unaligned))
#endif

ENTRY (MEMMOVE_SYMBOL (__memmove, unaligned))
	movq	%rdi, %rax
L(start):
# ifdef __ILP32__
	/* Clear the upper 32 bits.  */
	movl	%edx, %edx
# endif
	cmp	$VEC_SIZE, %RDX_LP
	jb	L(less_vec)
	cmp	$(VEC_SIZE * 2), %RDX_LP
	ja	L(more_2x_vec)
#if !defined USE_MULTIARCH || !IS_IN (libc)
L(last_2x_vec):
#endif
	/* From VEC and to 2 * VEC.  No branch when size == VEC_SIZE.  */
	VMOVU	(%rsi), %VEC(0)
	VMOVU	-VEC_SIZE(%rsi,%rdx), %VEC(1)
	VMOVU	%VEC(0), (%rdi)
	VMOVU	%VEC(1), -VEC_SIZE(%rdi,%rdx)
#if !defined USE_MULTIARCH || !IS_IN (libc)
L(nop):
	ret
#else
	VZEROUPPER_RETURN
#endif
#if defined USE_MULTIARCH && IS_IN (libc)
END (MEMMOVE_SYMBOL (__memmove, unaligned))

# if VEC_SIZE == 16
ENTRY (__mempcpy_chk_erms)
	cmp	%RDX_LP, %RCX_LP
	jb	HIDDEN_JUMPTARGET (__chk_fail)
END (__mempcpy_chk_erms)

/* Only used to measure performance of REP MOVSB.  */
ENTRY (__mempcpy_erms)
	mov	%RDI_LP, %RAX_LP
	/* Skip zero length.  */
	test	%RDX_LP, %RDX_LP
	jz	2f
	add	%RDX_LP, %RAX_LP
	jmp	L(start_movsb)
END (__mempcpy_erms)

ENTRY (__memmove_chk_erms)
	cmp	%RDX_LP, %RCX_LP
	jb	HIDDEN_JUMPTARGET (__chk_fail)
END (__memmove_chk_erms)

ENTRY (__memmove_erms)
	movq	%rdi, %rax
	/* Skip zero length.  */
	test	%RDX_LP, %RDX_LP
	jz	2f
L(start_movsb):
	mov	%RDX_LP, %RCX_LP
	cmp	%RSI_LP, %RDI_LP
	jb	1f
	/* Source == destination is less common.  */
	je	2f
	lea	(%rsi,%rcx), %RDX_LP
	cmp	%RDX_LP, %RDI_LP
	jb	L(movsb_backward)
1:
	rep movsb
2:
	ret
L(movsb_backward):
	leaq	-1(%rdi,%rcx), %rdi
	leaq	-1(%rsi,%rcx), %rsi
	std
	rep movsb
	cld
	ret
END (__memmove_erms)
strong_alias (__memmove_erms, __memcpy_erms)
strong_alias (__memmove_chk_erms, __memcpy_chk_erms)
# endif

# ifdef SHARED
ENTRY (MEMMOVE_CHK_SYMBOL (__mempcpy_chk, unaligned_erms))
	cmp	%RDX_LP, %RCX_LP
	jb	HIDDEN_JUMPTARGET (__chk_fail)
END (MEMMOVE_CHK_SYMBOL (__mempcpy_chk, unaligned_erms))
# endif

ENTRY (MEMMOVE_SYMBOL (__mempcpy, unaligned_erms))
	mov	%RDI_LP, %RAX_LP
	add	%RDX_LP, %RAX_LP
	jmp	L(start_erms)
END (MEMMOVE_SYMBOL (__mempcpy, unaligned_erms))

# ifdef SHARED
ENTRY (MEMMOVE_CHK_SYMBOL (__memmove_chk, unaligned_erms))
	cmp	%RDX_LP, %RCX_LP
	jb	HIDDEN_JUMPTARGET (__chk_fail)
END (MEMMOVE_CHK_SYMBOL (__memmove_chk, unaligned_erms))
# endif

ENTRY (MEMMOVE_SYMBOL (__memmove, unaligned_erms))
	movq	%rdi, %rax
L(start_erms):
# ifdef __ILP32__
	/* Clear the upper 32 bits.  */
	movl	%edx, %edx
# endif
	cmp	$VEC_SIZE, %RDX_LP
	jb	L(less_vec)
	cmp	$(VEC_SIZE * 2), %RDX_LP
	ja	L(movsb_more_2x_vec)
L(last_2x_vec):
	/* From VEC and to 2 * VEC.  No branch when size == VEC_SIZE. */
	VMOVU	(%rsi), %VEC(0)
	VMOVU	-VEC_SIZE(%rsi,%rdx), %VEC(1)
	VMOVU	%VEC(0), (%rdi)
	VMOVU	%VEC(1), -VEC_SIZE(%rdi,%rdx)
L(return):
#if VEC_SIZE > 16
	ZERO_UPPER_VEC_REGISTERS_RETURN
#else
	ret
#endif

L(movsb):
	cmp     __x86_rep_movsb_stop_threshold(%rip), %RDX_LP
	jae	L(more_8x_vec)
	cmpq	%rsi, %rdi
	jb	1f
	/* Source == destination is less common.  */
	je	L(nop)
	leaq	(%rsi,%rdx), %r9
	cmpq	%r9, %rdi
	/* Avoid slow backward REP MOVSB.  */
	jb	L(more_8x_vec_backward)
# if AVOID_SHORT_DISTANCE_REP_MOVSB
	andl	$X86_STRING_CONTROL_AVOID_SHORT_DISTANCE_REP_MOVSB, __x86_string_control(%rip)
	jz	3f
	movq	%rdi, %rcx
	subq	%rsi, %rcx
	jmp	2f
# endif
1:
# if AVOID_SHORT_DISTANCE_REP_MOVSB
	andl	$X86_STRING_CONTROL_AVOID_SHORT_DISTANCE_REP_MOVSB, __x86_string_control(%rip)
	jz	3f
	movq	%rsi, %rcx
	subq	%rdi, %rcx
2:
/* Avoid "rep movsb" if RCX, the distance between source and destination,
   is N*4GB + [1..63] with N >= 0.  */
	cmpl	$63, %ecx
	jbe	L(more_2x_vec)	/* Avoid "rep movsb" if ECX <= 63.  */
3:
# endif
	mov	%RDX_LP, %RCX_LP
	rep movsb
L(nop):
	ret
#endif

L(less_vec):
	/* Less than 1 VEC.  */
#if VEC_SIZE != 16 && VEC_SIZE != 32 && VEC_SIZE != 64
# error Unsupported VEC_SIZE!
#endif
#if VEC_SIZE > 32
	cmpb	$32, %dl
	jae	L(between_32_63)
#endif
#if VEC_SIZE > 16
	cmpb	$16, %dl
	jae	L(between_16_31)
#endif
	cmpb	$8, %dl
	jae	L(between_8_15)
	cmpb	$4, %dl
	jae	L(between_4_7)
	cmpb	$1, %dl
	ja	L(between_2_3)
	jb	1f
	movzbl	(%rsi), %ecx
	movb	%cl, (%rdi)
1:
	ret
#if VEC_SIZE > 32
L(between_32_63):
	/* From 32 to 63.  No branch when size == 32.  */
	VMOVU	(%rsi), %YMM0
	VMOVU	-32(%rsi,%rdx), %YMM1
	VMOVU	%YMM0, (%rdi)
	VMOVU	%YMM1, -32(%rdi,%rdx)
	VZEROUPPER_RETURN
#endif
#if VEC_SIZE > 16
	/* From 16 to 31.  No branch when size == 16.  */
L(between_16_31):
	VMOVU	(%rsi), %XMM0
	VMOVU	-16(%rsi,%rdx), %XMM1
	VMOVU	%XMM0, (%rdi)
	VMOVU	%XMM1, -16(%rdi,%rdx)
	VZEROUPPER_RETURN
#endif
L(between_8_15):
	/* From 8 to 15.  No branch when size == 8.  */
	movq	-8(%rsi,%rdx), %rcx
	movq	(%rsi), %rsi
	movq	%rcx, -8(%rdi,%rdx)
	movq	%rsi, (%rdi)
	ret
L(between_4_7):
	/* From 4 to 7.  No branch when size == 4.  */
	movl	-4(%rsi,%rdx), %ecx
	movl	(%rsi), %esi
	movl	%ecx, -4(%rdi,%rdx)
	movl	%esi, (%rdi)
	ret
L(between_2_3):
	/* From 2 to 3.  No branch when size == 2.  */
	movzwl	-2(%rsi,%rdx), %ecx
	movzwl	(%rsi), %esi
	movw	%cx, -2(%rdi,%rdx)
	movw	%si, (%rdi)
	ret

#if defined USE_MULTIARCH && IS_IN (libc)
L(movsb_more_2x_vec):
	cmp	__x86_rep_movsb_threshold(%rip), %RDX_LP
	ja	L(movsb)
#endif
L(more_2x_vec):
	/* More than 2 * VEC and there may be overlap between destination
	   and source.  */
	cmpq	$(VEC_SIZE * 8), %rdx
	ja	L(more_8x_vec)
	cmpq	$(VEC_SIZE * 4), %rdx
	jbe	L(last_4x_vec)
	/* Copy from 4 * VEC + 1 to 8 * VEC, inclusively. */
	VMOVU	(%rsi), %VEC(0)
	VMOVU	VEC_SIZE(%rsi), %VEC(1)
	VMOVU	(VEC_SIZE * 2)(%rsi), %VEC(2)
	VMOVU	(VEC_SIZE * 3)(%rsi), %VEC(3)
	VMOVU	-VEC_SIZE(%rsi,%rdx), %VEC(4)
	VMOVU	-(VEC_SIZE * 2)(%rsi,%rdx), %VEC(5)
	VMOVU	-(VEC_SIZE * 3)(%rsi,%rdx), %VEC(6)
	VMOVU	-(VEC_SIZE * 4)(%rsi,%rdx), %VEC(7)
	VMOVU	%VEC(0), (%rdi)
	VMOVU	%VEC(1), VEC_SIZE(%rdi)
	VMOVU	%VEC(2), (VEC_SIZE * 2)(%rdi)
	VMOVU	%VEC(3), (VEC_SIZE * 3)(%rdi)
	VMOVU	%VEC(4), -VEC_SIZE(%rdi,%rdx)
	VMOVU	%VEC(5), -(VEC_SIZE * 2)(%rdi,%rdx)
	VMOVU	%VEC(6), -(VEC_SIZE * 3)(%rdi,%rdx)
	VMOVU	%VEC(7), -(VEC_SIZE * 4)(%rdi,%rdx)
	VZEROUPPER_RETURN
L(last_4x_vec):
	/* Copy from 2 * VEC + 1 to 4 * VEC, inclusively. */
	VMOVU	(%rsi), %VEC(0)
	VMOVU	VEC_SIZE(%rsi), %VEC(1)
	VMOVU	-VEC_SIZE(%rsi,%rdx), %VEC(2)
	VMOVU	-(VEC_SIZE * 2)(%rsi,%rdx), %VEC(3)
	VMOVU	%VEC(0), (%rdi)
	VMOVU	%VEC(1), VEC_SIZE(%rdi)
	VMOVU	%VEC(2), -VEC_SIZE(%rdi,%rdx)
	VMOVU	%VEC(3), -(VEC_SIZE * 2)(%rdi,%rdx)
	VZEROUPPER_RETURN

L(more_8x_vec):
	/* Check if non-temporal move candidate.  */
#if (defined USE_MULTIARCH || VEC_SIZE == 16) && IS_IN (libc)
	/* Check non-temporal store threshold.  */
	cmp __x86_shared_non_temporal_threshold(%rip), %RDX_LP
	ja	L(large_memcpy_2x)
#endif
	/* Entry if rdx is greater than non-temporal threshold but there
       is overlap.  */
L(more_8x_vec_check):
	cmpq	%rsi, %rdi
	ja	L(more_8x_vec_backward)
	/* Source == destination is less common.  */
	je	L(nop)
	/* Load the first VEC and last 4 * VEC to support overlapping
	   addresses.  */
	VMOVU	(%rsi), %VEC(4)
	VMOVU	-VEC_SIZE(%rsi, %rdx), %VEC(5)
	VMOVU	-(VEC_SIZE * 2)(%rsi, %rdx), %VEC(6)
	VMOVU	-(VEC_SIZE * 3)(%rsi, %rdx), %VEC(7)
	VMOVU	-(VEC_SIZE * 4)(%rsi, %rdx), %VEC(8)
	/* Save start and stop of the destination buffer.  */
	movq	%rdi, %r11
	leaq	-VEC_SIZE(%rdi, %rdx), %rcx
	/* Align destination for aligned stores in the loop.  Compute
	   how much destination is misaligned.  */
	movq	%rdi, %r8
	andq	$(VEC_SIZE - 1), %r8
	/* Get the negative of offset for alignment.  */
	subq	$VEC_SIZE, %r8
	/* Adjust source.  */
	subq	%r8, %rsi
	/* Adjust destination which should be aligned now.  */
	subq	%r8, %rdi
	/* Adjust length.  */
	addq	%r8, %rdx

	.p2align 4
L(loop_4x_vec_forward):
	/* Copy 4 * VEC a time forward.  */
	VMOVU	(%rsi), %VEC(0)
	VMOVU	VEC_SIZE(%rsi), %VEC(1)
	VMOVU	(VEC_SIZE * 2)(%rsi), %VEC(2)
	VMOVU	(VEC_SIZE * 3)(%rsi), %VEC(3)
	subq	$-(VEC_SIZE * 4), %rsi
	addq	$-(VEC_SIZE * 4), %rdx
	VMOVA	%VEC(0), (%rdi)
	VMOVA	%VEC(1), VEC_SIZE(%rdi)
	VMOVA	%VEC(2), (VEC_SIZE * 2)(%rdi)
	VMOVA	%VEC(3), (VEC_SIZE * 3)(%rdi)
	subq	$-(VEC_SIZE * 4), %rdi
	cmpq	$(VEC_SIZE * 4), %rdx
	ja	L(loop_4x_vec_forward)
	/* Store the last 4 * VEC.  */
	VMOVU	%VEC(5), (%rcx)
	VMOVU	%VEC(6), -VEC_SIZE(%rcx)
	VMOVU	%VEC(7), -(VEC_SIZE * 2)(%rcx)
	VMOVU	%VEC(8), -(VEC_SIZE * 3)(%rcx)
	/* Store the first VEC.  */
	VMOVU	%VEC(4), (%r11)
	VZEROUPPER_RETURN

L(more_8x_vec_backward):
	/* Load the first 4 * VEC and last VEC to support overlapping
	   addresses.  */
	VMOVU	(%rsi), %VEC(4)
	VMOVU	VEC_SIZE(%rsi), %VEC(5)
	VMOVU	(VEC_SIZE * 2)(%rsi), %VEC(6)
	VMOVU	(VEC_SIZE * 3)(%rsi), %VEC(7)
	VMOVU	-VEC_SIZE(%rsi,%rdx), %VEC(8)
	/* Save stop of the destination buffer.  */
	leaq	-VEC_SIZE(%rdi, %rdx), %r11
	/* Align destination end for aligned stores in the loop.  Compute
	   how much destination end is misaligned.  */
	leaq	-VEC_SIZE(%rsi, %rdx), %rcx
	movq	%r11, %r9
	movq	%r11, %r8
	andq	$(VEC_SIZE - 1), %r8
	/* Adjust source.  */
	subq	%r8, %rcx
	/* Adjust the end of destination which should be aligned now.  */
	subq	%r8, %r9
	/* Adjust length.  */
	subq	%r8, %rdx

	.p2align 4
L(loop_4x_vec_backward):
	/* Copy 4 * VEC a time backward.  */
	VMOVU	(%rcx), %VEC(0)
	VMOVU	-VEC_SIZE(%rcx), %VEC(1)
	VMOVU	-(VEC_SIZE * 2)(%rcx), %VEC(2)
	VMOVU	-(VEC_SIZE * 3)(%rcx), %VEC(3)
	addq	$-(VEC_SIZE * 4), %rcx
	addq	$-(VEC_SIZE * 4), %rdx
	VMOVA	%VEC(0), (%r9)
	VMOVA	%VEC(1), -VEC_SIZE(%r9)
	VMOVA	%VEC(2), -(VEC_SIZE * 2)(%r9)
	VMOVA	%VEC(3), -(VEC_SIZE * 3)(%r9)
	addq	$-(VEC_SIZE * 4), %r9
	cmpq	$(VEC_SIZE * 4), %rdx
	ja	L(loop_4x_vec_backward)
	/* Store the first 4 * VEC.  */
	VMOVU	%VEC(4), (%rdi)
	VMOVU	%VEC(5), VEC_SIZE(%rdi)
	VMOVU	%VEC(6), (VEC_SIZE * 2)(%rdi)
	VMOVU	%VEC(7), (VEC_SIZE * 3)(%rdi)
	/* Store the last VEC.  */
	VMOVU	%VEC(8), (%r11)
	VZEROUPPER_RETURN

#if (defined USE_MULTIARCH || VEC_SIZE == 16) && IS_IN (libc)
	.p2align 4
L(large_memcpy_2x):
	/* Compute absolute value of difference between source and
	   destination.  */
	movq	%rdi, %r9
	subq	%rsi, %r9
	movq	%r9, %r8
	leaq	-1(%r9), %rcx
	sarq	$63, %r8
	xorq	%r8, %r9
	subq	%r8, %r9
	/* Don't use non-temporal store if there is overlap between
	   destination and source since destination may be in cache when
	   source is loaded.  */
	cmpq	%r9, %rdx
	ja	L(more_8x_vec_check)

	/* Cache align destination. First store the first 64 bytes then
	   adjust alignments.  */
	VMOVU	(%rsi), %VEC(8)
#if VEC_SIZE < 64
	VMOVU	VEC_SIZE(%rsi), %VEC(9)
#if VEC_SIZE < 32
	VMOVU	(VEC_SIZE * 2)(%rsi), %VEC(10)
	VMOVU	(VEC_SIZE * 3)(%rsi), %VEC(11)
#endif
#endif
	VMOVU	%VEC(8), (%rdi)
#if VEC_SIZE < 64
	VMOVU	%VEC(9), VEC_SIZE(%rdi)
#if VEC_SIZE < 32
	VMOVU	%VEC(10), (VEC_SIZE * 2)(%rdi)
	VMOVU	%VEC(11), (VEC_SIZE * 3)(%rdi)
#endif
#endif
	/* Adjust source, destination, and size.  */
	movq	%rdi, %r8
	andq	$63, %r8
	/* Get the negative of offset for alignment.  */
	subq	$64, %r8
	/* Adjust source.  */
	subq	%r8, %rsi
	/* Adjust destination which should be aligned now.  */
	subq	%r8, %rdi
	/* Adjust length.  */
	addq	%r8, %rdx

	/* Test if source and destination addresses will alias. If they do
	   the larger pipeline in large_memcpy_4x alleviated the
	   performance drop.  */
	testl	$(PAGE_SIZE - VEC_SIZE * 8), %ecx
	jz	L(large_memcpy_4x)

	movq	%rdx, %r10
	shrq	$LOG_4X_MEMCPY_THRESH, %r10
	cmp	__x86_shared_non_temporal_threshold(%rip), %r10
	jae	L(large_memcpy_4x)

	/* edx will store remainder size for copying tail.  */
	andl	$(PAGE_SIZE * 2 - 1), %edx
	/* r10 stores outer loop counter.  */
	shrq	$((LOG_PAGE_SIZE + 1) - LOG_4X_MEMCPY_THRESH), %r10
	/* Copy 4x VEC at a time from 2 pages.  */
	.p2align 4
L(loop_large_memcpy_2x_outer):
	/* ecx stores inner loop counter.  */
	movl	$(PAGE_SIZE / LARGE_LOAD_SIZE), %ecx
L(loop_large_memcpy_2x_inner):
	PREFETCH_ONE_SET(1, (%rsi), PREFETCHED_LOAD_SIZE)
	PREFETCH_ONE_SET(1, (%rsi), PREFETCHED_LOAD_SIZE * 2)
	PREFETCH_ONE_SET(1, (%rsi), PAGE_SIZE + PREFETCHED_LOAD_SIZE)
	PREFETCH_ONE_SET(1, (%rsi), PAGE_SIZE + PREFETCHED_LOAD_SIZE * 2)
	/* Load vectors from rsi.  */
	LOAD_ONE_SET((%rsi), 0, %VEC(0), %VEC(1), %VEC(2), %VEC(3))
	LOAD_ONE_SET((%rsi), PAGE_SIZE, %VEC(4), %VEC(5), %VEC(6), %VEC(7))
	subq	$-LARGE_LOAD_SIZE, %rsi
	/* Non-temporal store vectors to rdi.  */
	STORE_ONE_SET((%rdi), 0, %VEC(0), %VEC(1), %VEC(2), %VEC(3))
	STORE_ONE_SET((%rdi), PAGE_SIZE, %VEC(4), %VEC(5), %VEC(6), %VEC(7))
	subq	$-LARGE_LOAD_SIZE, %rdi
	decl	%ecx
	jnz	L(loop_large_memcpy_2x_inner)
	addq	$PAGE_SIZE, %rdi
	addq	$PAGE_SIZE, %rsi
	decq	%r10
	jne	L(loop_large_memcpy_2x_outer)
	sfence

	/* Check if only last 4 loads are needed.  */
	cmpl	$(VEC_SIZE * 4), %edx
	jbe	L(large_memcpy_2x_end)

	/* Handle the last 2 * PAGE_SIZE bytes.  */
L(loop_large_memcpy_2x_tail):
	/* Copy 4 * VEC a time forward with non-temporal stores.  */
	PREFETCH_ONE_SET (1, (%rsi), PREFETCHED_LOAD_SIZE)
	PREFETCH_ONE_SET (1, (%rdi), PREFETCHED_LOAD_SIZE)
	VMOVU	(%rsi), %VEC(0)
	VMOVU	VEC_SIZE(%rsi), %VEC(1)
	VMOVU	(VEC_SIZE * 2)(%rsi), %VEC(2)
	VMOVU	(VEC_SIZE * 3)(%rsi), %VEC(3)
	subq	$-(VEC_SIZE * 4), %rsi
	addl	$-(VEC_SIZE * 4), %edx
	VMOVA	%VEC(0), (%rdi)
	VMOVA	%VEC(1), VEC_SIZE(%rdi)
	VMOVA	%VEC(2), (VEC_SIZE * 2)(%rdi)
	VMOVA	%VEC(3), (VEC_SIZE * 3)(%rdi)
	subq	$-(VEC_SIZE * 4), %rdi
	cmpl	$(VEC_SIZE * 4), %edx
	ja	L(loop_large_memcpy_2x_tail)

L(large_memcpy_2x_end):
	/* Store the last 4 * VEC.  */
	VMOVU	-(VEC_SIZE * 4)(%rsi, %rdx), %VEC(0)
	VMOVU	-(VEC_SIZE * 3)(%rsi, %rdx), %VEC(1)
	VMOVU	-(VEC_SIZE * 2)(%rsi, %rdx), %VEC(2)
	VMOVU	-VEC_SIZE(%rsi, %rdx), %VEC(3)

	VMOVU	%VEC(0), -(VEC_SIZE * 4)(%rdi, %rdx)
	VMOVU	%VEC(1), -(VEC_SIZE * 3)(%rdi, %rdx)
	VMOVU	%VEC(2), -(VEC_SIZE * 2)(%rdi, %rdx)
	VMOVU	%VEC(3), -VEC_SIZE(%rdi, %rdx)
	VZEROUPPER_RETURN

	.p2align 4
L(large_memcpy_4x):
	movq	%rdx, %r10
	/* edx will store remainder size for copying tail.  */
	andl	$(PAGE_SIZE * 4 - 1), %edx
	/* r10 stores outer loop counter.  */
	shrq	$(LOG_PAGE_SIZE + 2), %r10
	/* Copy 4x VEC at a time from 4 pages.  */
	.p2align 4
L(loop_large_memcpy_4x_outer):
	/* ecx stores inner loop counter.  */
	movl	$(PAGE_SIZE / LARGE_LOAD_SIZE), %ecx
L(loop_large_memcpy_4x_inner):
	/* Only one prefetch set per page as doing 4 pages give more time
	   for prefetcher to keep up.  */
	PREFETCH_ONE_SET(1, (%rsi), PREFETCHED_LOAD_SIZE)
	PREFETCH_ONE_SET(1, (%rsi), PAGE_SIZE + PREFETCHED_LOAD_SIZE)
	PREFETCH_ONE_SET(1, (%rsi), PAGE_SIZE * 2 + PREFETCHED_LOAD_SIZE)
	PREFETCH_ONE_SET(1, (%rsi), PAGE_SIZE * 3 + PREFETCHED_LOAD_SIZE)
	/* Load vectors from rsi.  */
	LOAD_ONE_SET((%rsi), 0, %VEC(0), %VEC(1), %VEC(2), %VEC(3))
	LOAD_ONE_SET((%rsi), PAGE_SIZE, %VEC(4), %VEC(5), %VEC(6), %VEC(7))
	LOAD_ONE_SET((%rsi), PAGE_SIZE * 2, %VEC(8), %VEC(9), %VEC(10), %VEC(11))
	LOAD_ONE_SET((%rsi), PAGE_SIZE * 3, %VEC(12), %VEC(13), %VEC(14), %VEC(15))
	subq	$-LARGE_LOAD_SIZE, %rsi
	/* Non-temporal store vectors to rdi.  */
	STORE_ONE_SET((%rdi), 0, %VEC(0), %VEC(1), %VEC(2), %VEC(3))
	STORE_ONE_SET((%rdi), PAGE_SIZE, %VEC(4), %VEC(5), %VEC(6), %VEC(7))
	STORE_ONE_SET((%rdi), PAGE_SIZE * 2, %VEC(8), %VEC(9), %VEC(10), %VEC(11))
	STORE_ONE_SET((%rdi), PAGE_SIZE * 3, %VEC(12), %VEC(13), %VEC(14), %VEC(15))
	subq	$-LARGE_LOAD_SIZE, %rdi
	decl	%ecx
	jnz	L(loop_large_memcpy_4x_inner)
	addq	$(PAGE_SIZE * 3), %rdi
	addq	$(PAGE_SIZE * 3), %rsi
	decq	%r10
	jne	L(loop_large_memcpy_4x_outer)
	sfence
	/* Check if only last 4 loads are needed.  */
	cmpl	$(VEC_SIZE * 4), %edx
	jbe	L(large_memcpy_4x_end)

	/* Handle the last 4  * PAGE_SIZE bytes.  */
L(loop_large_memcpy_4x_tail):
	/* Copy 4 * VEC a time forward with non-temporal stores.  */
	PREFETCH_ONE_SET (1, (%rsi), PREFETCHED_LOAD_SIZE)
	PREFETCH_ONE_SET (1, (%rdi), PREFETCHED_LOAD_SIZE)
	VMOVU	(%rsi), %VEC(0)
	VMOVU	VEC_SIZE(%rsi), %VEC(1)
	VMOVU	(VEC_SIZE * 2)(%rsi), %VEC(2)
	VMOVU	(VEC_SIZE * 3)(%rsi), %VEC(3)
	subq	$-(VEC_SIZE * 4), %rsi
	addl	$-(VEC_SIZE * 4), %edx
	VMOVA	%VEC(0), (%rdi)
	VMOVA	%VEC(1), VEC_SIZE(%rdi)
	VMOVA	%VEC(2), (VEC_SIZE * 2)(%rdi)
	VMOVA	%VEC(3), (VEC_SIZE * 3)(%rdi)
	subq	$-(VEC_SIZE * 4), %rdi
	cmpl	$(VEC_SIZE * 4), %edx
	ja	L(loop_large_memcpy_4x_tail)

L(large_memcpy_4x_end):
	/* Store the last 4 * VEC.  */
	VMOVU	-(VEC_SIZE * 4)(%rsi, %rdx), %VEC(0)
	VMOVU	-(VEC_SIZE * 3)(%rsi, %rdx), %VEC(1)
	VMOVU	-(VEC_SIZE * 2)(%rsi, %rdx), %VEC(2)
	VMOVU	-VEC_SIZE(%rsi, %rdx), %VEC(3)

	VMOVU	%VEC(0), -(VEC_SIZE * 4)(%rdi, %rdx)
	VMOVU	%VEC(1), -(VEC_SIZE * 3)(%rdi, %rdx)
	VMOVU	%VEC(2), -(VEC_SIZE * 2)(%rdi, %rdx)
	VMOVU	%VEC(3), -VEC_SIZE(%rdi, %rdx)
	VZEROUPPER_RETURN
#endif
END (MEMMOVE_SYMBOL (__memmove, unaligned_erms))

#if IS_IN (libc)
# ifdef USE_MULTIARCH
strong_alias (MEMMOVE_SYMBOL (__memmove, unaligned_erms),
	      MEMMOVE_SYMBOL (__memcpy, unaligned_erms))
#  ifdef SHARED
strong_alias (MEMMOVE_SYMBOL (__memmove_chk, unaligned_erms),
	      MEMMOVE_SYMBOL (__memcpy_chk, unaligned_erms))
#  endif
# endif
# ifdef SHARED
strong_alias (MEMMOVE_CHK_SYMBOL (__memmove_chk, unaligned),
	      MEMMOVE_CHK_SYMBOL (__memcpy_chk, unaligned))
# endif
#endif
strong_alias (MEMMOVE_SYMBOL (__memmove, unaligned),
	      MEMCPY_SYMBOL (__memcpy, unaligned))
