/* Optimized memset for Fujitsu A64FX processor.
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
   License along with the GNU C Library.  If not, see
   <https://www.gnu.org/licenses/>.  */

#include <sysdep.h>
#include <sysdeps/aarch64/memset-reg.h>

/* Assumptions:
 *
 * ARMv8.2-a, AArch64, unaligned accesses, sve
 *
 */

#define L1_SIZE		(64*1024)	// L1 64KB
#define L2_SIZE         (8*1024*1024)	// L2 8MB - 1MB
#define CACHE_LINE_SIZE	256
#define PF_DIST_L1	(CACHE_LINE_SIZE * 16)	// Prefetch distance L1
#define ZF_DIST		(CACHE_LINE_SIZE * 21)	// Zerofill distance
#define rest		x8
#define vector_length	x9
#define vl_remainder	x10	// vector_length remainder
#define cl_remainder	x11	// CACHE_LINE_SIZE remainder

#if HAVE_AARCH64_SVE_ASM
# if IS_IN (libc)
#  define MEMSET __memset_a64fx

	.arch armv8.2-a+sve

	.macro dc_zva times
	dc	zva, tmp1
	add	tmp1, tmp1, CACHE_LINE_SIZE
	.if \times-1
	dc_zva "(\times-1)"
	.endif
	.endm

	.macro st1b_unroll first=0, last=7
	st1b	z0.b, p0, [dst, #\first, mul vl]
	.if \last-\first
	st1b_unroll "(\first+1)", \last
	.endif
	.endm

	.macro shortcut_for_small_size exit
	// if rest <= vector_length * 2
	whilelo	p0.b, xzr, count
	whilelo	p1.b, vector_length, count
	b.last	1f
	st1b	z0.b, p0, [dstin, #0, mul vl]
	st1b	z0.b, p1, [dstin, #1, mul vl]
	ret
1:	// if rest > vector_length * 8
	cmp	count, vector_length, lsl 3	// vector_length * 8
	b.hi	\exit
	// if rest <= vector_length * 4
	lsl	tmp1, vector_length, 1	// vector_length * 2
	whilelo	p2.b, tmp1, count
	incb	tmp1
	whilelo	p3.b, tmp1, count
	b.last	1f
	st1b	z0.b, p0, [dstin, #0, mul vl]
	st1b	z0.b, p1, [dstin, #1, mul vl]
	st1b	z0.b, p2, [dstin, #2, mul vl]
	st1b	z0.b, p3, [dstin, #3, mul vl]
	ret
1:	// if rest <= vector_length * 8
	lsl	tmp1, vector_length, 2	// vector_length * 4
	whilelo	p4.b, tmp1, count
	incb	tmp1
	whilelo	p5.b, tmp1, count
	b.last	1f
	st1b	z0.b, p0, [dstin, #0, mul vl]
	st1b	z0.b, p1, [dstin, #1, mul vl]
	st1b	z0.b, p2, [dstin, #2, mul vl]
	st1b	z0.b, p3, [dstin, #3, mul vl]
	st1b	z0.b, p4, [dstin, #4, mul vl]
	st1b	z0.b, p5, [dstin, #5, mul vl]
	ret
1:	lsl	tmp1, vector_length, 2	// vector_length * 4
	incb	tmp1			// vector_length * 5
	incb	tmp1			// vector_length * 6
	whilelo	p6.b, tmp1, count
	incb	tmp1
	whilelo	p7.b, tmp1, count
	st1b	z0.b, p0, [dstin, #0, mul vl]
	st1b	z0.b, p1, [dstin, #1, mul vl]
	st1b	z0.b, p2, [dstin, #2, mul vl]
	st1b	z0.b, p3, [dstin, #3, mul vl]
	st1b	z0.b, p4, [dstin, #4, mul vl]
	st1b	z0.b, p5, [dstin, #5, mul vl]
	st1b	z0.b, p6, [dstin, #6, mul vl]
	st1b	z0.b, p7, [dstin, #7, mul vl]
	ret
	.endm

ENTRY (MEMSET)

	PTR_ARG (0)
	SIZE_ARG (2)

	cbnz	count, 1f
	ret
1:	dup	z0.b, valw
	cntb	vector_length
	// shortcut for less than vector_length * 8
	// gives a free ptrue to p0.b for n >= vector_length
	shortcut_for_small_size L(vl_agnostic)
	// end of shortcut

L(vl_agnostic): // VL Agnostic
	mov	rest, count
	mov	dst, dstin
	add	dstend, dstin, count
	// if rest >= L2_SIZE && vector_length == 64 then L(L2)
	mov	tmp1, 64
	cmp	rest, L2_SIZE
	ccmp	vector_length, tmp1, 0, cs
	b.eq	L(L2)
	// if rest >= L1_SIZE && vector_length == 64 then L(L1_prefetch)
	cmp	rest, L1_SIZE
	ccmp	vector_length, tmp1, 0, cs
	b.eq	L(L1_prefetch)

L(unroll32):
	lsl	tmp1, vector_length, 3	// vector_length * 8
	lsl	tmp2, vector_length, 5	// vector_length * 32
	.p2align 3
1:	cmp	rest, tmp2
	b.cc	L(unroll8)
	st1b_unroll
	add	dst, dst, tmp1
	st1b_unroll
	add	dst, dst, tmp1
	st1b_unroll
	add	dst, dst, tmp1
	st1b_unroll
	add	dst, dst, tmp1
	sub	rest, rest, tmp2
	b	1b

L(unroll8):
	lsl	tmp1, vector_length, 3
	.p2align 3
1:	cmp	rest, tmp1
	b.cc	L(last)
	st1b_unroll
	add	dst, dst, tmp1
	sub	rest, rest, tmp1
	b	1b

L(last):
	whilelo	p0.b, xzr, rest
	whilelo	p1.b, vector_length, rest
	b.last	1f
	st1b	z0.b, p0, [dst, #0, mul vl]
	st1b	z0.b, p1, [dst, #1, mul vl]
	ret
1:	lsl	tmp1, vector_length, 1	// vector_length * 2
	whilelo	p2.b, tmp1, rest
	incb	tmp1
	whilelo	p3.b, tmp1, rest
	b.last	1f
	st1b	z0.b, p0, [dst, #0, mul vl]
	st1b	z0.b, p1, [dst, #1, mul vl]
	st1b	z0.b, p2, [dst, #2, mul vl]
	st1b	z0.b, p3, [dst, #3, mul vl]
	ret
1:	lsl	tmp1, vector_length, 2	// vector_length * 4
	whilelo	p4.b, tmp1, rest
	incb	tmp1
	whilelo	p5.b, tmp1, rest
	incb	tmp1
	whilelo	p6.b, tmp1, rest
	incb	tmp1
	whilelo	p7.b, tmp1, rest
	st1b	z0.b, p0, [dst, #0, mul vl]
	st1b	z0.b, p1, [dst, #1, mul vl]
	st1b	z0.b, p2, [dst, #2, mul vl]
	st1b	z0.b, p3, [dst, #3, mul vl]
	st1b	z0.b, p4, [dst, #4, mul vl]
	st1b	z0.b, p5, [dst, #5, mul vl]
	st1b	z0.b, p6, [dst, #6, mul vl]
	st1b	z0.b, p7, [dst, #7, mul vl]
	ret

L(L1_prefetch): // if rest >= L1_SIZE
	.p2align 3
1:	st1b_unroll 0, 3
	prfm	pstl1keep, [dst, PF_DIST_L1]
	st1b_unroll 4, 7
	prfm	pstl1keep, [dst, PF_DIST_L1 + CACHE_LINE_SIZE]
	add	dst, dst, CACHE_LINE_SIZE * 2
	sub	rest, rest, CACHE_LINE_SIZE * 2
	cmp	rest, L1_SIZE
	b.ge	1b
	cbnz	rest, L(unroll32)
	ret

L(L2):
	// align dst address at vector_length byte boundary
	sub	tmp1, vector_length, 1
	ands	tmp2, dst, tmp1
	// if vl_remainder == 0
	b.eq	1f
	sub	vl_remainder, vector_length, tmp2
	// process remainder until the first vector_length boundary
	whilelt	p2.b, xzr, vl_remainder
	st1b	z0.b, p2, [dst]
	add	dst, dst, vl_remainder
	sub	rest, rest, vl_remainder
	// align dstin address at CACHE_LINE_SIZE byte boundary
1:	mov	tmp1, CACHE_LINE_SIZE
	ands	tmp2, dst, CACHE_LINE_SIZE - 1
	// if cl_remainder == 0
	b.eq	L(L2_dc_zva)
	sub	cl_remainder, tmp1, tmp2
	// process remainder until the first CACHE_LINE_SIZE boundary
	mov	tmp1, xzr       // index
2:	whilelt	p2.b, tmp1, cl_remainder
	st1b	z0.b, p2, [dst, tmp1]
	incb	tmp1
	cmp	tmp1, cl_remainder
	b.lo	2b
	add	dst, dst, cl_remainder
	sub	rest, rest, cl_remainder

L(L2_dc_zva):
	// zero fill
	mov	tmp1, dst
	dc_zva	(ZF_DIST / CACHE_LINE_SIZE) - 1
	mov	zva_len, ZF_DIST
	add	tmp1, zva_len, CACHE_LINE_SIZE * 2
	// unroll
	.p2align 3
1:	st1b_unroll 0, 3
	add	tmp2, dst, zva_len
	dc	 zva, tmp2
	st1b_unroll 4, 7
	add	tmp2, tmp2, CACHE_LINE_SIZE
	dc	zva, tmp2
	add	dst, dst, CACHE_LINE_SIZE * 2
	sub	rest, rest, CACHE_LINE_SIZE * 2
	cmp	rest, tmp1	// ZF_DIST + CACHE_LINE_SIZE * 2
	b.ge	1b
	cbnz	rest, L(unroll8)
	ret

END (MEMSET)
libc_hidden_builtin_def (MEMSET)

#endif /* IS_IN (libc) */
#endif /* HAVE_AARCH64_SVE_ASM */
