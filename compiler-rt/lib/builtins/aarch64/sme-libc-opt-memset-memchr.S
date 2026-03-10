//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains assembly-optimized implementations of Scalable Matrix
/// Extension (SME) compatible memset and memchr functions.
///
/// These implementations depend on unaligned access and floating-point support.
///
/// Routines taken from libc/AOR_v20.02/string/aarch64.
///
//===----------------------------------------------------------------------===//

#include "../assembly.h"

//
//  __arm_sc_memset
//

#define dstin    x0
#define val      x1
#define valw     w1
#define count    x2
#define dst      x3
#define dstend2  x4
#define zva_val  x5

DEFINE_COMPILERRT_FUNCTION(__arm_sc_memset)
#ifdef __ARM_FEATURE_SVE
        mov     z0.b, valw
#else
        bfi valw, valw, #8, #8
        bfi valw, valw, #16, #16
        bfi val, val, #32, #32
        fmov d0, val
        fmov v0.d[1], val
#endif
        add     dstend2, dstin, count

        cmp     count, 96
        b.hi    7f  // set_long
        cmp     count, 16
        b.hs    4f  // set_medium
        mov     val, v0.D[0]

        /* Set 0..15 bytes.  */
        tbz     count, 3, 1f
        str     val, [dstin]
        str     val, [dstend2, -8]
        ret
        nop
1:      tbz     count, 2, 2f
        str     valw, [dstin]
        str     valw, [dstend2, -4]
        ret
2:      cbz     count, 3f
        strb    valw, [dstin]
        tbz     count, 1, 3f
        strh    valw, [dstend2, -2]
3:      ret

        /* Set 17..96 bytes.  */
4:  // set_medium
        str     q0, [dstin]
        tbnz    count, 6, 6f  // set96
        str     q0, [dstend2, -16]
        tbz     count, 5, 5f
        str     q0, [dstin, 16]
        str     q0, [dstend2, -32]
5:      ret

        .p2align 4
        /* Set 64..96 bytes.  Write 64 bytes from the start and
           32 bytes from the end.  */
6:  // set96
        str     q0, [dstin, 16]
        stp     q0, q0, [dstin, 32]
        stp     q0, q0, [dstend2, -32]
        ret

        .p2align 4
7:  // set_long
        and     valw, valw, 255
        bic     dst, dstin, 15
        str     q0, [dstin]
        cmp     count, 160
        ccmp    valw, 0, 0, hs
        b.ne    9f  // no_zva

#ifndef SKIP_ZVA_CHECK
        mrs     zva_val, dczid_el0
        and     zva_val, zva_val, 31
        cmp     zva_val, 4              /* ZVA size is 64 bytes.  */
        b.ne    9f  // no_zva
#endif
        str     q0, [dst, 16]
        stp     q0, q0, [dst, 32]
        bic     dst, dst, 63
        sub     count, dstend2, dst      /* Count is now 64 too large.  */
        sub     count, count, 128       /* Adjust count and bias for loop.  */

        .p2align 4
8:  // zva_loop
        add     dst, dst, 64
        dc      zva, dst
        subs    count, count, 64
        b.hi    8b  // zva_loop
        stp     q0, q0, [dstend2, -64]
        stp     q0, q0, [dstend2, -32]
        ret

9:  // no_zva
        sub     count, dstend2, dst      /* Count is 16 too large.  */
        sub     dst, dst, 16            /* Dst is biased by -32.  */
        sub     count, count, 64 + 16   /* Adjust count and bias for loop.  */
10: // no_zva_loop
        stp     q0, q0, [dst, 32]
        stp     q0, q0, [dst, 64]!
        subs    count, count, 64
        b.hi    10b  // no_zva_loop
        stp     q0, q0, [dstend2, -64]
        stp     q0, q0, [dstend2, -32]
        ret
END_COMPILERRT_FUNCTION(__arm_sc_memset)

//
//  __arm_sc_memchr
//

#define srcin		x0
#define chrin		w1
#define cntin		x2

#define result		x0

#define src		x3
#define	tmp		x4
#define wtmp2		w5
#define synd		x6
#define soff		x9
#define cntrem		x10

#define vrepchr		v0
#define vdata1		v1
#define vdata2		v2
#define vhas_chr1	v3
#define vhas_chr2	v4
#define vrepmask	v5
#define vend		v6

/*
 * Core algorithm:
 *
 * For each 32-byte chunk we calculate a 64-bit syndrome value, with two bits
 * per byte. For each tuple, bit 0 is set if the relevant byte matched the
 * requested character and bit 1 is not used (faster than using a 32bit
 * syndrome). Since the bits in the syndrome reflect exactly the order in which
 * things occur in the original string, counting trailing zeros allows to
 * identify exactly which byte has matched.
 */

DEFINE_COMPILERRT_FUNCTION(__arm_sc_memchr)
	/* Do not dereference srcin if no bytes to compare.  */
	cbz	cntin, 4f
	/*
	 * Magic constant 0x40100401 allows us to identify which lane matches
	 * the requested byte.
	 */
	mov	wtmp2, #0x0401
	movk	wtmp2, #0x4010, lsl #16
	dup	vrepchr.16b, chrin
	/* Work with aligned 32-byte chunks */
	bic	src, srcin, #31
	dup	vrepmask.4s, wtmp2
	ands	soff, srcin, #31
	and	cntrem, cntin, #31
	b.eq	0f

	/*
	 * Input string is not 32-byte aligned. We calculate the syndrome
	 * value for the aligned 32 bytes block containing the first bytes
	 * and mask the irrelevant part.
	 */

	ld1	{vdata1.16b, vdata2.16b}, [src], #32
	sub	tmp, soff, #32
	adds	cntin, cntin, tmp
	cmeq	vhas_chr1.16b, vdata1.16b, vrepchr.16b
	cmeq	vhas_chr2.16b, vdata2.16b, vrepchr.16b
	and	vhas_chr1.16b, vhas_chr1.16b, vrepmask.16b
	and	vhas_chr2.16b, vhas_chr2.16b, vrepmask.16b
	addp	vend.16b, vhas_chr1.16b, vhas_chr2.16b		/* 256->128 */
	addp	vend.16b, vend.16b, vend.16b			/* 128->64 */
	mov	synd, vend.d[0]
	/* Clear the soff*2 lower bits */
	lsl	tmp, soff, #1
	lsr	synd, synd, tmp
	lsl	synd, synd, tmp
	/* The first block can also be the last */
	b.ls	2f
	/* Have we found something already? */
	cbnz	synd, 3f

0: // loop
	ld1	{vdata1.16b, vdata2.16b}, [src], #32
	subs	cntin, cntin, #32
	cmeq	vhas_chr1.16b, vdata1.16b, vrepchr.16b
	cmeq	vhas_chr2.16b, vdata2.16b, vrepchr.16b
	/* If we're out of data we finish regardless of the result */
	b.ls	1f
	/* Use a fast check for the termination condition */
	orr	vend.16b, vhas_chr1.16b, vhas_chr2.16b
	addp	vend.2d, vend.2d, vend.2d
	mov	synd, vend.d[0]
	/* We're not out of data, loop if we haven't found the character */
	cbz	synd, 0b

1: // end
	/* Termination condition found, let's calculate the syndrome value */
	and	vhas_chr1.16b, vhas_chr1.16b, vrepmask.16b
	and	vhas_chr2.16b, vhas_chr2.16b, vrepmask.16b
	addp	vend.16b, vhas_chr1.16b, vhas_chr2.16b		/* 256->128 */
	addp	vend.16b, vend.16b, vend.16b			/* 128->64 */
	mov	synd, vend.d[0]
	/* Only do the clear for the last possible block */
	b.hi	3f

2: // masklast
	/* Clear the (32 - ((cntrem + soff) % 32)) * 2 upper bits */
	add	tmp, cntrem, soff
	and	tmp, tmp, #31
	sub	tmp, tmp, #32
	neg	tmp, tmp, lsl #1
	lsl	synd, synd, tmp
	lsr	synd, synd, tmp

3: // tail
	/* Count the trailing zeros using bit reversing */
	rbit	synd, synd
	/* Compensate the last post-increment */
	sub	src, src, #32
	/* Check that we have found a character */
	cmp	synd, #0
	/* And count the leading zeros */
	clz	synd, synd
	/* Compute the potential result */
	add	result, src, synd, lsr #1
	/* Select result or NULL */
	csel	result, xzr, result, eq
	ret

4: // zero_length
	mov	result, #0
	ret
END_COMPILERRT_FUNCTION(__arm_sc_memchr)

