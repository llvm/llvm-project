/*
 * strrchr - find last position of a character in a string.
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

/* Assumptions:
 *
 * ARMv8-a, AArch64
 * Neon Available.
 */

#include "../asmdefs.h"

/* Arguments and results.  */
#define srcin		x0
#define chrin		w1

#define result		x0

#define src		x2
#define	tmp1		x3
#define wtmp2		w4
#define tmp3		x5
#define src_match	x6
#define src_offset	x7
#define const_m1	x8
#define tmp4		x9
#define nul_match	x10
#define chr_match	x11

#define vrepchr		v0
#define vdata1		v1
#define vdata2		v2
#define vhas_nul1	v3
#define vhas_nul2	v4
#define vhas_chr1	v5
#define vhas_chr2	v6
#define vrepmask_0	v7
#define vrepmask_c	v16
#define vend1		v17
#define vend2		v18

/* Core algorithm.

   For each 32-byte hunk we calculate a 64-bit syndrome value, with
   two bits per byte (LSB is always in bits 0 and 1, for both big
   and little-endian systems).  For each tuple, bit 0 is set iff
   the relevant byte matched the requested character; bit 1 is set
   iff the relevant byte matched the NUL end of string (we trigger
   off bit0 for the special case of looking for NUL).  Since the bits
   in the syndrome reflect exactly the order in which things occur
   in the original string a count_trailing_zeros() operation will
   identify exactly which byte is causing the termination, and why.  */

ENTRY (__strrchr_aarch64)
	/* Magic constant 0x40100401 to allow us to identify which lane
	   matches the requested byte.  Magic constant 0x80200802 used
	   similarly for NUL termination.  */
	mov	wtmp2, #0x0401
	movk	wtmp2, #0x4010, lsl #16
	dup	vrepchr.16b, chrin
	bic	src, srcin, #31		/* Work with aligned 32-byte hunks.  */
	dup	vrepmask_c.4s, wtmp2
	mov	src_offset, #0
	ands	tmp1, srcin, #31
	add	vrepmask_0.4s, vrepmask_c.4s, vrepmask_c.4s /* equiv: lsl #1 */
	b.eq	L(aligned)

	/* Input string is not 32-byte aligned.  Rather than forcing
	   the padding bytes to a safe value, we calculate the syndrome
	   for all the bytes, but then mask off those bits of the
	   syndrome that are related to the padding.  */
	ld1	{vdata1.16b, vdata2.16b}, [src], #32
	neg	tmp1, tmp1
	cmeq	vhas_nul1.16b, vdata1.16b, #0
	cmeq	vhas_chr1.16b, vdata1.16b, vrepchr.16b
	cmeq	vhas_nul2.16b, vdata2.16b, #0
	cmeq	vhas_chr2.16b, vdata2.16b, vrepchr.16b
	and	vhas_nul1.16b, vhas_nul1.16b, vrepmask_0.16b
	and	vhas_chr1.16b, vhas_chr1.16b, vrepmask_c.16b
	and	vhas_nul2.16b, vhas_nul2.16b, vrepmask_0.16b
	and	vhas_chr2.16b, vhas_chr2.16b, vrepmask_c.16b
	addp	vhas_nul1.16b, vhas_nul1.16b, vhas_nul2.16b	// 256->128
	addp	vhas_chr1.16b, vhas_chr1.16b, vhas_chr2.16b	// 256->128
	addp	vhas_nul1.16b, vhas_nul1.16b, vhas_nul1.16b	// 128->64
	addp	vhas_chr1.16b, vhas_chr1.16b, vhas_chr1.16b	// 128->64
	mov	nul_match, vhas_nul1.d[0]
	lsl	tmp1, tmp1, #1
	mov	const_m1, #~0
	mov	chr_match, vhas_chr1.d[0]
	lsr	tmp3, const_m1, tmp1

	bic	nul_match, nul_match, tmp3	// Mask padding bits.
	bic	chr_match, chr_match, tmp3	// Mask padding bits.
	cbnz	nul_match, L(tail)

L(loop):
	cmp	chr_match, #0
	csel	src_match, src, src_match, ne
	csel	src_offset, chr_match, src_offset, ne
L(aligned):
	ld1	{vdata1.16b, vdata2.16b}, [src], #32
	cmeq	vhas_nul1.16b, vdata1.16b, #0
	cmeq	vhas_chr1.16b, vdata1.16b, vrepchr.16b
	cmeq	vhas_nul2.16b, vdata2.16b, #0
	cmeq	vhas_chr2.16b, vdata2.16b, vrepchr.16b
	addp	vend1.16b, vhas_nul1.16b, vhas_nul2.16b	// 256->128
	and	vhas_chr1.16b, vhas_chr1.16b, vrepmask_c.16b
	and	vhas_chr2.16b, vhas_chr2.16b, vrepmask_c.16b
	addp	vhas_chr1.16b, vhas_chr1.16b, vhas_chr2.16b	// 256->128
	addp	vend1.16b, vend1.16b, vend1.16b	// 128->64
	addp	vhas_chr1.16b, vhas_chr1.16b, vhas_chr1.16b	// 128->64
	mov	nul_match, vend1.d[0]
	mov	chr_match, vhas_chr1.d[0]
	cbz	nul_match, L(loop)

	and	vhas_nul1.16b, vhas_nul1.16b, vrepmask_0.16b
	and	vhas_nul2.16b, vhas_nul2.16b, vrepmask_0.16b
	addp	vhas_nul1.16b, vhas_nul1.16b, vhas_nul2.16b
	addp	vhas_nul1.16b, vhas_nul1.16b, vhas_nul1.16b
	mov	nul_match, vhas_nul1.d[0]

L(tail):
	/* Work out exactly where the string ends.  */
	sub	tmp4, nul_match, #1
	eor	tmp4, tmp4, nul_match
	ands	chr_match, chr_match, tmp4
	/* And pick the values corresponding to the last match.  */
	csel	src_match, src, src_match, ne
	csel	src_offset, chr_match, src_offset, ne

	/* Count down from the top of the syndrome to find the last match.  */
	clz	tmp3, src_offset
	/* Src_match points beyond the word containing the match, so we can
	   simply subtract half the bit-offset into the syndrome.  Because
	   we are counting down, we need to go back one more character.  */
	add	tmp3, tmp3, #2
	sub	result, src_match, tmp3, lsr #1
	/* But if the syndrome shows no match was found, then return NULL.  */
	cmp	src_offset, #0
	csel	result, result, xzr, ne

	ret

END (__strrchr_aarch64)
