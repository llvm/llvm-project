/*
 * strchr - find a character in a string
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

#define vrepchr		v0
#define qdata		q1
#define vdata		v1
#define vhas_nul	v2
#define vhas_chr	v3
#define vrepmask_0	v4
#define vrepmask_c	v5
#define vend		v6

#define L(l) .L ## l

/* Core algorithm.

   For each 16-byte chunk we calculate a 64-bit syndrome value, with
   four bits per byte (LSB is always in bits 0 and 1, for both big
   and little-endian systems).  For each tuple, bit 0 is set if
   the relevant byte matched the requested character; bit 1 is set
   if the relevant byte matched the NUL end of string (we trigger
   off bit0 for the special case of looking for NUL) and bits 2 and 3
   are not used.
   Since the bits in the syndrome reflect exactly the order in which
   things occur in the original string a count_trailing_zeros()
   operation will identify exactly which byte is causing the termination,
   and why. */

/* Locals and temporaries. */

ENTRY(__strchr_aarch64_mte)
	/* Magic constant 0x10011001 to allow us to identify which lane
	   matches the requested byte.  Magic constant 0x20022002 used
	   similarly for NUL termination. */
	mov	wtmp2, #0x1001
	movk	wtmp2, #0x1001, lsl #16
	dup	vrepchr.16b, chrin
	bic	src, srcin, #15		/* Work with aligned 16-byte chunks. */
	dup	vrepmask_c.4s, wtmp2
	ands	tmp1, srcin, #15
	add	vrepmask_0.4s, vrepmask_c.4s, vrepmask_c.4s /* equiv: lsl #1 */
	b.eq	L(loop)

	/* Input string is not 16-byte aligned.  Rather than forcing
	   the padding bytes to a safe value, we calculate the syndrome
	   for all the bytes, but then mask off those bits of the
	   syndrome that are related to the padding.  */
	ldr	qdata, [src], #16
	cmeq	vhas_nul.16b, vdata.16b, #0
	cmeq	vhas_chr.16b, vdata.16b, vrepchr.16b
	and	vhas_nul.16b, vhas_nul.16b, vrepmask_0.16b
	and	vhas_chr.16b, vhas_chr.16b, vrepmask_c.16b
	lsl	tmp1, tmp1, #2
	orr	vend.16b, vhas_nul.16b, vhas_chr.16b
	mov	tmp3, #~0
	addp	vend.16b, vend.16b, vend.16b		/* 128->64 */
	lsl	tmp1, tmp3, tmp1

	mov	tmp3, vend.d[0]
	ands	tmp1, tmp3, tmp1	/* Mask padding bits. */
	b.ne	L(tail)

L(loop):
	ldr	qdata, [src], #32
	cmeq	vhas_nul.16b, vdata.16b, #0
	cmeq	vhas_chr.16b, vdata.16b, vrepchr.16b
	/* Use a fast check for the termination condition.  */
	orr	vend.16b, vhas_nul.16b, vhas_chr.16b
	addp	vend.16b, vend.16b, vend.16b		/* 128->64 */
	mov	tmp1, vend.d[0]
	cbnz	tmp1, L(end)

	ldr	qdata, [src, #-16]
	cmeq	vhas_nul.16b, vdata.16b, #0
	cmeq	vhas_chr.16b, vdata.16b, vrepchr.16b
	/* Use a fast check for the termination condition.  */
	orr	vend.16b, vhas_nul.16b, vhas_chr.16b
	addp	vend.16b, vend.16b, vend.16b		/* 128->64 */
	mov	tmp1, vend.d[0]
	cbz	tmp1, L(loop)

	/* Adjust src for next two subtractions. */
	add	src, src, #16
L(end):
	/* Termination condition found.  Now need to establish exactly why
	   we terminated.  */
	and	vhas_nul.16b, vhas_nul.16b, vrepmask_0.16b
	and	vhas_chr.16b, vhas_chr.16b, vrepmask_c.16b
	sub	src, src, #16
	orr	vend.16b, vhas_nul.16b, vhas_chr.16b
	addp	vend.16b, vend.16b, vend.16b		/* 128->64 */

	mov	tmp1, vend.d[0]
L(tail):
	/* Count the trailing zeros, by bit reversing...  */
	rbit	tmp1, tmp1
	/* Re-bias source.  */
	sub	src, src, #16
	clz	tmp1, tmp1	/* And counting the leading zeros.  */
	/* Tmp1 is even if the target character was found first.  Otherwise
	   we've found the end of string and we weren't looking for NUL.  */
	tst	tmp1, #1
	add	result, src, tmp1, lsr #2
	csel	result, result, xzr, eq
	ret

END(__strchr_aarch64_mte)
