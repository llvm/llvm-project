/* memchr implemented using NEON.
   Copyright (C) 2011-2021 Free Software Foundation, Inc.
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

/* For __ARM_NEON__ this file defines memchr.  */
#ifndef __ARM_NEON__
# define memchr __memchr_neon
# undef libc_hidden_builtin_def
# define libc_hidden_builtin_def(a)
#endif

	.arch	armv7-a
	.fpu	neon


/* Arguments */
#define srcin		r0
#define chrin		r1
#define cntin		r2

/* Retval */
#define result		r0	/* Live range does not overlap with srcin */

/* Working registers */
#define src		r1	/* Live range does not overlap with chrin */
#define tmp		r3
#define synd		r0	/* No overlap with srcin or result */
#define soff		r12

/* Working NEON registers */
#define vrepchr		q0
#define vdata0		q1
#define vdata0_0	d2	/* Lower half of vdata0 */
#define vdata0_1	d3	/* Upper half of vdata0 */
#define vdata1		q2
#define vdata1_0	d4	/* Lower half of vhas_chr0 */
#define vdata1_1	d5	/* Upper half of vhas_chr0 */
#define vrepmask	q3
#define vrepmask0	d6
#define vrepmask1	d7
#define vend		q4
#define vend0		d8
#define vend1		d9

/*
 * Core algorithm:
 *
 * For each 32-byte chunk we calculate a 32-bit syndrome value, with one bit per
 * byte. Each bit is set if the relevant byte matched the requested character
 * and cleared otherwise. Since the bits in the syndrome reflect exactly the
 * order in which things occur in the original string, counting trailing zeros
 * allows to identify exactly which byte has matched.
 */

	.thumb_func
	.p2align 4,,15

ENTRY(memchr)
	/* Use a simple loop if there are less than 8 bytes to search.  */
	cmp	cntin, #7
	bhi	.Llargestr
	and	chrin, chrin, #0xff

.Lsmallstr:
	subs	cntin, cntin, #1
	blo	.Lnotfound	/* Return not found if reached end.  */
	ldrb	tmp, [srcin], #1
	cmp	tmp, chrin
	bne	.Lsmallstr	/* Loop again if not found.  */
	/* Otherwise fixup address and return.  */
	sub	result, srcin, #1
	bx	lr


.Llargestr:
	vdup.8	vrepchr, chrin	/* Duplicate char across all lanes. */
	/*
	 * Magic constant 0x8040201008040201 allows us to identify which lane
	 * matches the requested byte.
	 */
	movw	tmp, #0x0201
	movt	tmp, #0x0804
	lsl	soff, tmp, #4
	vmov	vrepmask0, tmp, soff
	vmov	vrepmask1, tmp, soff
	/* Work with aligned 32-byte chunks */
	bic	src, srcin, #31
	ands	soff, srcin, #31
	beq	.Lloopintro	/* Go straight to main loop if it's aligned. */

	/*
	 * Input string is not 32-byte aligned. We calculate the syndrome
	 * value for the aligned 32 bytes block containing the first bytes
	 * and mask the irrelevant part.
	 */
	vld1.8		{vdata0, vdata1}, [src:256]!
	sub		tmp, soff, #32
	adds		cntin, cntin, tmp
	vceq.i8		vdata0, vdata0, vrepchr
	vceq.i8		vdata1, vdata1, vrepchr
	vand		vdata0, vdata0, vrepmask
	vand		vdata1, vdata1, vrepmask
	vpadd.i8	vdata0_0, vdata0_0, vdata0_1
	vpadd.i8	vdata1_0, vdata1_0, vdata1_1
	vpadd.i8	vdata0_0, vdata0_0, vdata1_0
	vpadd.i8	vdata0_0, vdata0_0, vdata0_0
	vmov		synd, vdata0_0[0]

	/* Clear the soff lower bits */
	lsr		synd, synd, soff
	lsl		synd, synd, soff
	/* The first block can also be the last */
	bls		.Lmasklast
	/* Have we found something already? */
	cbnz		synd, .Ltail


.Lloopintro:
	vpush	{vend}
	/* 264/265 correspond to d8/d9 for q4 */
	cfi_adjust_cfa_offset (16)
	cfi_rel_offset (264, 0)
	cfi_rel_offset (265, 8)
	.p2align 3,,7
.Lloop:
	vld1.8		{vdata0, vdata1}, [src:256]!
	subs		cntin, cntin, #32
	vceq.i8		vdata0, vdata0, vrepchr
	vceq.i8		vdata1, vdata1, vrepchr
	/* If we're out of data we finish regardless of the result. */
	bls		.Lend
	/* Use a fast check for the termination condition. */
	vorr		vend, vdata0, vdata1
	vorr		vend0, vend0, vend1
	vmov		synd, tmp, vend0
	orrs		synd, synd, tmp
	/* We're not out of data, loop if we haven't found the character. */
	beq		.Lloop

.Lend:
	vpop		{vend}
	cfi_adjust_cfa_offset (-16)
	cfi_restore (264)
	cfi_restore (265)

	/* Termination condition found, let's calculate the syndrome value. */
	vand		vdata0, vdata0, vrepmask
	vand		vdata1, vdata1, vrepmask
	vpadd.i8	vdata0_0, vdata0_0, vdata0_1
	vpadd.i8	vdata1_0, vdata1_0, vdata1_1
	vpadd.i8	vdata0_0, vdata0_0, vdata1_0
	vpadd.i8	vdata0_0, vdata0_0, vdata0_0
	vmov		synd, vdata0_0[0]
	cbz		synd, .Lnotfound
	bhi		.Ltail	/* Uses the condition code from
				   subs cntin, cntin, #32 above.  */


.Lmasklast:
	/* Clear the (-cntin) upper bits to avoid out-of-bounds matches. */
	neg	cntin, cntin
	lsl	synd, synd, cntin
	lsrs	synd, synd, cntin
	it	eq
	moveq	src, #0	/* If no match, set src to 0 so the retval is 0. */


.Ltail:
	/* Count the trailing zeros using bit reversing */
	rbit	synd, synd
	/* Compensate the last post-increment */
	sub	src, src, #32
	/* Count the leading zeros */
	clz	synd, synd
	/* Compute the potential result and return */
	add	result, src, synd
	bx	lr


.Lnotfound:
	/* Set result to NULL if not found and return */
	mov	result, #0
	bx	lr

END(memchr)
libc_hidden_builtin_def (memchr)
