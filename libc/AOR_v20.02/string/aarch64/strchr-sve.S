/*
 * strchr/strchrnul - find a character in a string
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#if __ARM_FEATURE_SVE
/* Assumptions:
 *
 * ARMv8-a, AArch64
 * SVE Available.
 */

	.arch	armv8-a+sve
	.text

/* To build as strchrnul, define BUILD_STRCHRNUL before compiling this file.  */
#ifdef BUILD_STRCHRNUL
#define FUNC  __strchrnul_aarch64_sve
#else
#define FUNC  __strchr_aarch64_sve
#endif

	.globl	FUNC
	.type	FUNC, %function
	.p2align 4
FUNC:
	dup	z1.b, w1		/* replicate byte across vector */
	setffr				/* initialize FFR */
	ptrue	p1.b			/* all ones; loop invariant */

	.p2align 4
	/* Read a vector's worth of bytes, stopping on first fault.  */
0:	ldff1b	z0.b, p1/z, [x0, xzr]
	rdffrs	p0.b, p1/z
	b.nlast	2f

	/* First fault did not fail: the whole vector is valid.
	   Avoid depending on the contents of FFR beyond the branch.  */
	incb	x0				/* speculate increment */
	cmpeq	p2.b, p1/z, z0.b, z1.b		/* search for c */
	cmpeq	p3.b, p1/z, z0.b, 0		/* search for 0 */
	orrs	p4.b, p1/z, p2.b, p3.b		/* c | 0 */
	b.none	0b
	decb	x0				/* undo speculate */

	/* Found C or 0.  */
1:	brka	p4.b, p1/z, p4.b	/* find first such */
	sub	x0, x0, 1		/* adjust pointer for that byte */
	incp	x0, p4.b
#ifndef BUILD_STRCHRNUL
	ptest	p4, p2.b		/* was first in c? */
	csel	x0, xzr, x0, none	/* if there was no c, return null */
#endif
	ret

	/* First fault failed: only some of the vector is valid.
	   Perform the comparison only on the valid bytes.  */
2:	cmpeq	p2.b, p0/z, z0.b, z1.b		/* search for c */
	cmpeq	p3.b, p0/z, z0.b, 0		/* search for 0 */
	orrs	p4.b, p0/z, p2.b, p3.b		/* c | 0 */
	b.any	1b

	/* No C or 0 found.  Re-init FFR, increment, and loop.  */
	setffr
	incp	x0, p0.b
	b	0b

	.size	FUNC, . - FUNC
#endif
