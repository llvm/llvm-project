/*
 * strcpy/stpcpy - copy a string returning pointer to start/end.
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

/* To build as stpcpy, define BUILD_STPCPY before compiling this file.  */
#ifdef BUILD_STPCPY
#define FUNC  __stpcpy_aarch64_sve
#else
#define FUNC  __strcpy_aarch64_sve
#endif

	.globl	FUNC
	.type	FUNC, %function
	.p2align 4
FUNC:
	setffr				/* initialize FFR */
	ptrue	p2.b, all		/* all ones; loop invariant */
	mov	x2, 0			/* initialize offset */

	.p2align 4
	/* Read a vector's worth of bytes, stopping on first fault.  */
0:	ldff1b	z0.b, p2/z, [x1, x2]
	rdffrs	p0.b, p2/z
	b.nlast	1f

	/* First fault did not fail: the whole vector is valid.
	   Avoid depending on the contexts of FFR beyond the branch.  */
	cmpeq	p1.b, p2/z, z0.b, 0	/* search for zeros */
	b.any	2f

	/* No zero found.  Store the whole vector and loop.  */
	st1b	z0.b, p2, [x0, x2]
	incb	x2, all
	b	0b

	/* First fault failed: only some of the vector is valid.
	   Perform the comparison only on the valid bytes.  */
1:	cmpeq	p1.b, p0/z, z0.b, 0	/* search for zeros */
	b.any	2f

	/* No zero found.  Store the valid portion of the vector and loop.  */
	setffr				/* re-init FFR */
	st1b	z0.b, p0, [x0, x2]
	incp	x2, p0.b
	b	0b

	/* Zero found.  Crop the vector to the found zero and finish.  */
2:	brka	p0.b, p2/z, p1.b
	st1b	z0.b, p0, [x0, x2]
#ifdef BUILD_STPCPY
	add	x0, x0, x2
	sub	x0, x0, 1
	incp	x0, p0.b
#endif
	ret

	.size	FUNC, . - FUNC
#endif
