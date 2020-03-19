/*
 * strncmp - compare two strings with limit
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

	.globl	__strncmp_aarch64_sve
	.type	__strncmp_aarch64_sve, %function
	.p2align 4
__strncmp_aarch64_sve:
	setffr				/* initialize FFR */
	mov	x3, 0			/* initialize off */

0:	whilelo	p0.b, x3, x2		/* while off < max */
	b.none	9f

	ldff1b	z0.b, p0/z, [x0, x3]
	ldff1b	z1.b, p0/z, [x1, x3]
	rdffrs	p1.b, p0/z
	b.nlast	2f

	/* First fault did not fail: the vector up to max is valid.
	   Avoid depending on the contents of FFR beyond the branch.
	   Increment for a whole vector, even if we've only read a partial.
	   This is significantly cheaper than INCP, and since OFF is not
	   used after the loop it is ok to increment OFF past MAX.  */
	incb	x3
	cmpeq	p1.b, p0/z, z0.b, z1.b	/* compare strings */
	cmpne	p2.b, p0/z, z0.b, 0	/* search for ~zero */
	nands	p2.b, p0/z, p1.b, p2.b	/* ~(eq & ~zero) -> ne | zero */
	b.none	0b

	/* Found end-of-string or inequality.  */
1:	brkb	p2.b, p0/z, p2.b	/* find first such */
	lasta	w0, p2, z0.b		/* extract each char */
	lasta	w1, p2, z1.b
	sub	x0, x0, x1		/* return comparison */
	ret

	/* First fault failed: only some of the vector is valid.
	   Perform the comparison only on the valid bytes.  */
2:	cmpeq	p2.b, p1/z, z0.b, z1.b	/* compare strings, as above */
	cmpne	p3.b, p1/z, z0.b, 0
	nands	p2.b, p1/z, p2.b, p3.b
	b.any	1b

	/* No inequality or zero found.  Re-init FFR, incr and loop.  */
	setffr
	incp	x3, p1.b
	b	0b

	/* Found end-of-count.  */
9:	mov	x0, 0			/* return equal */
	ret

	.size	__strncmp_aarch64_sve, . - __strncmp_aarch64_sve
#endif
