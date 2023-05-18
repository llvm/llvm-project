/*
 * __strcmp_aarch64_sve - compare two strings
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

	.globl	__strcmp_aarch64_sve
	.type	__strcmp_aarch64_sve, %function
	.p2align 4
__strcmp_aarch64_sve:
	setffr				/* initialize FFR */
	ptrue	p1.b, all		/* all ones; loop invariant */
	mov	x2, 0			/* initialize offset */
	nop

	/* Read a vector's worth of bytes, stopping on first fault.  */
0:	ldff1b	z0.b, p1/z, [x0, x2]
	ldff1b	z1.b, p1/z, [x1, x2]
	rdffrs	p0.b, p1/z
	b.nlast	2f

	/* First fault did not fail: the whole vector is valid.
	   Avoid depending on the contents of FFR beyond the branch.  */
	incb	x2, all			/* skip bytes for next round */
	cmpeq	p2.b, p1/z, z0.b, z1.b	/* compare strings */
	cmpne	p3.b, p1/z, z0.b, 0	/* search for ~zero */
	nands	p2.b, p1/z, p2.b, p3.b	/* ~(eq & ~zero) -> ne | zero */
	b.none	0b

	/* Found end-of-string or inequality.  */
1:	brkb	p2.b, p1/z, p2.b	/* find first such */
	lasta	w0, p2, z0.b		/* extract each char */
	lasta	w1, p2, z1.b
	sub	x0, x0, x1		/* return comparison */
	ret

	/* First fault failed: only some of the vector is valid.
	   Perform the comparison only on the valid bytes.  */
2:	incp	x2, p0.b		/* skip bytes for next round */
	setffr				/* re-init FFR for next round */
	cmpeq	p2.b, p0/z, z0.b, z1.b	/* compare strings, as above */
	cmpne	p3.b, p0/z, z0.b, 0
	nands	p2.b, p0/z, p2.b, p3.b
	b.none	0b
	b	1b

	.size	__strcmp_aarch64_sve, . - __strcmp_aarch64_sve
#endif
