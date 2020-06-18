/*
 * strnlen - calculate the length of a string with limit.
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

	.globl	__strnlen_aarch64_sve
	.type	__strnlen_aarch64_sve, %function
	.p2align 4
__strnlen_aarch64_sve:
	setffr				/* initialize FFR */
	mov	x2, 0			/* initialize len */
	b	1f

	.p2align 4
	/* We have off + vl <= max, and so may read the whole vector.  */
0:	ldff1b	z0.b, p0/z, [x0, x2]
	rdffrs	p1.b, p0/z
	b.nlast	2f

	/* First fault did not fail: the whole vector is valid.
	   Avoid depending on the contents of FFR beyond the branch.  */
	cmpeq	p2.b, p0/z, z0.b, 0
	b.any	8f
	incb	x2

1:	whilelo	p0.b, x2, x1
	b.last	0b

	/* We have off + vl < max.  Test for off == max before proceeding.  */
	b.none	9f

	ldff1b	z0.b, p0/z, [x0, x2]
	rdffrs	p1.b, p0/z
	b.nlast	2f

	/* First fault did not fail: the vector up to max is valid.
	   Avoid depending on the contents of FFR beyond the branch.
	   Compare for end-of-string, but there are no more bytes.  */
	cmpeq	p2.b, p0/z, z0.b, 0

	/* Found end-of-string or zero.  */
8:	brkb	p2.b, p0/z, p2.b
	mov	x0, x2
	incp	x0, p2.b
	ret

	/* First fault failed: only some of the vector is valid.
	   Perform the comparison only on the valid bytes.  */
2:	cmpeq	p2.b, p1/z, z0.b, 0
	b.any	8b

	/* No inequality or zero found.  Re-init FFR, incr and loop.  */
	setffr
	incp	x2, p1.b
	b	1b

	/* End of count.  Return max.  */
9:	mov	x0, x2
	ret

	.size	__strnlen_aarch64_sve, . - __strnlen_aarch64_sve
#endif
