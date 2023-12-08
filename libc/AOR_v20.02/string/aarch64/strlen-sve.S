/*
 * __strlen_aarch64_sve - compute the length of a string
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

	.globl	__strlen_aarch64_sve
	.type	__strlen_aarch64_sve, %function
	.p2align 4
__strlen_aarch64_sve:
	setffr			/* initialize FFR */
	ptrue	p2.b		/* all ones; loop invariant */
	mov	x1, 0		/* initialize length */
	nop

	/* Read a vector's worth of bytes, stopping on first fault.  */
0:	ldff1b	z0.b, p2/z, [x0, x1]
	nop
	rdffrs	p0.b, p2/z
	b.nlast	2f

	/* First fault did not fail: the whole vector is valid.
	   Avoid depending on the contents of FFR beyond the branch.  */
	incb	x1, all			/* speculate increment */
	cmpeq	p1.b, p2/z, z0.b, 0	/* loop if no zeros */
	b.none	0b
	decb	x1, all			/* undo speculate */

	/* Zero found.  Select the bytes before the first and count them.  */
1:	brkb	p0.b, p2/z, p1.b
	incp	x1, p0.b
	mov	x0, x1
	ret

	/* First fault failed: only some of the vector is valid.
	   Perform the comparison only on the valid bytes.  */
2:	cmpeq	p1.b, p0/z, z0.b, 0
	b.any	1b

	/* No zero found.  Re-init FFR, increment, and loop.  */
	setffr
	incp	x1, p0.b
	b	0b

	.size	__strlen_aarch64_sve, . - __strlen_aarch64_sve
#endif
