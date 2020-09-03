/*
 * strrchr - find the last of a character in a string
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

	.globl	__strrchr_aarch64_sve
	.type	__strrchr_aarch64_sve, %function
	.p2align 4
__strrchr_aarch64_sve:
	dup	z1.b, w1		/* replicate byte across vector */
	setffr				/* initialize FFR */
	ptrue	p1.b			/* all ones; loop invariant */
	mov	x2, 0			/* no match found so far */
	pfalse	p2.b

	.p2align 4
	/* Read a vector's worth of bytes, stopping on first fault.  */
0:	ldff1b	z0.b, p1/z, [x0, xzr]
	rdffrs	p0.b, p1/z
	b.nlast	1f

	/* First fault did not fail: the whole vector is valid.
	   Avoid depending on the contents of FFR beyond the branch.  */
	incb	x0, all			/* skip bytes this round */
	cmpeq	p3.b, p1/z, z0.b, 0	/* search for 0 */
	b.any	3f

	cmpeq	p3.b, p1/z, z0.b, z1.b	/* search for c; no eos */
	b.none	0b

	mov	x2, x0			/* save advanced base */
	mov	p2.b, p3.b		/* save current search */
	b	0b

	/* First fault failed: only some of the vector is valid.
	   Perform the comparisons only on the valid bytes.  */
1:	cmpeq	p3.b, p0/z, z0.b, 0	/* search for 0 */
	b.any	2f

	cmpeq	p3.b, p0/z, z0.b, z1.b	/* search for c; no eos */
	mov	x3, x0
	incp	x0, p0.b		/* skip bytes this round */
	setffr				/* re-init FFR */
	b.none	0b

	addvl	x2, x3, 1		/* save advanced base */
	mov	p2.b, p3.b		/* save current search */
	b	0b

	/* Found end-of-string.  */
2:	incb	x0, all			/* advance base */
3:	brka	p3.b, p1/z, p3.b	/* mask after first 0 */
	cmpeq	p3.b, p3/z, z0.b, z1.b	/* search for c not after eos */
	b.any	4f

	/* No C within last vector.  Did we have one before?  */
	cbz	x2, 5f
	mov	x0, x2			/* restore advanced base */
	mov	p3.b, p2.b		/* restore saved search */

	/* Find the *last* match in the predicate.  This is slightly
	   more complicated than finding the first match.  */
4:	rev	p3.b, p3.b		/* reverse the bits */
	brka	p3.b, p1/z, p3.b	/* find position of last match */
	decp	x0, p3.b		/* retard pointer to last match */
	ret

	/* No C whatsoever.  Return NULL.  */
5:	mov	x0, 0
	ret

	.size	__strrchr_aarch64_sve, . - __strrchr_aarch64_sve
#endif
