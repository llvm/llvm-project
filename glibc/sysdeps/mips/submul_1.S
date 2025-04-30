/* MIPS __mpn_submul_1 -- Multiply a limb vector with a single limb and
subtract the product from a second limb vector.

Copyright (C) 1995-2021 Free Software Foundation, Inc.

This file is part of the GNU MP Library.

The GNU MP Library is free software; you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation; either version 2.1 of the License, or (at your
option) any later version.

The GNU MP Library is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public
License for more details.

You should have received a copy of the GNU Lesser General Public License
along with the GNU MP Library.  If not, see
<https://www.gnu.org/licenses/>.  */

#include <sysdep.h>

/* INPUT PARAMETERS
   res_ptr	$4
   s1_ptr	$5
   size		$6
   s2_limb	$7
*/
#ifdef __PIC__
	.option pic2
#endif
ENTRY (__mpn_submul_1)
	.set    noreorder
#ifdef __PIC__
	.cpload t9
#endif
	.set    nomacro

	/* warm up phase 0 */
	lw	$8,0($5)

	/* warm up phase 1 */
	addiu	$5,$5,4
#if __mips_isa_rev < 6
	multu	$8,$7
#else
	mulu	$11,$8,$7
	muhu	$12,$8,$7
#endif

	addiu	$6,$6,-1
	beq	$6,$0,L(LC0)
	move	$2,$0		/* zero cy2 */

	addiu	$6,$6,-1
	beq	$6,$0,L(LC1)
	lw	$8,0($5)	/* load new s1 limb as early as possible */

L(Loop):	lw	$10,0($4)
#if __mips_isa_rev < 6
	mflo	$3
	mfhi	$9
#else
	move	$3,$11
	move	$9,$12
#endif
	addiu	$5,$5,4
	addu	$3,$3,$2	/* add old carry limb to low product limb */
#if __mips_isa_rev < 6
	multu	$8,$7
#else
	mulu	$11,$8,$7
	muhu	$12,$8,$7
#endif
	lw	$8,0($5)	/* load new s1 limb as early as possible */
	addiu	$6,$6,-1	/* decrement loop counter */
	sltu	$2,$3,$2	/* carry from previous addition -> $2 */
	subu	$3,$10,$3
	sgtu	$10,$3,$10
	addu	$2,$2,$10
	sw	$3,0($4)
	addiu	$4,$4,4
	bne	$6,$0,L(Loop)	/* should be "bnel" */
	addu	$2,$9,$2	/* add high product limb and carry from addition */

	/* cool down phase 1 */
L(LC1):	lw	$10,0($4)
#if __mips_isa_rev < 6
	mflo	$3
	mfhi	$9
#else
	move	$3,$11
	move	$9,$12
#endif
	addu	$3,$3,$2
	sltu	$2,$3,$2
#if __mips_isa_rev < 6
	multu	$8,$7
#else
	mulu	$11,$8,$7
	muhu	$12,$8,$7
#endif
	subu	$3,$10,$3
	sgtu	$10,$3,$10
	addu	$2,$2,$10
	sw	$3,0($4)
	addiu	$4,$4,4
	addu	$2,$9,$2	/* add high product limb and carry from addition */

	/* cool down phase 0 */
L(LC0):	lw	$10,0($4)
#if __mips_isa_rev < 6
	mflo	$3
	mfhi	$9
#else
	move	$3,$11
	move	$9,$12
#endif
	addu	$3,$3,$2
	sltu	$2,$3,$2
	subu	$3,$10,$3
	sgtu	$10,$3,$10
	addu	$2,$2,$10
	sw	$3,0($4)
	j	$31
	addu	$2,$9,$2	/* add high product limb and carry from addition */
	END (__mpn_submul_1)
