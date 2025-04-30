/* MIPS3 __mpn_mul_1 -- Multiply a limb vector with a single limb and
 * store the product in a second limb vector.
 *
 * Copyright (C) 1992-2021 Free Software Foundation, Inc.
 *
 * This file is part of the GNU MP Library.
 *
 * The GNU MP Library is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 2.1 of the License, or (at your
 * option) any later version.
 *
 * The GNU MP Library is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
 * or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public
 * License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with the GNU MP Library.  If not, see
 * <https://www.gnu.org/licenses/>.
 */

#include <sysdep.h>
#include <sys/asm.h>

/* INPUT PARAMETERS
 * res_ptr	$4
 * s1_ptr	$5
 * size		$6
 * s2_limb	$7
 */

#ifdef __PIC__
	.option pic2
#endif
ENTRY (__mpn_mul_1)
#ifdef __PIC__
	SETUP_GP /* ??? unused */
#endif
	.set    noreorder
	.set    nomacro

 # warm up phase 0
	ld	$8,0($5)

 # warm up phase 1
	daddiu	$5,$5,8
#if __mips_isa_rev < 6
	dmultu	$8,$7
#else
	dmulu	$11,$8,$7
	dmuhu	$12,$8,$7
#endif

	daddiu	$6,$6,-1
	beq	$6,$0,L(LC0)
	move	$2,$0		# zero cy2

	daddiu	$6,$6,-1
	beq	$6,$0,L(LC1)
	ld	$8,0($5)	# load new s1 limb as early as possible

#if __mips_isa_rev < 6
L(Loop):	mflo	$10
	mfhi	$9
#else
L(Loop):	move	$10,$11
		move	$9,$12
#endif
	daddiu	$5,$5,8
	daddu	$10,$10,$2	# add old carry limb to low product limb
#if __mips_isa_rev < 6
	dmultu	$8,$7
#else
	dmulu	$11,$8,$7
	dmuhu	$12,$8,$7
#endif
	ld	$8,0($5)	# load new s1 limb as early as possible
	daddiu	$6,$6,-1	# decrement loop counter
	sltu	$2,$10,$2	# carry from previous addition -> $2
	sd	$10,0($4)
	daddiu	$4,$4,8
	bne	$6,$0,L(Loop)
	daddu	$2,$9,$2	# add high product limb and carry from addition

 # cool down phase 1
#if __mips_isa_rev < 6
L(LC1):	mflo	$10
	mfhi	$9
#else
L(LC1):	move	$10,$11
	move	$9,$12
#endif
	daddu	$10,$10,$2
	sltu	$2,$10,$2
#if __mips_isa_rev < 6
	dmultu	$8,$7
#else
	dmulu	$11,$8,$7
	dmuhu	$12,$8,$7
#endif
	sd	$10,0($4)
	daddiu	$4,$4,8
	daddu	$2,$9,$2	# add high product limb and carry from addition

 # cool down phase 0
#if  __mips_isa_rev < 6
L(LC0):	mflo	$10
	mfhi	$9
#else
L(LC0):	move	$10,$11
	move	$9,$12
#endif
	daddu	$10,$10,$2
	sltu	$2,$10,$2
	sd	$10,0($4)
	j	$31
	daddu	$2,$9,$2	# add high product limb and carry from addition

END (__mpn_mul_1)
