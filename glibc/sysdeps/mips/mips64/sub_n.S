/* MIPS3 __mpn_sub_n -- Subtract two limb vectors of the same length > 0 and
 * store difference in a third limb vector.
 *
 * Copyright (C) 1995-2021 Free Software Foundation, Inc.
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
 * s2_ptr	$6
 * size		$7
 */

#ifdef __PIC__
	.option pic2
#endif
ENTRY (__mpn_sub_n)
#ifdef __PIC__
	SETUP_GP /* ??? unused */
#endif
	.set	noreorder
	.set	nomacro

	ld	$10,0($5)
	ld	$11,0($6)

	daddiu	$7,$7,-1
	and	$9,$7,4-1	# number of limbs in first loop
	beq	$9,$0,L(L0)	# if multiple of 4 limbs, skip first loop
	move	$2,$0

	dsubu	$7,$7,$9

L(Loop0):	daddiu	$9,$9,-1
	ld	$12,8($5)
	daddu	$11,$11,$2
	ld	$13,8($6)
	sltu	$8,$11,$2
	dsubu	$11,$10,$11
	sltu	$2,$10,$11
	sd	$11,0($4)
	or	$2,$2,$8

	daddiu	$5,$5,8
	daddiu	$6,$6,8
	move	$10,$12
	move	$11,$13
	bne	$9,$0,L(Loop0)
	daddiu	$4,$4,8

L(L0):	beq	$7,$0,L(Lend)
	nop

L(Loop):	daddiu	$7,$7,-4

	ld	$12,8($5)
	daddu	$11,$11,$2
	ld	$13,8($6)
	sltu	$8,$11,$2
	dsubu	$11,$10,$11
	sltu	$2,$10,$11
	sd	$11,0($4)
	or	$2,$2,$8

	ld	$10,16($5)
	daddu	$13,$13,$2
	ld	$11,16($6)
	sltu	$8,$13,$2
	dsubu	$13,$12,$13
	sltu	$2,$12,$13
	sd	$13,8($4)
	or	$2,$2,$8

	ld	$12,24($5)
	daddu	$11,$11,$2
	ld	$13,24($6)
	sltu	$8,$11,$2
	dsubu	$11,$10,$11
	sltu	$2,$10,$11
	sd	$11,16($4)
	or	$2,$2,$8

	ld	$10,32($5)
	daddu	$13,$13,$2
	ld	$11,32($6)
	sltu	$8,$13,$2
	dsubu	$13,$12,$13
	sltu	$2,$12,$13
	sd	$13,24($4)
	or	$2,$2,$8

	daddiu	$5,$5,32
	daddiu	$6,$6,32

	bne	$7,$0,L(Loop)
	daddiu	$4,$4,32

L(Lend):	daddu	$11,$11,$2
	sltu	$8,$11,$2
	dsubu	$11,$10,$11
	sltu	$2,$10,$11
	sd	$11,0($4)
	j	$31
	or	$2,$2,$8

END (__mpn_sub_n)
