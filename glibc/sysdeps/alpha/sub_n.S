 # Alpha __mpn_sub_n -- Subtract two limb vectors of the same length > 0 and
 # store difference in a third limb vector.

 # Copyright (C) 1995-2021 Free Software Foundation, Inc.

 # This file is part of the GNU MP Library.

 # The GNU MP Library is free software; you can redistribute it and/or modify
 # it under the terms of the GNU Lesser General Public License as published by
 # the Free Software Foundation; either version 2.1 of the License, or (at your
 # option) any later version.

 # The GNU MP Library is distributed in the hope that it will be useful, but
 # WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
 # or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public
 # License for more details.

 # You should have received a copy of the GNU Lesser General Public License
 # along with the GNU MP Library.  If not, see <https://www.gnu.org/licenses/>.


 # INPUT PARAMETERS
 # res_ptr	$16
 # s1_ptr	$17
 # s2_ptr	$18
 # size		$19

	.set	noreorder
	.set	noat
.text
	.align	3
	.globl	__mpn_sub_n
	.ent	__mpn_sub_n
__mpn_sub_n:
	.frame	$30,0,$26,0

	ldq	$3,0($17)
	ldq	$4,0($18)

	subq	$19,1,$19
	and	$19,4-1,$2	# number of limbs in first loop
	bis	$31,$31,$0
	beq	$2,.L0		# if multiple of 4 limbs, skip first loop

	subq	$19,$2,$19

.Loop0:	subq	$2,1,$2
	ldq	$5,8($17)
	addq	$4,$0,$4
	ldq	$6,8($18)
	cmpult	$4,$0,$1
	subq	$3,$4,$4
	cmpult	$3,$4,$0
	stq	$4,0($16)
	or	$0,$1,$0

	addq	$17,8,$17
	addq	$18,8,$18
	bis	$5,$5,$3
	bis	$6,$6,$4
	addq	$16,8,$16
	bne	$2,.Loop0

.L0:	beq	$19,.Lend

	.align	3
.Loop:	subq	$19,4,$19

	ldq	$5,8($17)
	addq	$4,$0,$4
	ldq	$6,8($18)
	cmpult	$4,$0,$1
	subq	$3,$4,$4
	cmpult	$3,$4,$0
	stq	$4,0($16)
	or	$0,$1,$0

	ldq	$3,16($17)
	addq	$6,$0,$6
	ldq	$4,16($18)
	cmpult	$6,$0,$1
	subq	$5,$6,$6
	cmpult	$5,$6,$0
	stq	$6,8($16)
	or	$0,$1,$0

	ldq	$5,24($17)
	addq	$4,$0,$4
	ldq	$6,24($18)
	cmpult	$4,$0,$1
	subq	$3,$4,$4
	cmpult	$3,$4,$0
	stq	$4,16($16)
	or	$0,$1,$0

	ldq	$3,32($17)
	addq	$6,$0,$6
	ldq	$4,32($18)
	cmpult	$6,$0,$1
	subq	$5,$6,$6
	cmpult	$5,$6,$0
	stq	$6,24($16)
	or	$0,$1,$0

	addq	$17,32,$17
	addq	$18,32,$18
	addq	$16,32,$16
	bne	$19,.Loop

.Lend:	addq	$4,$0,$4
	cmpult	$4,$0,$1
	subq	$3,$4,$4
	cmpult	$3,$4,$0
	stq	$4,0($16)
	or	$0,$1,$0
	ret	$31,($26),1

	.end	__mpn_sub_n
