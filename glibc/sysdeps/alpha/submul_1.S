 # Alpha 21064 __mpn_submul_1 -- Multiply a limb vector with a limb and
 # subtract the result from a second limb vector.

 # Copyright (C) 1992-2021 Free Software Foundation, Inc.

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
 # res_ptr	r16
 # s1_ptr	r17
 # size		r18
 # s2_limb	r19

 # This code runs at 42 cycles/limb on EV4 and 18 cycles/limb on EV5.

	.set	noreorder
	.set	noat
.text
	.align	3
	.globl	__mpn_submul_1
	.ent	__mpn_submul_1 2
__mpn_submul_1:
	.frame	$30,0,$26

	ldq	$2,0($17)	# $2 = s1_limb
	addq	$17,8,$17	# s1_ptr++
	subq	$18,1,$18	# size--
	mulq	$2,$19,$3	# $3 = prod_low
	ldq	$5,0($16)	# $5 = *res_ptr
	umulh	$2,$19,$0	# $0 = prod_high
	beq	$18,.Lend1	# jump if size was == 1
	ldq	$2,0($17)	# $2 = s1_limb
	addq	$17,8,$17	# s1_ptr++
	subq	$18,1,$18	# size--
	subq	$5,$3,$3
	cmpult	$5,$3,$4
	stq	$3,0($16)
	addq	$16,8,$16	# res_ptr++
	beq	$18,.Lend2	# jump if size was == 2

	.align	3
.Loop:	mulq	$2,$19,$3	# $3 = prod_low
	ldq	$5,0($16)	# $5 = *res_ptr
	addq	$4,$0,$0	# cy_limb = cy_limb + 'cy'
	subq	$18,1,$18	# size--
	umulh	$2,$19,$4	# $4 = cy_limb
	ldq	$2,0($17)	# $2 = s1_limb
	addq	$17,8,$17	# s1_ptr++
	addq	$3,$0,$3	# $3 = cy_limb + prod_low
	cmpult	$3,$0,$0	# $0 = carry from (cy_limb + prod_low)
	subq	$5,$3,$3
	cmpult	$5,$3,$5
	stq	$3,0($16)
	addq	$16,8,$16	# res_ptr++
	addq	$5,$0,$0	# combine carries
	bne	$18,.Loop

.Lend2:	mulq	$2,$19,$3	# $3 = prod_low
	ldq	$5,0($16)	# $5 = *res_ptr
	addq	$4,$0,$0	# cy_limb = cy_limb + 'cy'
	umulh	$2,$19,$4	# $4 = cy_limb
	addq	$3,$0,$3	# $3 = cy_limb + prod_low
	cmpult	$3,$0,$0	# $0 = carry from (cy_limb + prod_low)
	subq	$5,$3,$3
	cmpult	$5,$3,$5
	stq	$3,0($16)
	addq	$5,$0,$0	# combine carries
	addq	$4,$0,$0	# cy_limb = prod_high + cy
	ret	$31,($26),1
.Lend1:	subq	$5,$3,$3
	cmpult	$5,$3,$5
	stq	$3,0($16)
	addq	$0,$5,$0
	ret	$31,($26),1

	.end	__mpn_submul_1
