 # Alpha EV5 __mpn_lshift --

 # Copyright (C) 1994-2021 Free Software Foundation, Inc.

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
 # cnt		r19

 # This code runs at 3.25 cycles/limb on the EV5.

	.set	noreorder
	.set	noat
.text
	.align	3
	.globl	__mpn_lshift
	.ent	__mpn_lshift
__mpn_lshift:
	.frame	$30,0,$26,0

	s8addq	$18,$17,$17	# make r17 point at end of s1
	ldq	$4,-8($17)	# load first limb
	subq	$31,$19,$20
	s8addq	$18,$16,$16	# make r16 point at end of RES
	subq	$18,1,$18
	and	$18,4-1,$28	# number of limbs in first loop
	srl	$4,$20,$0	# compute function result

	beq	$28,.L0
	subq	$18,$28,$18

	.align	3
.Loop0:	ldq	$3,-16($17)
	subq	$16,8,$16
	sll	$4,$19,$5
	subq	$17,8,$17
	subq	$28,1,$28
	srl	$3,$20,$6
	or	$3,$3,$4
	or	$5,$6,$8
	stq	$8,0($16)
	bne	$28,.Loop0

.L0:	sll	$4,$19,$24
	beq	$18,.Lend
 # warm up phase 1
	ldq	$1,-16($17)
	subq	$18,4,$18
	ldq	$2,-24($17)
	ldq	$3,-32($17)
	ldq	$4,-40($17)
	beq	$18,.Lend1
 # warm up phase 2
	srl	$1,$20,$7
	sll	$1,$19,$21
	srl	$2,$20,$8
	ldq	$1,-48($17)
	sll	$2,$19,$22
	ldq	$2,-56($17)
	srl	$3,$20,$5
	or	$7,$24,$7
	sll	$3,$19,$23
	or	$8,$21,$8
	srl	$4,$20,$6
	ldq	$3,-64($17)
	sll	$4,$19,$24
	ldq	$4,-72($17)
	subq	$18,4,$18
	beq	$18,.Lend2
	.align  4
 # main loop
.Loop:	stq	$7,-8($16)
	or	$5,$22,$5
	stq	$8,-16($16)
	or	$6,$23,$6

	srl	$1,$20,$7
	subq	$18,4,$18
	sll	$1,$19,$21
	unop	# ldq	$31,-96($17)

	srl	$2,$20,$8
	ldq	$1,-80($17)
	sll	$2,$19,$22
	ldq	$2,-88($17)

	stq	$5,-24($16)
	or	$7,$24,$7
	stq	$6,-32($16)
	or	$8,$21,$8

	srl	$3,$20,$5
	unop	# ldq	$31,-96($17)
	sll	$3,$19,$23
	subq	$16,32,$16

	srl	$4,$20,$6
	ldq	$3,-96($17)
	sll	$4,$19,$24
	ldq	$4,-104($17)

	subq	$17,32,$17
	bne	$18,.Loop
 # cool down phase 2/1
.Lend2:	stq	$7,-8($16)
	or	$5,$22,$5
	stq	$8,-16($16)
	or	$6,$23,$6
	srl	$1,$20,$7
	sll	$1,$19,$21
	srl	$2,$20,$8
	sll	$2,$19,$22
	stq	$5,-24($16)
	or	$7,$24,$7
	stq	$6,-32($16)
	or	$8,$21,$8
	srl	$3,$20,$5
	sll	$3,$19,$23
	srl	$4,$20,$6
	sll	$4,$19,$24
 # cool down phase 2/2
	stq	$7,-40($16)
	or	$5,$22,$5
	stq	$8,-48($16)
	or	$6,$23,$6
	stq	$5,-56($16)
	stq	$6,-64($16)
 # cool down phase 2/3
	stq	$24,-72($16)
	ret	$31,($26),1

 # cool down phase 1/1
.Lend1:	srl	$1,$20,$7
	sll	$1,$19,$21
	srl	$2,$20,$8
	sll	$2,$19,$22
	srl	$3,$20,$5
	or	$7,$24,$7
	sll	$3,$19,$23
	or	$8,$21,$8
	srl	$4,$20,$6
	sll	$4,$19,$24
 # cool down phase 1/2
	stq	$7,-8($16)
	or	$5,$22,$5
	stq	$8,-16($16)
	or	$6,$23,$6
	stq	$5,-24($16)
	stq	$6,-32($16)
	stq	$24,-40($16)
	ret	$31,($26),1

.Lend:	stq	$24,-8($16)
	ret	$31,($26),1
	.end	__mpn_lshift
