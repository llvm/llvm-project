 # Alpha EV5 __mpn_rshift --

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
	.globl	__mpn_rshift
	.ent	__mpn_rshift
__mpn_rshift:
	.frame	$30,0,$26,0

	ldq	$4,0($17)	# load first limb
	subq	$31,$19,$20
	subq	$18,1,$18
	and	$18,4-1,$28	# number of limbs in first loop
	sll	$4,$20,$0	# compute function result

	beq	$28,.L0
	subq	$18,$28,$18

	.align	3
.Loop0:	ldq	$3,8($17)
	addq	$16,8,$16
	srl	$4,$19,$5
	addq	$17,8,$17
	subq	$28,1,$28
	sll	$3,$20,$6
	or	$3,$3,$4
	or	$5,$6,$8
	stq	$8,-8($16)
	bne	$28,.Loop0

.L0:	srl	$4,$19,$24
	beq	$18,.Lend
 # warm up phase 1
	ldq	$1,8($17)
	subq	$18,4,$18
	ldq	$2,16($17)
	ldq	$3,24($17)
	ldq	$4,32($17)
	beq	$18,.Lend1
 # warm up phase 2
	sll	$1,$20,$7
	srl	$1,$19,$21
	sll	$2,$20,$8
	ldq	$1,40($17)
	srl	$2,$19,$22
	ldq	$2,48($17)
	sll	$3,$20,$5
	or	$7,$24,$7
	srl	$3,$19,$23
	or	$8,$21,$8
	sll	$4,$20,$6
	ldq	$3,56($17)
	srl	$4,$19,$24
	ldq	$4,64($17)
	subq	$18,4,$18
	beq	$18,.Lend2
	.align  4
 # main loop
.Loop:	stq	$7,0($16)
	or	$5,$22,$5
	stq	$8,8($16)
	or	$6,$23,$6

	sll	$1,$20,$7
	subq	$18,4,$18
	srl	$1,$19,$21
	unop	# ldq	$31,-96($17)

	sll	$2,$20,$8
	ldq	$1,72($17)
	srl	$2,$19,$22
	ldq	$2,80($17)

	stq	$5,16($16)
	or	$7,$24,$7
	stq	$6,24($16)
	or	$8,$21,$8

	sll	$3,$20,$5
	unop	# ldq	$31,-96($17)
	srl	$3,$19,$23
	addq	$16,32,$16

	sll	$4,$20,$6
	ldq	$3,88($17)
	srl	$4,$19,$24
	ldq	$4,96($17)

	addq	$17,32,$17
	bne	$18,.Loop
 # cool down phase 2/1
.Lend2:	stq	$7,0($16)
	or	$5,$22,$5
	stq	$8,8($16)
	or	$6,$23,$6
	sll	$1,$20,$7
	srl	$1,$19,$21
	sll	$2,$20,$8
	srl	$2,$19,$22
	stq	$5,16($16)
	or	$7,$24,$7
	stq	$6,24($16)
	or	$8,$21,$8
	sll	$3,$20,$5
	srl	$3,$19,$23
	sll	$4,$20,$6
	srl	$4,$19,$24
 # cool down phase 2/2
	stq	$7,32($16)
	or	$5,$22,$5
	stq	$8,40($16)
	or	$6,$23,$6
	stq	$5,48($16)
	stq	$6,56($16)
 # cool down phase 2/3
	stq	$24,64($16)
	ret	$31,($26),1

 # cool down phase 1/1
.Lend1:	sll	$1,$20,$7
	srl	$1,$19,$21
	sll	$2,$20,$8
	srl	$2,$19,$22
	sll	$3,$20,$5
	or	$7,$24,$7
	srl	$3,$19,$23
	or	$8,$21,$8
	sll	$4,$20,$6
	srl	$4,$19,$24
 # cool down phase 1/2
	stq	$7,0($16)
	or	$5,$22,$5
	stq	$8,8($16)
	or	$6,$23,$6
	stq	$5,16($16)
	stq	$6,24($16)
	stq	$24,32($16)
	ret	$31,($26),1

.Lend:	stq	$24,0($16)
	ret	$31,($26),1
	.end	__mpn_rshift
