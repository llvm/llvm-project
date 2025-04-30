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

	or	$31,$31,$25		# clear cy
	subq	$19,4,$19		# decr loop cnt
	blt	$19,.Lend2		# if less than 4 limbs, goto 2nd loop
 # Start software pipeline for 1st loop
	ldq	$0,0($18)
	ldq	$1,8($18)
	ldq	$4,0($17)
	ldq	$5,8($17)
	addq	$17,32,$17		# update s1_ptr
	ldq	$2,16($18)
	subq	$4,$0,$20		# 1st main sub
	ldq	$3,24($18)
	subq	$19,4,$19		# decr loop cnt
	ldq	$6,-16($17)
	cmpult	$4,$20,$25		# compute cy from last sub
	ldq	$7,-8($17)
	addq	$1,$25,$28		# cy add
	addq	$18,32,$18		# update s2_ptr
	subq	$5,$28,$21		# 2nd main sub
	cmpult	$28,$25,$8		# compute cy from last add
	blt	$19,.Lend1		# if less than 4 limbs remain, jump
 # 1st loop handles groups of 4 limbs in a software pipeline
	.align	4
.Loop:	cmpult	$5,$21,$25		# compute cy from last add
	ldq	$0,0($18)
	or	$8,$25,$25		# combine cy from the two adds
	ldq	$1,8($18)
	addq	$2,$25,$28		# cy add
	ldq	$4,0($17)
	subq	$6,$28,$22		# 3rd main sub
	ldq	$5,8($17)
	cmpult	$28,$25,$8		# compute cy from last add
	cmpult	$6,$22,$25		# compute cy from last add
	stq	$20,0($16)
	or	$8,$25,$25		# combine cy from the two adds
	stq	$21,8($16)
	addq	$3,$25,$28		# cy add
	subq	$7,$28,$23		# 4th main sub
	cmpult	$28,$25,$8		# compute cy from last add
	cmpult	$7,$23,$25		# compute cy from last add
	addq	$17,32,$17		# update s1_ptr
	or	$8,$25,$25		# combine cy from the two adds
	addq	$16,32,$16		# update res_ptr
	addq	$0,$25,$28		# cy add
	ldq	$2,16($18)
	subq	$4,$28,$20		# 1st main sub
	ldq	$3,24($18)
	cmpult	$28,$25,$8		# compute cy from last add
	ldq	$6,-16($17)
	cmpult	$4,$20,$25		# compute cy from last add
	ldq	$7,-8($17)
	or	$8,$25,$25		# combine cy from the two adds
	subq	$19,4,$19		# decr loop cnt
	stq	$22,-16($16)
	addq	$1,$25,$28		# cy add
	stq	$23,-8($16)
	subq	$5,$28,$21		# 2nd main sub
	addq	$18,32,$18		# update s2_ptr
	cmpult	$28,$25,$8		# compute cy from last add
	bge	$19,.Loop
 # Finish software pipeline for 1st loop
.Lend1:	cmpult	$5,$21,$25		# compute cy from last add
	or	$8,$25,$25		# combine cy from the two adds
	addq	$2,$25,$28		# cy add
	subq	$6,$28,$22		# 3rd main sub
	cmpult	$28,$25,$8		# compute cy from last add
	cmpult	$6,$22,$25		# compute cy from last add
	stq	$20,0($16)
	or	$8,$25,$25		# combine cy from the two adds
	stq	$21,8($16)
	addq	$3,$25,$28		# cy add
	subq	$7,$28,$23		# 4th main sub
	cmpult	$28,$25,$8		# compute cy from last add
	cmpult	$7,$23,$25		# compute cy from last add
	or	$8,$25,$25		# combine cy from the two adds
	addq	$16,32,$16		# update res_ptr
	stq	$22,-16($16)
	stq	$23,-8($16)
.Lend2:	addq	$19,4,$19		# restore loop cnt
	beq	$19,.Lret
 # Start software pipeline for 2nd loop
	ldq	$0,0($18)
	ldq	$4,0($17)
	subq	$19,1,$19
	beq	$19,.Lend0
 # 2nd loop handles remaining 1-3 limbs
	.align	4
.Loop0:	addq	$0,$25,$28		# cy add
	ldq	$0,8($18)
	subq	$4,$28,$20		# main sub
	ldq	$1,8($17)
	addq	$18,8,$18
	cmpult	$28,$25,$8		# compute cy from last add
	addq	$17,8,$17
	stq	$20,0($16)
	cmpult	$4,$20,$25		# compute cy from last add
	subq	$19,1,$19		# decr loop cnt
	or	$8,$25,$25		# combine cy from the two adds
	addq	$16,8,$16
	or	$1,$31,$4
	bne	$19,.Loop0
.Lend0:	addq	$0,$25,$28		# cy add
	subq	$4,$28,$20		# main sub
	cmpult	$28,$25,$8		# compute cy from last add
	cmpult	$4,$20,$25		# compute cy from last add
	stq	$20,0($16)
	or	$8,$25,$25		# combine cy from the two adds

.Lret:	or	$25,$31,$0		# return cy
	ret	$31,($26),1
	.end	__mpn_sub_n
