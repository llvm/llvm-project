 # Alpha ev6 mpn_addmul_1 -- Multiply a limb vector with a limb and add
 # the result to a second limb vector.
 #
 #  Copyright (C) 2000-2021 Free Software Foundation, Inc.
 #
 #  This file is part of the GNU MP Library.
 #
 #  The GNU MP Library is free software; you can redistribute it and/or modify
 #  it under the terms of the GNU Lesser General Public License as published
 #  by the Free Software Foundation; either version 2.1 of the License, or (at
 #  your option) any later version.
 #
 #  The GNU MP Library is distributed in the hope that it will be useful, but
 #  WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
 #  or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public
 #  License for more details.
 #
 #  You should have received a copy of the GNU Lesser General Public License
 #  along with the GNU MP Library.  If not, see <https://www.gnu.org/licenses/>.

 #  INPUT PARAMETERS
 #  res_ptr	$16
 #  s1_ptr	$17
 #  size	$18
 #  s2_limb	$19
 #
 #  This code runs at 42 cycles/limb on EV4, 18 cycles/limb on EV5, and
 #  exactly 3.625 cycles/limb on EV6...
 #
 # This code was written in close cooperation with ev6 pipeline expert
 # Steve Root (root@toober.hlo.dec.com).  Any errors are tege's fault, though.
 #
 #   Register usages for unrolled loop:
 #	  0-3     mul's
 #	  4-7     acc's
 #	  8-15    mul results
 #	  20,21   carry's
 #	  22,23   save for stores
 #
 #   Sustains 8 mul-adds in 29 cycles in the unrolled inner loop.
 #
 #   The stores can issue a cycle late so we have paired no-op's to 'catch'
 #   them, so that further disturbance to the schedule is damped.
 #
 #   We couldn't pair the loads, because the entangled schedule of the
 #   carry's has to happen on one side {0} of the machine. Note, the total
 #   use of U0, and the total use of L0 (after attending to the stores).
 #   which is part of the reason why....
 #
 #   This is a great schedule for the d_cache, a poor schedule for the
 #   b_cache. The lockup on U0 means that any stall can't be recovered
 #   from. Consider a ldq in L1.  say that load gets stalled because it
 #   collides with a fill from the b_Cache. On the next cycle, this load
 #   gets priority. If first looks at L0, and goes there. The instruction
 #   we intended for L0 gets to look at L1, which is NOT where we want
 #   it. It either stalls 1, because it can't go in L0, or goes there, and
 #   causes a further instruction to stall.
 #
 #   So for b_cache, we're likely going to want to put one or more cycles
 #   back into the code! And, of course, put in prefetches. For the
 #   accumulator, lds, intent to modify.  For the multiplier, you might
 #   want ldq, evict next, if you're not wanting to use it again soon. Use
 #   256 ahead of present pointer value. At a place where we have an mt
 #   followed by a bookkeeping, put the bookkeeping in upper, and the
 #   prefetch into lower.
 #
 #   Note, the usage of physical registers per cycle is smoothed off, as
 #   much as possible.
 #
 #   Note, the ldq's and stq's are at the end of the quadpacks.  note, we'd
 #   like not to have a ldq or stq to preceded a conditional branch in a
 #   quadpack. The conditional branch moves the retire pointer one cycle
 #   later.
 #
 #   Optimization notes:
 #   Callee-saves regs: $9 $10 $11 $12 $13 $14 $15 $26 ?$27?
 #   Reserved regs:	 $29 $30 $31
 #   Free caller-saves regs in unrolled code: $24 $25 $28
 #   We should swap some of the callee-saves regs for some of the free
 #   caller-saves regs, saving some overhead cycles.
 #   Most importantly, we should write fast code for the 0-7 case.
 #   The code we use there are for the 21164, and runs at 7 cycles/limb
 #   on the 21264.  Should not be hard, if we write specialized code for
 #   1-7 limbs (the one for 0 limbs should be straightforward).  We then just
 #   need a jump table indexed by the low 3 bits of the count argument.

	.set	noreorder
	.set	noat
	.text

	.globl	__mpn_addmul_1
	.ent	__mpn_addmul_1
__mpn_addmul_1:
	.frame	$30,0,$26,0
	.prologue 0

	cmpult	$18,	8,	$1
	beq	$1,	$Large

	ldq	$2,	0($17)		# $2 = s1_limb
	addq	$17,	8,	$17	# s1_ptr++
	subq	$18,	1,	$18	# size--
	mulq	$2,	$19,	$3	# $3 = prod_low
	ldq	$5,	0($16)		# $5 = *res_ptr
	umulh	$2,	$19,	$0	# $0 = prod_high
	beq	$18,	$Lend0b		# jump if size was == 1
	ldq	$2,	0($17)		# $2 = s1_limb
	addq	$17,	8,	$17	# s1_ptr++
	subq	$18,	1,	$18	# size--
	addq	$5,	$3,	$3
	cmpult	$3,	$5,	$4
	stq	$3,	0($16)
	addq	$16,	8,	$16	# res_ptr++
	beq	$18,	$Lend0a		# jump if size was == 2

	.align 3
$Loop0:	mulq	$2,	$19,	$3	# $3 = prod_low
	ldq	$5,	0($16)		# $5 = *res_ptr
	addq	$4,	$0,	$0	# cy_limb = cy_limb + 'cy'
	subq	$18,	1,	$18	# size--
	umulh	$2,	$19,	$4	# $4 = cy_limb
	ldq	$2,	0($17)		# $2 = s1_limb
	addq	$17,	8,	$17	# s1_ptr++
	addq	$3,	$0,	$3	# $3 = cy_limb + prod_low
	cmpult	$3,	$0,	$0	# $0 = carry from (cy_limb + prod_low)
	addq	$5,	$3,	$3
	cmpult	$3,	$5,	$5
	stq	$3,	0($16)
	addq	$16,	8,	$16	# res_ptr++
	addq	$5,	$0,	$0	# combine carries
	bne	$18,	$Loop0
$Lend0a:
	mulq	$2,	$19,	$3	# $3 = prod_low
	ldq	$5,	0($16)		# $5 = *res_ptr
	addq	$4,	$0,	$0	# cy_limb = cy_limb + 'cy'
	umulh	$2,	$19,	$4	# $4 = cy_limb
	addq	$3,	$0,	$3	# $3 = cy_limb + prod_low
	cmpult	$3,	$0,	$0	# $0 = carry from (cy_limb + prod_low)
	addq	$5,	$3,	$3
	cmpult	$3,	$5,	$5
	stq	$3,	0($16)
	addq	$5,	$0,	$0	# combine carries
	addq	$4,	$0,	$0	# cy_limb = prod_high + cy
	ret	$31,	($26),	1
$Lend0b:
	addq	$5,	$3,	$3
	cmpult	$3,	$5,	$5
	stq	$3,	0($16)
	addq	$0,	$5,	$0
	ret	$31,	($26),	1

$Large:
	lda	$30,	-240($30)
	stq	$9,	8($30)
	stq	$10,	16($30)
	stq	$11,	24($30)
	stq	$12,	32($30)
	stq	$13,	40($30)
	stq	$14,	48($30)
	stq	$15,	56($30)

	and	$18,	7,	$20	# count for the first loop, 0-7
	srl	$18,	3,	$18	# count for unrolled loop
	bis	$31,	$31,	$0
	beq	$20,	$Lunroll
	ldq	$2,	0($17)		# $2 = s1_limb
	addq	$17,	8,	$17	# s1_ptr++
	subq	$20,	1,	$20	# size--
	mulq	$2,	$19,	$3	# $3 = prod_low
	ldq	$5,	0($16)		# $5 = *res_ptr
	umulh	$2,	$19,	$0	# $0 = prod_high
	beq	$20,	$Lend1b		# jump if size was == 1
	ldq	$2,	0($17)		# $2 = s1_limb
	addq	$17,	8,	$17	# s1_ptr++
	subq	$20,	1,	$20	# size--
	addq	$5,	$3,	$3
	cmpult	$3,	$5,	$4
	stq	$3,	0($16)
	addq	$16,	8,	$16	# res_ptr++
	beq	$20,	$Lend1a		# jump if size was == 2

	.align 3
$Loop1:	mulq	$2,	$19,	$3	# $3 = prod_low
	ldq	$5,	0($16)		# $5 = *res_ptr
	addq	$4,	$0,	$0	# cy_limb = cy_limb + 'cy'
	subq	$20,	1,	$20	# size--
	umulh	$2,	$19,	$4	# $4 = cy_limb
	ldq	$2,	0($17)		# $2 = s1_limb
	addq	$17,	8,	$17	# s1_ptr++
	addq	$3,	$0,	$3	# $3 = cy_limb + prod_low
	cmpult	$3,	$0,	$0	# $0 = carry from (cy_limb + prod_low)
	addq	$5,	$3,	$3
	cmpult	$3,	$5,	$5
	stq	$3,	0($16)
	addq	$16,	8,	$16	# res_ptr++
	addq	$5,	$0,	$0	# combine carries
	bne	$20,	$Loop1

$Lend1a:
	mulq	$2,	$19,	$3	# $3 = prod_low
	ldq	$5,	0($16)		# $5 = *res_ptr
	addq	$4,	$0,	$0	# cy_limb = cy_limb + 'cy'
	umulh	$2,	$19,	$4	# $4 = cy_limb
	addq	$3,	$0,	$3	# $3 = cy_limb + prod_low
	cmpult	$3,	$0,	$0	# $0 = carry from (cy_limb + prod_low)
	addq	$5,	$3,	$3
	cmpult	$3,	$5,	$5
	stq	$3,	0($16)
	addq	$16,	8,	$16	# res_ptr++
	addq	$5,	$0,	$0	# combine carries
	addq	$4,	$0,	$0	# cy_limb = prod_high + cy
	br	$31,	$Lunroll
$Lend1b:
	addq	$5,	$3,	$3
	cmpult	$3,	$5,	$5
	stq	$3,	0($16)
	addq	$16,	8,	$16	# res_ptr++
	addq	$0,	$5,	$0

$Lunroll:
	lda	$17,	-16($17)	# L1 bookkeeping
	lda	$16,	-16($16)	# L1 bookkeeping
	bis	$0,	$31,	$12

 # ____ UNROLLED LOOP SOFTWARE PIPELINE STARTUP ____

	ldq	$2,	16($17)		# L1
	ldq	$3,	24($17)		# L1
	lda	$18,	-1($18)		# L1 bookkeeping
	ldq	$6,	16($16)		# L1
	ldq	$7,	24($16)		# L1
	ldq	$0,	32($17)		# L1
	mulq	$19,	$2,	$13	# U1
	ldq	$1,	40($17)		# L1
	umulh	$19,	$2,	$14	# U1
	mulq	$19,	$3,	$15	# U1
	lda	$17,	64($17)		# L1 bookkeeping
	ldq	$4,	32($16)		# L1
	ldq	$5,	40($16)		# L1
	umulh	$19,	$3,	$8	# U1
	ldq	$2,	-16($17)	# L1
	mulq	$19,	$0,	$9	# U1
	ldq	$3,	-8($17)		# L1
	umulh	$19,	$0,	$10	# U1
	addq	$6,	$13,	$6	# L0 lo + acc
	mulq	$19,	$1,	$11	# U1
	cmpult	$6,	$13,	$20	# L0 lo add => carry
	lda	$16,	64($16)		# L1 bookkeeping
	addq	$6,	$12,	$22	# U0 hi add => answer
	cmpult	$22,	$12,	$21	# L0 hi add => carry
	addq	$14,	$20,	$14	# U0 hi mul + carry
	ldq	$6,	-16($16)	# L1
	addq	$7,	$15,	$23	# L0 lo + acc
	addq	$14,	$21,	$14	# U0 hi mul + carry
	ldq	$7,	-8($16)		# L1
	umulh	$19,	$1,	$12	# U1
	cmpult	$23,	$15,	$20	# L0 lo add => carry
	addq	$23,	$14,	$23	# U0 hi add => answer
	ldq	$0,	0($17)		# L1
	mulq	$19,	$2,	$13	# U1
	cmpult	$23,	$14,	$21	# L0 hi add => carry
	addq	$8,	$20,	$8	# U0 hi mul + carry
	ldq	$1,	8($17)		# L1
	umulh	$19,	$2,	$14	# U1
	addq	$4,	$9,	$4	# L0 lo + acc
	stq	$22,	-48($16)	# L0
	stq	$23,	-40($16)	# L1
	mulq	$19,	$3,	$15	# U1
	addq	$8,	$21,	$8	# U0 hi mul + carry
	cmpult	$4,	$9,	$20	# L0 lo add => carry
	addq	$4,	$8,	$22	# U0 hi add => answer
	ble	$18,	$Lend		# U1 bookkeeping

 # ____ MAIN UNROLLED LOOP ____
	.align 4
$Loop:
	bis	$31,	$31,	$31	# U1 mt
	cmpult	$22,	$8,	$21	# L0 hi add => carry
	addq	$10,	$20,	$10	# U0 hi mul + carry
	ldq	$4,	0($16)		# L1

	bis	$31,	$31,	$31	# U1 mt
	addq	$5,	$11,	$23	# L0 lo + acc
	addq	$10,	$21,	$10	# L0 hi mul + carry
	ldq	$5,	8($16)		# L1

	umulh	$19,	$3,	$8	# U1
	cmpult	$23,	$11,	$20	# L0 lo add => carry
	addq	$23,	$10,	$23	# U0 hi add => answer
	ldq	$2,	16($17)		# L1

	mulq	$19,	$0,	$9	# U1
	cmpult	$23,	$10,	$21	# L0 hi add => carry
	addq	$12,	$20,	$12	# U0 hi mul + carry
	ldq	$3,	24($17)		# L1

	umulh	$19,	$0,	$10	# U1
	addq	$6,	$13,	$6	# L0 lo + acc
	stq	$22,	-32($16)	# L0
	stq	$23,	-24($16)	# L1

	bis	$31,	$31,	$31	# L0 st slosh
	mulq	$19,	$1,	$11	# U1
	bis	$31,	$31,	$31	# L1 st slosh
	addq	$12,	$21,	$12	# U0 hi mul + carry

	cmpult	$6,	$13,	$20	# L0 lo add => carry
	bis	$31,	$31,	$31	# U1 mt
	lda	$18,	-1($18)		# L1 bookkeeping
	addq	$6,	$12,	$22	# U0 hi add => answer

	bis	$31,	$31,	$31	# U1 mt
	cmpult	$22,	$12,	$21	# L0 hi add => carry
	addq	$14,	$20,	$14	# U0 hi mul + carry
	ldq	$6,	16($16)		# L1

	bis	$31,	$31,	$31	# U1 mt
	addq	$7,	$15,	$23	# L0 lo + acc
	addq	$14,	$21,	$14	# U0 hi mul + carry
	ldq	$7,	24($16)		# L1

	umulh	$19,	$1,	$12	# U1
	cmpult	$23,	$15,	$20	# L0 lo add => carry
	addq	$23,	$14,	$23	# U0 hi add => answer
	ldq	$0,	32($17)		# L1

	mulq	$19,	$2,	$13	# U1
	cmpult	$23,	$14,	$21	# L0 hi add => carry
	addq	$8,	$20,	$8	# U0 hi mul + carry
	ldq	$1,	40($17)		# L1

	umulh	$19,	$2,	$14	# U1
	addq	$4,	$9,	$4	# U0 lo + acc
	stq	$22,	-16($16)	# L0
	stq	$23,	-8($16)		# L1

	bis	$31,	$31,	$31	# L0 st slosh
	mulq	$19,	$3,	$15	# U1
	bis	$31,	$31,	$31	# L1 st slosh
	addq	$8,	$21,	$8	# L0 hi mul + carry

	cmpult	$4,	$9,	$20	# L0 lo add => carry
	bis	$31,	$31,	$31	# U1 mt
	lda	$17,	64($17)		# L1 bookkeeping
	addq	$4,	$8,	$22	# U0 hi add => answer

	bis	$31,	$31,	$31	# U1 mt
	cmpult	$22,	$8,	$21	# L0 hi add => carry
	addq	$10,	$20,	$10	# U0 hi mul + carry
	ldq	$4,	32($16)		# L1

	bis	$31,	$31,	$31	# U1 mt
	addq	$5,	$11,	$23	# L0 lo + acc
	addq	$10,	$21,	$10	# L0 hi mul + carry
	ldq	$5,	40($16)		# L1

	umulh	$19,	$3,	$8	# U1
	cmpult	$23,	$11,	$20	# L0 lo add => carry
	addq	$23,	$10,	$23	# U0 hi add => answer
	ldq	$2,	-16($17)	# L1

	mulq	$19,	$0,	$9	# U1
	cmpult	$23,	$10,	$21	# L0 hi add => carry
	addq	$12,	$20,	$12	# U0 hi mul + carry
	ldq	$3,	-8($17)		# L1

	umulh	$19,	$0,	$10	# U1
	addq	$6,	$13,	$6	# L0 lo + acc
	stq	$22,	0($16)		# L0
	stq	$23,	8($16)		# L1

	bis	$31,	$31,	$31	# L0 st slosh
	mulq	$19,	$1,	$11	# U1
	bis	$31,	$31,	$31	# L1 st slosh
	addq	$12,	$21,	$12	# U0 hi mul + carry

	cmpult	$6,	$13,	$20	# L0 lo add => carry
	bis	$31,	$31,	$31	# U1 mt
	lda	$16,	64($16)		# L1 bookkeeping
	addq	$6,	$12,	$22	# U0 hi add => answer

	bis	$31,	$31,	$31	# U1 mt
	cmpult	$22,	$12,	$21	# L0 hi add => carry
	addq	$14,	$20,	$14	# U0 hi mul + carry
	ldq	$6,	-16($16)	# L1

	bis	$31,	$31,	$31	# U1 mt
	addq	$7,	$15,	$23	# L0 lo + acc
	addq	$14,	$21,	$14	# U0 hi mul + carry
	ldq	$7,	-8($16)		# L1

	umulh	$19,	$1,	$12	# U1
	cmpult	$23,	$15,	$20	# L0 lo add => carry
	addq	$23,	$14,	$23	# U0 hi add => answer
	ldq	$0,	0($17)		# L1

	mulq	$19,	$2,	$13	# U1
	cmpult	$23,	$14,	$21	# L0 hi add => carry
	addq	$8,	$20,	$8	# U0 hi mul + carry
	ldq	$1,	8($17)		# L1

	umulh	$19,	$2,	$14	# U1
	addq	$4,	$9,	$4	# L0 lo + acc
	stq	$22,	-48($16)	# L0
	stq	$23,	-40($16)	# L1

	bis	$31,	$31,	$31	# L0 st slosh
	mulq	$19,	$3,	$15	# U1
	bis	$31,	$31,	$31	# L1 st slosh
	addq	$8,	$21,	$8	# U0 hi mul + carry

	cmpult	$4,	$9,	$20	# L0 lo add => carry
	addq	$4,	$8,	$22	# U0 hi add => answer
	bis	$31,	$31,	$31	# L1 mt
	bgt	$18,	$Loop		# U1 bookkeeping

# ____ UNROLLED LOOP SOFTWARE PIPELINE FINISH ____
$Lend:
	cmpult	$22,	$8,	$21	# L0 hi add => carry
	addq	$10,	$20,	$10	# U0 hi mul + carry
	ldq	$4,	0($16)		# L1
	addq	$5,	$11,	$23	# L0 lo + acc
	addq	$10,	$21,	$10	# L0 hi mul + carry
	ldq	$5,	8($16)		# L1
	umulh	$19,	$3,	$8	# U1
	cmpult	$23,	$11,	$20	# L0 lo add => carry
	addq	$23,	$10,	$23	# U0 hi add => answer
	mulq	$19,	$0,	$9	# U1
	cmpult	$23,	$10,	$21	# L0 hi add => carry
	addq	$12,	$20,	$12	# U0 hi mul + carry
	umulh	$19,	$0,	$10	# U1
	addq	$6,	$13,	$6	# L0 lo + acc
	stq	$22,	-32($16)	# L0
	stq	$23,	-24($16)	# L1
	mulq	$19,	$1,	$11	# U1
	addq	$12,	$21,	$12	# U0 hi mul + carry
	cmpult	$6,	$13,	$20	# L0 lo add => carry
	addq	$6,	$12,	$22	# U0 hi add => answer
	cmpult	$22,	$12,	$21	# L0 hi add => carry
	addq	$14,	$20,	$14	# U0 hi mul + carry
	addq	$7,	$15,	$23	# L0 lo + acc
	addq	$14,	$21,	$14	# U0 hi mul + carry
	umulh	$19,	$1,	$12	# U1
	cmpult	$23,	$15,	$20	# L0 lo add => carry
	addq	$23,	$14,	$23	# U0 hi add => answer
	cmpult	$23,	$14,	$21	# L0 hi add => carry
	addq	$8,	$20,	$8	# U0 hi mul + carry
	addq	$4,	$9,	$4	# U0 lo + acc
	stq	$22,	-16($16)	# L0
	stq	$23,	-8($16)		# L1
	bis	$31,	$31,	$31	# L0 st slosh
	addq	$8,	$21,	$8	# L0 hi mul + carry
	cmpult	$4,	$9,	$20	# L0 lo add => carry
	addq	$4,	$8,	$22	# U0 hi add => answer
	cmpult	$22,	$8,	$21	# L0 hi add => carry
	addq	$10,	$20,	$10	# U0 hi mul + carry
	addq	$5,	$11,	$23	# L0 lo + acc
	addq	$10,	$21,	$10	# L0 hi mul + carry
	cmpult	$23,	$11,	$20	# L0 lo add => carry
	addq	$23,	$10,	$23	# U0 hi add => answer
	cmpult	$23,	$10,	$21	# L0 hi add => carry
	addq	$12,	$20,	$12	# U0 hi mul + carry
	stq	$22,	0($16)		# L0
	stq	$23,	8($16)		# L1
	addq	$12,	$21,	$0	# U0 hi mul + carry

	ldq	$9,	8($30)
	ldq	$10,	16($30)
	ldq	$11,	24($30)
	ldq	$12,	32($30)
	ldq	$13,	40($30)
	ldq	$14,	48($30)
	ldq	$15,	56($30)
	lda	$30,	240($30)
	ret	$31,	($26),	1

	.end	__mpn_addmul_1
