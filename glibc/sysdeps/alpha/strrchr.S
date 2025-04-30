/* Copyright (C) 1996-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library.  If not, see
   <https://www.gnu.org/licenses/>.  */

/* Return the address of the last occurrence of a given character
   within a null-terminated string, or null if it is not found.

   This is generally scheduled for the EV5 (got to look out for my own
   interests :-), but with EV4 needs in mind.  There are, in fact, fewer
   stalls on the EV4 than there are on the EV5.
*/

#include <sysdep.h>

	.set noreorder
	.set noat

ENTRY(strrchr)
#ifdef PROF
	ldgp	gp, 0(pv)
	lda	AT, _mcount
	jsr	AT, (AT), _mcount
	.prologue 1
#else
	.prologue 0
#endif

	and	a1, 0xff, a1	# e0    : zero extend our test character
	mov	zero, t6	# .. e1 : t6 is last match aligned addr
	sll	a1, 8, t5	# e0    : replicate our test character
	mov	zero, t7	# .. e1 : t7 is last match byte compare mask
	or	t5, a1, a1	# e0    :
	ldq_u   t0, 0(a0)	# .. e1 : load first quadword
	sll	a1, 16, t5	# e0    :
	andnot  a0, 7, v0	# .. e1 : align source addr
	or	t5, a1, a1	# e0    :
	lda	t4, -1		# .. e1 : build garbage mask
	sll	a1, 32, t5	# e0    :
	cmpbge  zero, t0, t1	# .. e1 : bits set iff byte == zero
	mskqh	t4, a0, t4	# e0    :
	or	t5, a1, a1	# .. e1 : character replication complete
	xor	t0, a1, t2	# e0    : make bytes == c zero
	cmpbge	zero, t4, t4	# .. e1 : bits set iff byte is garbage
	cmpbge  zero, t2, t3	# e0    : bits set iff byte == c
	andnot	t1, t4, t1	# .. e1 : clear garbage from null test
	andnot	t3, t4, t3	# e0    : clear garbage from char test
	bne	t1, $eos	# .. e1 : did we already hit the terminator?

	/* Character search main loop */
$loop:
	ldq	t0, 8(v0)	# e0    : load next quadword
	cmovne	t3, v0, t6	# .. e1 : save previous comparisons match
	cmovne	t3, t3, t7	# e0    :
	addq	v0, 8, v0	# .. e1 :
	xor	t0, a1, t2	# e0    :
	cmpbge	zero, t0, t1	# .. e1 : bits set iff byte == zero
	cmpbge	zero, t2, t3	# e0    : bits set iff byte == c
	beq	t1, $loop	# .. e1 : if we havnt seen a null, loop

	/* Mask out character matches after terminator */
$eos:
	negq	t1, t4		# e0    : isolate first null byte match
	and	t1, t4, t4	# e1    :
	subq	t4, 1, t5	# e0    : build a mask of the bytes upto...
	or	t4, t5, t4	# e1    : ... and including the null

	and	t3, t4, t3	# e0    : mask out char matches after null
	cmovne	t3, t3, t7	# .. e1 : save it, if match found
	cmovne	t3, v0, t6	# e0    :

	/* Locate the address of the last matched character */

	/* Retain the early exit for the ev4 -- the ev5 mispredict penalty
	   is 5 cycles -- the same as just falling through.  */
	beq	t7, $retnull	# .. e1 :

	and	t7, 0xf0, t2	# e0    : binary search for the high bit set
	cmovne	t2, t2, t7	# .. e1 (zdb)
	cmovne	t2, 4, t2	# e0    :
	and	t7, 0xcc, t1	# .. e1 :
	cmovne	t1, t1, t7	# e0    :
	cmovne	t1, 2, t1	# .. e1 :
	and	t7, 0xaa, t0	# e0    :
	cmovne	t0, 1, t0	# .. e1 (zdb)
	addq	t2, t1, t1	# e0    :
	addq	t6, t0, v0	# .. e1 : add our aligned base ptr to the mix
	addq	v0, t1, v0	# e0    :
	ret			# .. e1 :

$retnull:
	mov	zero, v0	# e0    :
	ret			# .. e1 :

	END(strrchr)

weak_alias (strrchr, rindex)
libc_hidden_builtin_def (strrchr)
