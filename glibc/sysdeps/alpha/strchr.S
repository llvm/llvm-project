/* Copyright (C) 1996-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Richard Henderson (rth@tamu.edu)

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

/* Return the address of a given character within a null-terminated
   string, or null if it is not found.

   This is generally scheduled for the EV5 (got to look out for my own
   interests :-), but with EV4 needs in mind.  There *should* be no more
   stalls for the EV4 than there are for the EV5.
*/

#include <sysdep.h>

	.set noreorder
	.set noat

ENTRY(strchr)
#ifdef PROF
	ldgp	gp, 0(pv)
	lda	AT, _mcount
	jsr	AT, (AT), _mcount
	.prologue 1
#else
	.prologue 0
#endif

	zapnot	a1, 1, a1	# e0    : zero extend the search character
	ldq_u   t0, 0(a0)	# .. e1 : load first quadword
	sll	a1, 8, t5	# e0    : replicate the search character
	andnot  a0, 7, v0	# .. e1 : align our loop pointer
	or	t5, a1, a1	# e0    :
	lda	t4, -1		# .. e1 : build garbage mask
	sll	a1, 16, t5	# e0    :
	cmpbge  zero, t0, t2	# .. e1 : bits set iff byte == zero
	mskqh	t4, a0, t4	# e0    :
	or	t5, a1, a1	# .. e1 :
	sll	a1, 32, t5	# e0    :
	cmpbge	zero, t4, t4	# .. e1 : bits set iff byte is garbage
	or	t5, a1, a1	# e0    :
	xor	t0, a1, t1	# .. e1 : make bytes == c zero
	cmpbge  zero, t1, t3	# e0    : bits set iff byte == c
	or	t2, t3, t0	# e1    : bits set iff char match or zero match
	andnot	t0, t4, t0	# e0    : clear garbage bits
	bne	t0, $found	# .. e1 (zdb)

$loop:	ldq	t0, 8(v0)	# e0    :
	addq	v0, 8, v0	# .. e1 :
	nop			# e0    :
	xor	t0, a1, t1	# .. e1 (ev5 data stall)
	cmpbge	zero, t0, t2	# e0    : bits set iff byte == 0
	cmpbge	zero, t1, t3	# .. e1 : bits set iff byte == c
	or	t2, t3, t0	# e0    :
	beq	t0, $loop	# .. e1 (zdb)

$found:	negq    t0, t1		# e0    : clear all but least set bit
	and     t0, t1, t0	# e1 (stall)

	and	t0, t3, t1	# e0    : bit set iff byte was the char
	beq	t1, $retnull	# .. e1 (zdb)

	and     t0, 0xf0, t2	# e0    : binary search for that set bit
	and	t0, 0xcc, t3	# .. e1 :
	and	t0, 0xaa, t4	# e0    :
	cmovne	t2, 4, t2	# .. e1 :
	cmovne	t3, 2, t3	# e0    :
	cmovne	t4, 1, t4	# .. e1 :
	addq	t2, t3, t2	# e0    :
	addq	v0, t4, v0	# .. e1 :
	addq	v0, t2, v0	# e0    :
	ret			# .. e1 :

$retnull:
	mov	zero, v0	# e0    :
	ret			# .. e1 :

	END(strchr)

weak_alias (strchr, index)
libc_hidden_builtin_def (strchr)
