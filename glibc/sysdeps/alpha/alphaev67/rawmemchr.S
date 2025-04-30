/* Copyright (C) 2000-2021 Free Software Foundation, Inc.
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

/* Return pointer to first occurrence of CH in STR.  */

#include <sysdep.h>

	.arch ev6
	.set noreorder
	.set noat

ENTRY(__rawmemchr)
#ifdef PROF
	ldgp	gp, 0(pv)
	lda	AT, _mcount
	jsr	AT, (AT), _mcount
	.prologue 1
#else
	.prologue 0
#endif

	ldq_u   t0, 0(a0)	# L : load first quadword Latency=3
	and	a1, 0xff, t3	# E : 00000000000000ch
	insbl	a1, 1, t5	# U : 000000000000ch00
	insbl	a1, 7, a2	# U : ch00000000000000

	insbl	t3, 6, a3	# U : 00ch000000000000
	or	t5, t3, a1	# E : 000000000000chch
	andnot  a0, 7, v0	# E : align our loop pointer
	lda	t4, -1		# E : build garbage mask

	mskqh	t4, a0, t4	# U : only want relevant part of first quad
	or	a2, a3, a2	# E : chch000000000000
	inswl	a1, 2, t5	# E : 00000000chch0000
	inswl	a1, 4, a3	# E : 0000chch00000000

	or	a1, a2, a1	# E : chch00000000chch
	or	a3, t5, t5	# E : 0000chchchch0000
	cmpbge	zero, t4, t4	# E : bits set iff byte is garbage
	nop			# E :

	/* This quad is _very_ serialized.  Lots of stalling happens */
	or	t5, a1, a1	# E : chchchchchchchch
	xor	t0, a1, t1	# E : make bytes == c zero
	cmpbge  zero, t1, t0	# E : bits set iff byte == c
	andnot	t0, t4, t0	# E : clear garbage bits

	cttz	t0, a2		# U0 : speculative (in case we get a match)
	nop			# E :
	nop			# E :
	bne	t0, $found	# U :

	/*
	 * Yuk.  This loop is going to stall like crazy waiting for the
	 * data to be loaded.  Not much can be done about it unless it's
	 * unrolled multiple times, which is generally unsafe.
	 */
$loop:
	ldq	t0, 8(v0)	# L : Latency=3
	addq	v0, 8, v0	# E :
	xor	t0, a1, t1	# E :
	cmpbge	zero, t1, t0	# E : bits set iff byte == c

	cttz	t0, a2		# U0 : speculative (in case we get a match)
	nop			# E :
	nop			# E :
	beq	t0, $loop	# U :

$found:
	negq    t0, t1		# E : clear all but least set bit
	and     t0, t1, t0	# E :
	addq	v0, a2, v0	# E : Add in the bit number from above
	ret			# L0 :

	END(__rawmemchr)

libc_hidden_def (__rawmemchr)
weak_alias (__rawmemchr, rawmemchr)
