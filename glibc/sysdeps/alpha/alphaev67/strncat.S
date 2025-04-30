/* Copyright (C) 2000-2021 Free Software Foundation, Inc.
   Contributed by Richard Henderson <rth@tamu.edu>, 1996.
   EV67 optimized by Rick Gorton <rick.gorton@alpha-processor.com>.
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

/* Append no more than COUNT characters from the null-terminated string SRC
   to the null-terminated string DST.  Always null-terminate the new DST.  */

#include <sysdep.h>

	.arch ev6
	.set noreorder
	.text

ENTRY(strncat)
	ldgp	gp, 0(pv)
#ifdef PROF
	.set noat
	lda	AT, _mcount
	jsr	AT, (AT), _mcount
	.set at
#endif
	.prologue 1

	mov	a0, v0		# set up return value
	beq	a2, $zerocount	# U :
	/* Find the end of the string.  */
	ldq_u   t0, 0(a0)	# L : load first quadword (a0 may be misaligned)
	lda     t1, -1		# E :

	insqh   t1, v0, t1	# U :
	andnot  a0, 7, a0	# E :
	nop			# E :
	or      t1, t0, t0	# E :

	nop			# E :
	nop			# E :
	cmpbge  zero, t0, t1	# E : bits set iff byte == 0
	bne     t1, $found	# U :

$loop:	ldq     t0, 8(a0)	# L :
	addq    a0, 8, a0	# E :
	cmpbge  zero, t0, t1	# E :
	beq     t1, $loop	# U :

$found:	cttz	t1, t2		# U0 :
	addq	a0, t2, a0	# E :
	jsr	t9, __stxncpy	# L0 : Now do the append.

	/* Worry about the null termination.  */

	cttz	t10, t2		# U0: byte offset of end-of-count.
	bic	a0, 7, a0	# E : word align the last write address.
	zapnot	t0, t8, t1	# U : was last byte a null?
	nop			# E :

	bne	t1, 0f		# U :
	nop			# E :
	nop			# E :
	ret			# L0 :

0:	addq	t2, a0, a0	# E : address of end-of-count
	stb	zero, 1(a0)	# L :
	nop			# E :
	ret			# L0 :

$zerocount:
	nop			# E :
	nop			# E :
	nop			# E :
	ret			# L0 :

	END(strncat)
