/* Copyright (C) 1996-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Richard Henderson <rth@tamu.edu>, 1996.

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

/* Append a null-terminated string from SRC to DST.  */

#include <sysdep.h>

	.text

ENTRY(strcat)
	ldgp	gp, 0(pv)
#ifdef PROF
	.set noat
	lda	AT, _mcount
	jsr	AT, (AT), _mcount
	.set at
#endif
	.prologue 1

	mov	a0, v0		# set up return value

	/* Find the end of the string.  */

	ldq_u   t0, 0(a0)	# load first quadword (a0 may be misaligned)
	lda     t1, -1(zero)
	insqh   t1, a0, t1
	andnot  a0, 7, a0
	or      t1, t0, t0
	cmpbge  zero, t0, t1	# t1 <- bitmask: bit i == 1 <==> i-th byte == 0
	bne     t1, $found

$loop:	ldq     t0, 8(a0)
	addq    a0, 8, a0	# addr += 8
	cmpbge  zero, t0, t1
	beq     t1, $loop

$found:	negq    t1, t2		# clear all but least set bit
	and     t1, t2, t1

	and     t1, 0xf0, t2	# binary search for that set bit
	and	t1, 0xcc, t3
	and	t1, 0xaa, t4
	cmovne	t2, 4, t2
	cmovne	t3, 2, t3
	cmovne	t4, 1, t4
	addq	t2, t3, t2
	addq	a0, t4, a0
	addq	a0, t2, a0

	/* Now do the append.  */

	mov	ra, t9
	jmp	$31, __stxcpy

	END(strcat)
libc_hidden_builtin_def (strcat)
