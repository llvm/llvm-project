/* Copyright (C) 2000-2021 Free Software Foundation, Inc.
   Contributed by Richard Henderson (rth@redhat.com)
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

/* Copy no more than N bytes from SRC to DEST, returning the address of
   the terminating '\0' in DEST.  */

#include <sysdep.h>

	.arch ev6
	.set noat
	.set noreorder
	.text

ENTRY(__stpncpy)
	ldgp	gp, 0(pv)
#ifdef PROF
	lda	AT, _mcount
	jsr	AT, (AT), _mcount
#endif
	.prologue 1

	mov	a0, v0
	beq	a2, $zerocount

	.align 4
	nop
	nop
	jsr	t9, __stxncpy	# do the work of the copy

	cttz	t8, t4
	zapnot	t0, t8, t5
	andnot	a0, 7, a0
	bne	a2, $multiword	# do we have full words left?

	subq	t8, 1, t2
	subq	t10, 1, t3
	cmpult	zero, t5, t5
	addq	a0, t4, v0

	or	t2, t8, t2
	or	t3, t10, t3
	addq	v0, t5, v0
	andnot	t3, t2, t3

	zap	t0, t3, t0
	nop
	stq	t0, 0(a0)
	ret

$multiword:
	subq	t8, 1, t7	# clear the final bits in the prev word
	cmpult	zero, t5, t5
	or	t7, t8, t7
	zapnot	t0, t7, t0

	subq	a2, 1, a2
	stq	t0, 0(a0)
	addq	a0, 8, a1
	beq	a2, 1f		# loop over full words remaining

	nop
	nop
	nop
	blbc	a2, 0f

	stq	zero, 0(a1)
	subq	a2, 1, a2
	addq	a1, 8, a1
	beq	a2, 1f

0:	stq	zero, 0(a1)
	subq	a2, 2, a2
	nop
	nop

	stq	zero, 8(a1)
	addq	a1, 16, a1
	nop
	bne	a2, 0b

1:	ldq	t0, 0(a1)	# clear the leading bits in the final word
	subq	t10, 1, t7
	addq	a0, t4, v0
	nop

	or	t7, t10, t7
	addq	v0, t5, v0
	zap	t0, t7, t0
	stq	t0, 0(a1)

$zerocount:
	nop
	nop
	nop
	ret

	END(__stpncpy)

libc_hidden_def (__stpncpy)
weak_alias (__stpncpy, stpncpy)
