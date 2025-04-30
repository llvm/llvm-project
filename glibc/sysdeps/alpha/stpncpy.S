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

/* Copy no more than COUNT bytes of the null-terminated string from
   SRC to DST.  If SRC does not cover all of COUNT, the balance is
   zeroed.  Return the address of the terminating null in DEST, if
   any, else DEST + COUNT.  */

#include <sysdep.h>

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

	beq	a2, $zerocount
	jsr	t9, __stxncpy	# do the work of the copy

	and	t8, 0xf0, t3	# binary search for byte offset of the
	and	t8, 0xcc, t2	# last byte written.
	and	t8, 0xaa, t1
	andnot	a0, 7, v0
	cmovne	t3, 4, t3
	cmovne	t2, 2, t2
	cmovne	t1, 1, t1
	addq	v0, t3, v0
	addq	t1, t2, t1
	addq	v0, t1, v0

	bne	a2, $multiword	# do we have full words left?

	.align 3
	zapnot	t0, t8, t4	# e0    : was last byte a null?
	subq	t8, 1, t2	# .. e1 :
	addq	v0, 1, t5	# e0    :
	subq	t10, 1, t3	# .. e1 :
	or	t2, t8, t2	# e0    : clear the bits between the last
	or	t3, t10, t3	# .. e1 : written byte and the last byte in
	andnot	t3, t2, t3	# e0    : COUNT
	cmovne	t4, t5, v0	# .. e1 : if last written wasnt null, inc v0
	zap	t0, t3, t0	# e0    :
	stq_u	t0, 0(a0)	# e1    :
	ret			# .. e1 :

	.align 3
$multiword:
	subq	t8, 1, t7	# e0    : clear the final bits in the prev
	or	t7, t8, t7	# e1    : word
	zapnot	t0, t7, t0	# e0    :
	subq	a2, 1, a2	# .. e1 :
	stq_u	t0, 0(a0)	# e0    :
	addq	a0, 8, a0	# .. e1 :

	beq	a2, 1f		# e1    :
	blbc	a2, 0f		# e1    :

	stq_u	zero, 0(a0)	# e0    : zero one word
	subq	a2, 1, a2	# .. e1 :
	addq	a0, 8, a0	# e0    :
	beq	a2, 1f		# .. e1 :

0:	stq_u	zero, 0(a0)	# e0    : zero two words
	subq	a2, 2, a2	# .. e1 :
	stq_u	zero, 8(a0)	# e0    :
	addq	a0, 16, a0	# .. e1 :
	bne	a2, 0b		# e1    :
	unop

1:	ldq_u	t0, 0(a0)	# e0    : clear the leading bits in the final
	subq	t10, 1, t7	# .. e1 : word
	or	t7, t10, t7	# e0    :
	zap	t0, t7, t0	# e1 (stall)
	stq_u	t0, 0(a0)	# e0    :
	ret			# .. e1 :

$zerocount:
	mov	a0, v0
	ret

	END(__stpncpy)

libc_hidden_def (__stpncpy)
weak_alias (__stpncpy, stpncpy)
