/* Copyright (C) 1996-2021 Free Software Foundation, Inc.
   Contributed by Richard Henderson (rth@tamu.edu)
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

/* Copy no more than COUNT bytes of the null-terminated string from
   SRC to DST.  If SRC does not cover all of COUNT, the balance is
   zeroed.  */

#include <sysdep.h>

	.set noat
	.set noreorder

	.text

ENTRY(strncpy)
	ldgp	gp, 0(pv)
#ifdef PROF
	lda	AT, _mcount
	jsr	AT, (AT), _mcount
#endif
	.prologue 1

	mov	a0, v0		# set return value now
	beq	a2, $zerocount
	jsr	t9, __stxncpy	# do the work of the copy

	bne	a2, $multiword	# do we have full words left?

	.align 3
	subq	t8, 1, t2	# e0    : guess not
	subq	t10, 1, t3	# .. e1 :
	or	t2, t8, t2	# e0    : clear the bits between the last
	or	t3, t10, t3	# .. e1 : written byte and the last byte in
	andnot	t3, t2, t3	# e0    : COUNT
	zap	t0, t3, t0	# e1    :
	stq_u	t0, 0(a0)	# e0    :
	ret			# .. e1 :

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

$zerocount:
	ret			# .. e1 :

	END(strncpy)
libc_hidden_builtin_def (strncpy)
