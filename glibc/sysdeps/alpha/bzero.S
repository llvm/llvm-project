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

/* Fill a block of memory with zeros.  Optimized for the Alpha architecture:

   - memory accessed as aligned quadwords only
   - destination memory not read unless needed for good cache behaviour
   - basic blocks arranged to optimize branch prediction for full-quadword
     aligned memory blocks.
   - partial head and tail quadwords constructed with byte-mask instructions

   This is generally scheduled for the EV5 (got to look out for my own
   interests :-), but with EV4 needs in mind.  There *should* be no more
   stalls for the EV4 than there are for the EV5.
*/


#include <sysdep.h>

	.set noat
	.set noreorder

	.text
	.type	__bzero, @function
	.globl	__bzero
	.usepv	__bzero, USEPV_PROF

	cfi_startproc

	/* On entry to this basic block:
	   t3 == loop counter
	   t4 == bytes in partial final word
	   a0 == possibly misaligned destination pointer  */

	.align 3
bzero_loop:
	beq	t3, $tail	#
	blbc	t3, 0f		# skip single store if count even

	stq_u	zero, 0(a0)	# e0    : store one word
	subq	t3, 1, t3	# .. e1 :
	addq	a0, 8, a0	# e0    :
	beq	t3, $tail	# .. e1 :

0:	stq_u	zero, 0(a0)	# e0    : store two words
	subq	t3, 2, t3	# .. e1 :
	stq_u	zero, 8(a0)	# e0    :
	addq	a0, 16, a0	# .. e1 :
	bne	t3, 0b		# e1    :

$tail:	bne	t4, 1f		# is there a tail to do?
	ret			# no

1:	ldq_u	t0, 0(a0)	# yes, load original data
	mskqh	t0, t4, t0	#
	stq_u	t0, 0(a0)	#
	ret			#

__bzero:
#ifdef PROF
	ldgp	gp, 0(pv)
	lda	AT, _mcount
	jsr	AT, (AT), _mcount
#endif

	mov	a0, v0		# e0    : move return value in place
	beq	a1, $done	# .. e1 : early exit for zero-length store
	and	a0, 7, t1	# e0    :
	addq	a1, t1, a1	# e1    : add dest misalignment to count
	srl	a1, 3, t3	# e0    : loop = count >> 3
	and	a1, 7, t4	# .. e1 : find number of bytes in tail
	unop			#       :
	beq	t1, bzero_loop	# e1    : aligned head, jump right in

	ldq_u	t0, 0(a0)	# e0    : load original data to mask into
	cmpult	a1, 8, t2	# .. e1 : is this a sub-word set?
	bne	t2, $oneq	# e1    :

	mskql	t0, a0, t0	# e0    : we span words.  finish this partial
	subq	t3, 1, t3	# e0    :
	addq	a0, 8, a0	# .. e1 :
	stq_u	t0, -8(a0)	# e0    :
	br 	bzero_loop	# .. e1 :

	.align 3
$oneq:
	mskql	t0, a0, t2	# e0    :
	mskqh	t0, a1, t3	# e0    :
	or	t2, t3, t0	# e1    :
	stq_u	t0, 0(a0)	# e0    :

$done:	ret

	cfi_endproc
weak_alias (__bzero, bzero)
