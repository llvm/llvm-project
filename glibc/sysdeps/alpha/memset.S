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

/* Fill a block of memory with a character.  Optimized for the Alpha
   architecture:

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
	.type	memset, @function
	.globl	memset
	.usepv	memset, USEPV_PROF

	cfi_startproc

	/* On entry to this basic block:
	   t3 == loop counter
	   t4 == bytes in partial final word
	   a0 == possibly misaligned destination pointer
	   a1 == replicated source character  */

	.align 3
memset_loop:
	beq	t3, $tail
	blbc	t3, 0f		# skip single store if count even

	stq_u	a1, 0(a0)	# e0    : store one word
	subq	t3, 1, t3	# .. e1 :
	addq	a0, 8, a0	# e0    :
	beq	t3, $tail	# .. e1 :

0:	stq_u	a1, 0(a0)	# e0    : store two words
	subq	t3, 2, t3	# .. e1 :
	stq_u	a1, 8(a0)	# e0    :
	addq	a0, 16, a0	# .. e1 :
	bne	t3, 0b		# e1    :

$tail:	bne	t4, 1f		# is there a tail to do?
	ret			# no

	.align 3
1:	ldq_u	t0, 0(a0)	# e1    : yes, load original data
	mskql	a1, t4, t1	# .. e0 :
	mskqh	t0, t4, t0	# e0    :
	or	t0, t1, t0	# e1 (stall)
	stq_u	t0, 0(a0)	# e0    :
	ret			# .. e1 :

memset:
#ifdef PROF
	ldgp	gp, 0(pv)
	lda	AT, _mcount
	jsr	AT, (AT), _mcount
#endif

	and	a1, 0xff, a1	# e0    : zero extend input character
	mov	a0, v0		# .. e1 : move return value in place
	sll	a1, 8, t0	# e0    : begin replicating the char
	beq	a2, $done	# .. e1 : early exit for zero-length store
	or	t0, a1, a1	# e0    :
	and	a0, 7, t1	# .. e1 : dest misalignment
	sll	a1, 16, t0	# e0    :
	addq	a2, t1, a2	# .. e1 : add dest misalignment to count
	or	t0, a1, a1	# e0    :
	srl	a2, 3, t3	# .. e1 : loop = count >> 3
	sll	a1, 32, t0	# e0    :
	and	a2, 7, t4	# .. e1 : find number of bytes in tail
	or	t0, a1, a1	# e0    : character replication done

	beq	t1, memset_loop	# .. e1 : aligned head, jump right in

	ldq_u	t0, 0(a0)	# e1    : load original data to mask into
	mskqh	a1, a0, t1	# .. e0 :

	cmpult	a2, 8, t2	# e0    : is this a sub-word set?
	bne	t2, $oneq	# .. e1 (zdb)

	mskql	t0, a0, t0	# e0    : we span words.  finish this partial
	subq	t3, 1, t3	# .. e1 :
	addq	a0, 8, a0	# e0    :
	or	t0, t1, t0	# .. e1 :
	stq_u	t0, -8(a0)	# e0    :
	br 	memset_loop	# .. e1 :

	.align 3
$oneq:
	mskql	t1, a2, t1	# e0    : entire operation within one word
	mskql	t0, a0, t2	# e0    :
	mskqh	t0, a2, t3	# e0    :
	or	t1, t2, t0	# .. e1 :
	or	t0, t3, t0	# e1    :
	stq_u	t0, 0(a0)	# e0 (stall)

$done:	ret

	cfi_endproc
libc_hidden_builtin_def (memset)
