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

/* Bytewise compare two null-terminated strings of length no longer than N.  */

#include <sysdep.h>

	.set noat
	.set noreorder

/* EV6 only predicts one branch per octaword.  We'll use these to push
   subsequent branches back to the next bundle.  This will generally add
   a fetch+decode cycle to older machines, so skip in that case.  */
#ifdef __alpha_fix__
# define ev6_unop	unop
#else
# define ev6_unop
#endif

	.text

ENTRY(strncmp)
#ifdef PROF
	ldgp	gp, 0(pv)
	lda	AT, _mcount
	jsr	AT, (AT), _mcount
	.prologue 1
#else
	.prologue 0
#endif

	xor	a0, a1, t2	# are s1 and s2 co-aligned?
	beq	a2, $zerolength
	ldq_u	t0, 0(a0)	# load asap to give cache time to catch up
	ldq_u	t1, 0(a1)
	lda	t3, -1
	and	t2, 7, t2
	srl	t3, 1, t6
	and	a0, 7, t4	# find s1 misalignment
	and	a1, 7, t5	# find s2 misalignment
	cmovlt	a2, t6, a2	# bound neg count to LONG_MAX
	addq	a1, a2, a3	# s2+count
	addq	a2, t4, a2	# bias count by s1 misalignment
	and	a2, 7, t10	# ofs of last byte in s1 last word
	srl	a2, 3, a2	# remaining full words in s1 count
	bne	t2, $unaligned

	/* On entry to this basic block:
	   t0 == the first word of s1.
	   t1 == the first word of s2.
	   t3 == -1.  */
$aligned:
	mskqh	t3, a1, t8	# mask off leading garbage
	ornot	t1, t8, t1
	ornot	t0, t8, t0
	cmpbge	zero, t1, t7	# bits set iff null found
	beq	a2, $eoc	# check end of count
	bne	t7, $eos
	beq	t10, $ant_loop

	/* Aligned compare main loop.
	   On entry to this basic block:
	   t0 == an s1 word.
	   t1 == an s2 word not containing a null.  */

	.align 4
$a_loop:
	xor	t0, t1, t2	# e0	:
	bne	t2, $wordcmp	# .. e1 (zdb)
	ldq_u	t1, 8(a1)	# e0    :
	ldq_u	t0, 8(a0)	# .. e1 :

	subq	a2, 1, a2	# e0    :
	addq	a1, 8, a1	# .. e1 :
	addq	a0, 8, a0	# e0    :
	beq	a2, $eoc	# .. e1 :

	cmpbge	zero, t1, t7	# e0    :
	beq	t7, $a_loop	# .. e1 :

	br	$eos

	/* Alternate aligned compare loop, for when there's no trailing
	   bytes on the count.  We have to avoid reading too much data.  */
	.align 4
$ant_loop:
	xor	t0, t1, t2	# e0	:
	ev6_unop
	ev6_unop
	bne	t2, $wordcmp	# .. e1 (zdb)

	subq	a2, 1, a2	# e0    :
	beq	a2, $zerolength	# .. e1 :
	ldq_u	t1, 8(a1)	# e0    :
	ldq_u	t0, 8(a0)	# .. e1 :

	addq	a1, 8, a1	# e0    :
	addq	a0, 8, a0	# .. e1 :
	cmpbge	zero, t1, t7	# e0    :
	beq	t7, $ant_loop	# .. e1 :

	br	$eos

	/* The two strings are not co-aligned.  Align s1 and cope.  */
	/* On entry to this basic block:
	   t0 == the first word of s1.
	   t1 == the first word of s2.
	   t3 == -1.
	   t4 == misalignment of s1.
	   t5 == misalignment of s2.
	  t10 == misalignment of s1 end.  */
	.align	4
$unaligned:
	/* If s1 misalignment is larger than s2 misalignment, we need
	   extra startup checks to avoid SEGV.  */
	subq	a1, t4, a1	# adjust s2 for s1 misalignment
	cmpult	t4, t5, t9
	subq	a3, 1, a3	# last byte of s2
	bic	a1, 7, t8
	mskqh	t3, t5, t7	# mask garbage in s2
	subq	a3, t8, a3
	ornot	t1, t7, t7
	srl	a3, 3, a3	# remaining full words in s2 count
	beq	t9, $u_head

	/* Failing that, we need to look for both eos and eoc within the
	   first word of s2.  If we find either, we can continue by
	   pretending that the next word of s2 is all zeros.  */
	lda	t2, 0		# next = zero
	cmpeq	a3, 0, t8	# eoc in the first word of s2?
	cmpbge	zero, t7, t7	# eos in the first word of s2?
	or	t7, t8, t8
	bne	t8, $u_head_nl

	/* We know just enough now to be able to assemble the first
	   full word of s2.  We can still find a zero at the end of it.

	   On entry to this basic block:
	   t0 == first word of s1
	   t1 == first partial word of s2.
	   t3 == -1.
	   t10 == ofs of last byte in s1 last word.
	   t11 == ofs of last byte in s2 last word.  */
$u_head:
	ldq_u	t2, 8(a1)	# load second partial s2 word
	subq	a3, 1, a3
$u_head_nl:
	extql	t1, a1, t1	# create first s2 word
	mskqh	t3, a0, t8
	extqh	t2, a1, t4
	ornot	t0, t8, t0	# kill s1 garbage
	or	t1, t4, t1	# s2 word now complete
	cmpbge	zero, t0, t7	# find eos in first s1 word
	ornot	t1, t8, t1	# kill s2 garbage
	beq	a2, $eoc
	subq	a2, 1, a2
	bne	t7, $eos
	mskql	t3, a1, t8	# mask out s2[1] bits we have seen
	xor	t0, t1, t4	# compare aligned words
	or	t2, t8, t8
	bne	t4, $wordcmp
	cmpbge	zero, t8, t7	# eos in high bits of s2[1]?
	cmpeq	a3, 0, t8	# eoc in s2[1]?
	or	t7, t8, t7
	bne	t7, $u_final

	/* Unaligned copy main loop.  In order to avoid reading too much,
	   the loop is structured to detect zeros in aligned words from s2.
	   This has, unfortunately, effectively pulled half of a loop
	   iteration out into the head and half into the tail, but it does
	   prevent nastiness from accumulating in the very thing we want
	   to run as fast as possible.

	   On entry to this basic block:
	   t2 == the unshifted low-bits from the next s2 word.
	   t10 == ofs of last byte in s1 last word.
	   t11 == ofs of last byte in s2 last word.  */
	.align 4
$u_loop:
	extql	t2, a1, t3	# e0    :
	ldq_u	t2, 16(a1)	# .. e1 : load next s2 high bits
	ldq_u	t0, 8(a0)	# e0    : load next s1 word
	addq	a1, 8, a1	# .. e1 :

	addq	a0, 8, a0	# e0    :
	subq	a3, 1, a3	# .. e1 :
	extqh	t2, a1, t1	# e0    :
	cmpbge	zero, t0, t7	# .. e1 : eos in current s1 word

	or	t1, t3, t1	# e0    :
	beq	a2, $eoc	# .. e1 : eoc in current s1 word
	subq	a2, 1, a2	# e0    :
	cmpbge	zero, t2, t4	# .. e1 : eos in s2[1]

	xor	t0, t1, t3	# e0    : compare the words
	ev6_unop
	ev6_unop
	bne	t7, $eos	# .. e1 :

	cmpeq	a3, 0, t5	# e0    : eoc in s2[1]
	ev6_unop
	ev6_unop
	bne	t3, $wordcmp	# .. e1 :

	or	t4, t5, t4	# e0    : eos or eoc in s2[1].
	beq	t4, $u_loop	# .. e1 (zdb)

	/* We've found a zero in the low bits of the last s2 word.  Get
	   the next s1 word and align them.  */
	.align 3
$u_final:
	ldq_u	t0, 8(a0)
	extql	t2, a1, t1
	cmpbge	zero, t1, t7
	bne	a2, $eos

	/* We've hit end of count.  Zero everything after the count
	   and compare whats left.  */
	.align 3
$eoc:
	mskql	t0, t10, t0
	mskql	t1, t10, t1
	cmpbge	zero, t1, t7

	/* We've found a zero somewhere in a word we just read.
	   On entry to this basic block:
	   t0 == s1 word
	   t1 == s2 word
	   t7 == cmpbge mask containing the zero.  */
	.align 3
$eos:
	negq	t7, t6		# create bytemask of valid data
	and	t6, t7, t8
	subq	t8, 1, t6
	or	t6, t8, t7
	zapnot	t0, t7, t0	# kill the garbage
	zapnot	t1, t7, t1
	xor	t0, t1, v0	# ... and compare
	beq	v0, $done

	/* Here we have two differing co-aligned words in t0 & t1.
	   Bytewise compare them and return (t0 > t1 ? 1 : -1).  */
	.align 3
$wordcmp:
	cmpbge	t0, t1, t2	# comparison yields bit mask of ge
	cmpbge	t1, t0, t3
	xor	t2, t3, t0	# bits set iff t0/t1 bytes differ
	negq	t0, t1		# clear all but least bit
	and	t0, t1, t0
	lda	v0, -1
	and	t0, t2, t1	# was bit set in t0 > t1?
	cmovne	t1, 1, v0
$done:
	ret

	.align 3
$zerolength:
	clr	v0
	ret

	END(strncmp)
libc_hidden_builtin_def (strncmp)
