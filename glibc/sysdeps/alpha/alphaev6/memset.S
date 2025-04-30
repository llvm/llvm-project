/* Copyright (C) 2000-2021 Free Software Foundation, Inc.
   Contributed by Richard Henderson (rth@tamu.edu)
   EV6 optimized by Rick Gorton <rick.gorton@alpha-processor.com>.
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

#include <sysdep.h>

	.arch ev6
	.set noat
	.set noreorder

ENTRY(memset)
#ifdef PROF
	ldgp	gp, 0(pv)
	lda	AT, _mcount
	jsr	AT, (AT), _mcount
	.prologue 1
#else
	.prologue 0
#endif

	/*
	 * Serious stalling happens.  The only way to mitigate this is to
	 * undertake a major re-write to interleave the constant materialization
	 * with other parts of the fall-through code.  This is important, even
	 * though it makes maintenance tougher.
	 * Do this later.
	 */
	and	$17, 255, $1	# E : 00000000000000ch
	insbl	$17, 1, $2	# U : 000000000000ch00
	mov	$16, $0		# E : return value
	ble	$18, $end	# U : zero length requested?

	addq	$18, $16, $6	# E : max address to write to
	or	$1, $2, $17	# E : 000000000000chch
	insbl	$1, 2, $3	# U : 0000000000ch0000
	insbl	$1, 3, $4	# U : 00000000ch000000

	or	$3, $4, $3	# E : 00000000chch0000
	inswl	$17, 4, $5	# U : 0000chch00000000
	xor	$16, $6, $1	# E : will complete write be within one quadword?
	inswl	$17, 6, $2	# U : chch000000000000

	or	$17, $3, $17	# E : 00000000chchchch
	or	$2, $5, $2	# E : chchchch00000000
	bic	$1, 7, $1	# E : fit within a single quadword?
	and	$16, 7, $3	# E : Target addr misalignment

	or	$17, $2, $17	# E : chchchchchchchch
	beq	$1, $within_quad # U :
	nop			# E :
	beq	$3, $aligned	# U : target is 0mod8

	/*
	 * Target address is misaligned, and won't fit within a quadword.
	 */
	ldq_u	$4, 0($16)	# L : Fetch first partial
	mov	$16, $5		# E : Save the address
	insql	$17, $16, $2	# U : Insert new bytes
	subq	$3, 8, $3	# E : Invert (for addressing uses)

	addq	$18, $3, $18	# E : $18 is new count ($3 is negative)
	mskql	$4, $16, $4	# U : clear relevant parts of the quad
	subq	$16, $3, $16	# E : $16 is new aligned destination
	or	$2, $4, $1	# E : Final bytes

	nop
	stq_u	$1,0($5)	# L : Store result
	nop
	nop

	.align 4
$aligned:
	/*
	 * We are now guaranteed to be quad aligned, with at least
	 * one partial quad to write.
	 */

	sra	$18, 3, $3	# U : Number of remaining quads to write
	and	$18, 7, $18	# E : Number of trailing bytes to write
	mov	$16, $5		# E : Save dest address
	beq	$3, $no_quad	# U : tail stuff only

	/*
	 * It's worth the effort to unroll this and use wh64 if possible.
	 * At this point, entry values are:
	 * $16	Current destination address
	 * $5	A copy of $16
	 * $6	The max quadword address to write to
	 * $18	Number trailer bytes
	 * $3	Number quads to write
	 */

	and	$16, 0x3f, $2	# E : Forward work (only useful for unrolled loop)
	subq	$3, 16, $4	# E : Only try to unroll if > 128 bytes
	subq	$2, 0x40, $1	# E : bias counter (aligning stuff 0mod64)
	blt	$4, $loop	# U :

	/*
	 * We know we've got at least 16 quads, minimum of one trip
	 * through unrolled loop.  Do a quad at a time to get us 0mod64
	 * aligned.
	 */

	nop			# E :
	nop			# E :
	nop			# E :
	beq	$1, $bigalign	# U :

$alignmod64:
	stq	$17, 0($5)	# L :
	subq	$3, 1, $3	# E : For consistency later
	addq	$1, 8, $1	# E : Increment towards zero for alignment
	addq	$5, 8, $4	# E : Initial wh64 address (filler instruction)

	nop
	nop
	addq	$5, 8, $5	# E : Inc address
	blt	$1, $alignmod64 # U :

$bigalign:
	/*
	 * $3 - number quads left to go
	 * $5 - target address (aligned 0mod64)
	 * $17 - mask of stuff to store
	 * Scratch registers available: $7, $2, $4, $1
	 * We know that we'll be taking a minimum of one trip through.
	 * CWG Section 3.7.6: do not expect a sustained store rate of > 1/cycle
	 * Assumes the wh64 needs to be for 2 trips through the loop in the future.
	 * The wh64 is issued on for the starting destination address for trip +2
	 * through the loop, and if there are less than two trips left, the target
	 * address will be for the current trip.
	 */

$do_wh64:
	wh64	($4)		# L1 : memory subsystem write hint
	subq	$3, 24, $2	# E : For determining future wh64 addresses
	stq	$17, 0($5)	# L :
	nop			# E :

	addq	$5, 128, $4	# E : speculative target of next wh64
	stq	$17, 8($5)	# L :
	stq	$17, 16($5)	# L :
	addq	$5, 64, $7	# E : Fallback address for wh64 (== next trip addr)

	stq	$17, 24($5)	# L :
	stq	$17, 32($5)	# L :
	cmovlt	$2, $7, $4	# E : Latency 2, extra mapping cycle
	nop

	stq	$17, 40($5)	# L :
	stq	$17, 48($5)	# L :
	subq	$3, 16, $2	# E : Repeat the loop at least once more?
	nop

	stq	$17, 56($5)	# L :
	addq	$5, 64, $5	# E :
	subq	$3, 8, $3	# E :
	bge	$2, $do_wh64	# U :

	nop
	nop
	nop
	beq	$3, $no_quad	# U : Might have finished already

	.align 4
	/*
	 * Simple loop for trailing quadwords, or for small amounts
	 * of data (where we can't use an unrolled loop and wh64)
	 */
$loop:
	stq	$17, 0($5)	# L :
	subq	$3, 1, $3	# E : Decrement number quads left
	addq	$5, 8, $5	# E : Inc address
	bne	$3, $loop	# U : more?

$no_quad:
	/*
	 * Write 0..7 trailing bytes.
	 */
	nop			# E :
	beq	$18, $end	# U : All done?
	ldq	$7, 0($5)	# L :
	mskqh	$7, $6, $2	# U : Mask final quad

	insqh	$17, $6, $4	# U : New bits
	or	$2, $4, $1	# E : Put it all together
	stq	$1, 0($5)	# L : And back to memory
	ret	$31,($26),1	# L0 :

$within_quad:
	ldq_u	$1, 0($16)	# L :
	insql	$17, $16, $2	# U : New bits
	mskql	$1, $16, $4	# U : Clear old
	or	$2, $4, $2	# E : New result

	mskql	$2, $6, $4	# U :
	mskqh	$1, $6, $2	# U :
	or	$2, $4, $1	# E :
	stq_u	$1, 0($16)	# L :

$end:
	nop
	nop
	nop
	ret $31,($26),1		# L0 :

	END(memset)
libc_hidden_builtin_def (memset)
