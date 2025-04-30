/* Optimized memrchr implementation for PowerPC64/POWER7 using cmpb insn.
   Copyright (C) 2010-2021 Free Software Foundation, Inc.
   Contributed by Luis Machado <luisgpm@br.ibm.com>.
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
   License along with the GNU C Library; if not, see
   <https://www.gnu.org/licenses/>.  */

#include <sysdep.h>

/* int [r3] memrchr (char *s [r3], int byte [r4], int size [r5])  */

#ifndef MEMRCHR
# define MEMRCHR __memrchr
#endif
	.machine  power7
ENTRY_TOCLESS (MEMRCHR)
	CALL_MCOUNT 3
	add	r7,r3,r5      /* Calculate the last acceptable address.  */
	neg	r0,r7
	addi	r7,r7,-1
	mr	r10,r3
	clrrdi	r6,r7,7
	li	r9,3<<5
	dcbt	r9,r6,8       /* Stream hint, decreasing addresses.  */

	/* Replicate BYTE to doubleword.  */
	insrdi	r4,r4,8,48
	insrdi	r4,r4,16,32
	insrdi  r4,r4,32,0
	li	r6,-8
	li	r9,-1
	rlwinm	r0,r0,3,26,28 /* Calculate padding.  */
	clrrdi	r8,r7,3
	srd	r9,r9,r0
	cmpldi	r5,32
	clrrdi	r0,r10,3
	ble	L(small_range)

#ifdef __LITTLE_ENDIAN__
	ldx	r12,0,r8
#else
	ldbrx	r12,0,r8      /* Load reversed doubleword from memory.  */
#endif
	cmpb	r3,r12,r4     /* Check for BYTE in DWORD1.  */
	and	r3,r3,r9
	cmpldi	cr7,r3,0      /* If r3 == 0, no BYTEs have been found.  */
	bne	cr7,L(done)

	mtcrf   0x01,r8
	/* Are we now aligned to a quadword boundary?  If so, skip to
	   the main loop.  Otherwise, go through the alignment code.  */
	bf	28,L(loop_setup)

	/* Handle DWORD2 of pair.  */
#ifdef __LITTLE_ENDIAN__
	ldx	r12,r8,r6
#else
	ldbrx	r12,r8,r6
#endif
	addi	r8,r8,-8
	cmpb	r3,r12,r4
	cmpldi	cr7,r3,0
	bne	cr7,L(done)

L(loop_setup):
	/* The last dword we want to read in the loop below is the one
	   containing the first byte of the string, ie. the dword at
	   s & ~7, or r0.  The first dword read is at r8 - 8, we
	   read 2 * cnt dwords, so the last dword read will be at
	   r8 - 8 - 16 * cnt + 8.  Solving for cnt gives
	   cnt = (r8 - r0) / 16  */
	sub	r5,r8,r0
	addi	r8,r8,-8
	srdi	r9,r5,4       /* Number of loop iterations.  */
	mtctr	r9	      /* Setup the counter.  */

	/* Main loop to look for BYTE backwards in the string.
	   FIXME: Investigate whether 32 byte align helps with this
	   9 instruction loop.  */
	.align	5
L(loop):
	/* Load two doublewords, compare and merge in a
	   single register for speed.  This is an attempt
	   to speed up the byte-checking process for bigger strings.  */

#ifdef __LITTLE_ENDIAN__
	ldx	r12,0,r8
	ldx	r11,r8,r6
#else
	ldbrx	r12,0,r8
	ldbrx	r11,r8,r6
#endif
	cmpb	r3,r12,r4
	cmpb	r9,r11,r4
	or	r5,r9,r3      /* Merge everything in one doubleword.  */
	cmpldi	cr7,r5,0
	bne	cr7,L(found)
	addi	r8,r8,-16
	bdnz	L(loop)

	/* We may have one more word to read.  */
	cmpld	r8,r0
	bnelr

#ifdef __LITTLE_ENDIAN__
	ldx	r12,0,r8
#else
	ldbrx	r12,0,r8
#endif
	cmpb	r3,r12,r4
	cmpldi	cr7,r3,0
	bne	cr7,L(done)
	blr

	.align	4
L(found):
	/* OK, one (or both) of the dwords contains BYTE.  Check
	   the first dword.  */
	cmpldi	cr6,r3,0
	bne	cr6,L(done)

	/* BYTE must be in the second word.  Adjust the address
	   again and move the result of cmpb to r3 so we can calculate the
	   pointer.  */

	mr	r3,r9
	addi	r8,r8,-8

	/* r3 has the output of the cmpb instruction, that is, it contains
	   0xff in the same position as BYTE in the original
	   word from the string.  Use that to calculate the pointer.
	   We need to make sure BYTE is *before* the end of the
	   range.  */
L(done):
	cntlzd	r9,r3	      /* Count leading zeros before the match.  */
	cmpld	r8,r0         /* Are we on the last word?  */
	srdi	r6,r9,3	      /* Convert leading zeros to bytes.  */
	addi	r0,r6,-7
	sub	r3,r8,r0
	cmpld	cr7,r3,r10
	bnelr
	bgelr	cr7
	li	r3,0
	blr

	.align	4
L(null):
	li	r3,0
	blr

/* Deals with size <= 32.  */
	.align	4
L(small_range):
	cmpldi	r5,0
	beq	L(null)

#ifdef __LITTLE_ENDIAN__
	ldx	r12,0,r8
#else
	ldbrx	r12,0,r8      /* Load reversed doubleword from memory.  */
#endif
	cmpb	r3,r12,r4     /* Check for BYTE in DWORD1.  */
	and	r3,r3,r9
	cmpldi	cr7,r3,0
	bne	cr7,L(done)

	/* Are we done already?  */
	cmpld	r8,r0
	addi	r8,r8,-8
	beqlr

	.align	5
L(loop_small):
#ifdef __LITTLE_ENDIAN__
	ldx	r12,0,r8
#else
	ldbrx	r12,0,r8
#endif
	cmpb	r3,r12,r4
	cmpld	r8,r0
	cmpldi	cr7,r3,0
	bne	cr7,L(done)
	addi	r8,r8,-8
	bne	L(loop_small)
	blr

END (MEMRCHR)
weak_alias (__memrchr, memrchr)
libc_hidden_builtin_def (memrchr)
