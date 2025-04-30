/* Optimized memchr implementation for PowerPC64/POWER7 using cmpb insn.
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

/* int [r3] memchr (char *s [r3], int byte [r4], int size [r5])  */

#ifndef MEMCHR
# define MEMCHR __memchr
#endif
	.machine  power7
ENTRY_TOCLESS (MEMCHR)
	CALL_MCOUNT 3
	dcbt	0,r3
	clrrdi  r8,r3,3
	insrdi	r4,r4,8,48

	/* Calculate the last acceptable address and check for possible
	   addition overflow by using satured math:
	   r7 = r3 + r5
	   r7 |= -(r7 < x)  */
	add     r7,r3,r5
	subfc   r6,r3,r7
	subfe   r9,r9,r9
	extsw   r6,r9
	or      r7,r7,r6

	insrdi	r4,r4,16,32
	cmpldi	r5,32
	li	r9, -1
	rlwinm	r6,r3,3,26,28 /* Calculate padding.  */
	insrdi  r4,r4,32,0
	addi	r7,r7,-1
#ifdef __LITTLE_ENDIAN__
	sld	r9,r9,r6
#else
	srd	r9,r9,r6
#endif
	ble	L(small_range)

	ld	r12,0(r8)     /* Load doubleword from memory.  */
	cmpb	r3,r12,r4     /* Check for BYTEs in DWORD1.  */
	and	r3,r3,r9
	clrldi	r5,r7,61      /* Byte count - 1 in last dword.  */
	clrrdi	r7,r7,3       /* Address of last doubleword.  */
	cmpldi	cr7,r3,0      /* Does r3 indicate we got a hit?  */
	bne	cr7,L(done)

	mtcrf   0x01,r8
	/* Are we now aligned to a quadword boundary?  If so, skip to
	   the main loop.  Otherwise, go through the alignment code.  */
	bt	28,L(loop_setup)

	/* Handle DWORD2 of pair.  */
	ldu	r12,8(r8)
	cmpb	r3,r12,r4
	cmpldi	cr7,r3,0
	bne	cr7,L(done)

L(loop_setup):
	/* The last dword we want to read in the loop below is the one
	   containing the last byte of the string, ie. the dword at
	   (s + size - 1) & ~7, or r7.  The first dword read is at
	   r8 + 8, we read 2 * cnt dwords, so the last dword read will
	   be at r8 + 8 + 16 * cnt - 8.  Solving for cnt gives
	   cnt = (r7 - r8) / 16  */
	sub	r6,r7,r8
	srdi	r6,r6,4	      /* Number of loop iterations.  */
	mtctr	r6            /* Setup the counter.  */

	/* Main loop to look for BYTE in the string.  Since
	   it's a small loop (8 instructions), align it to 32-bytes.  */
	.align	5
L(loop):
	/* Load two doublewords, compare and merge in a
	   single register for speed.  This is an attempt
	   to speed up the byte-checking process for bigger strings.  */
	ld	r12,8(r8)
	ldu	r11,16(r8)
	cmpb	r3,r12,r4
	cmpb	r9,r11,r4
	or	r6,r9,r3      /* Merge everything in one doubleword.  */
	cmpldi	cr7,r6,0
	bne	cr7,L(found)
	bdnz	L(loop)

	/* We may have one more dword to read.  */
	cmpld	r8,r7
	beqlr

	ldu	r12,8(r8)
	cmpb	r3,r12,r4
	cmpldi	cr6,r3,0
	bne	cr6,L(done)
	blr

	.align	4
L(found):
	/* OK, one (or both) of the doublewords contains BYTE.  Check
	   the first doubleword and decrement the address in case the first
	   doubleword really contains BYTE.  */
	cmpldi	cr6,r3,0
	addi	r8,r8,-8
	bne	cr6,L(done)

	/* BYTE must be in the second doubleword.  Adjust the address
	   again and move the result of cmpb to r3 so we can calculate the
	   pointer.  */

	mr	r3,r9
	addi	r8,r8,8

	/* r3 has the output of the cmpb instruction, that is, it contains
	   0xff in the same position as BYTE in the original
	   doubleword from the string.  Use that to calculate the pointer.
	   We need to make sure BYTE is *before* the end of the range.  */
L(done):
#ifdef __LITTLE_ENDIAN__
	addi    r0,r3,-1
	andc    r0,r0,r3
	popcntd	r0,r0	      /* Count trailing zeros.  */
#else
	cntlzd	r0,r3	      /* Count leading zeros before the match.  */
#endif
	cmpld	r8,r7         /* Are we on the last dword?  */
	srdi	r0,r0,3	      /* Convert leading/trailing zeros to bytes.  */
	add	r3,r8,r0
	cmpld	cr7,r0,r5     /* If on the last dword, check byte offset.  */
	bnelr
	blelr	cr7
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
	ld	r12,0(r8)     /* Load word from memory.  */
	cmpb	r3,r12,r4     /* Check for BYTE in DWORD1.  */
	and	r3,r3,r9
	cmpldi	cr7,r3,0
	clrldi	r5,r7,61      /* Byte count - 1 in last dword.  */
	clrrdi	r7,r7,3       /* Address of last doubleword.  */
	cmpld	r8,r7         /* Are we done already?  */
	bne	cr7,L(done)
	beqlr

	ldu	r12,8(r8)
	cmpb	r3,r12,r4
	cmpldi	cr6,r3,0
	cmpld	r8,r7
	bne	cr6,L(done)   /* Found something.  */
	beqlr		      /* Hit end of string (length).  */

	ldu	r12,8(r8)
	cmpb	r3,r12,r4
	cmpldi	cr6,r3,0
	cmpld	r8,r7
	bne	cr6,L(done)
	beqlr

	ldu	r12,8(r8)
	cmpb	r3,r12,r4
	cmpldi	cr6,r3,0
	cmpld	r8,r7
	bne	cr6,L(done)
	beqlr

	ldu	r12,8(r8)
	cmpb	r3,r12,r4
	cmpldi	cr6,r3,0
	bne	cr6,L(done)
	blr

END (MEMCHR)
weak_alias (__memchr, memchr)
libc_hidden_builtin_def (memchr)
