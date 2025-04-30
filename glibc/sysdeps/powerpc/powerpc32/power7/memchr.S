/* Optimized memchr implementation for PowerPC32/POWER7 using cmpb insn.
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
	.machine  power7
ENTRY (__memchr)
	CALL_MCOUNT
	dcbt	0,r3
	clrrwi  r8,r3,2
	insrwi	r4,r4,8,16    /* Replicate byte to word.  */

	/* Calculate the last acceptable address and check for possible
	   addition overflow by using satured math:
	   r7 = r3 + r5
	   r7 |= -(r7 < x)  */
	add     r7,r3,r5
	subfc   r6,r3,r7
	subfe   r9,r9,r9
	or      r7,r7,r9

	insrwi	r4,r4,16,0
	cmplwi	r5,16
	li	r9, -1
	rlwinm	r6,r3,3,27,28 /* Calculate padding.  */
	addi	r7,r7,-1
#ifdef __LITTLE_ENDIAN__
	slw	r9,r9,r6
#else
	srw	r9,r9,r6
#endif
	ble	L(small_range)

	lwz	r12,0(r8)     /* Load word from memory.  */
	cmpb	r3,r12,r4     /* Check for BYTEs in WORD1.  */
	and	r3,r3,r9
	clrlwi	r5,r7,30      /* Byte count - 1 in last word.  */
	clrrwi	r7,r7,2       /* Address of last word.  */
	cmplwi	cr7,r3,0      /* If r3 == 0, no BYTEs have been found.  */
	bne	cr7,L(done)

	mtcrf   0x01,r8
	/* Are we now aligned to a doubleword boundary?  If so, skip to
	   the main loop.  Otherwise, go through the alignment code.  */
	bt	29,L(loop_setup)

	/* Handle WORD2 of pair.  */
	lwzu	r12,4(r8)
	cmpb	r3,r12,r4
	cmplwi	cr7,r3,0
	bne	cr7,L(done)

L(loop_setup):
	/* The last word we want to read in the loop below is the one
	   containing the last byte of the string, ie. the word at
	   (s + size - 1) & ~3, or r7.  The first word read is at
	   r8 + 4, we read 2 * cnt words, so the last word read will
	   be at r8 + 4 + 8 * cnt - 4.  Solving for cnt gives
	   cnt = (r7 - r8) / 8  */
	sub	r6,r7,r8
	srwi	r6,r6,3	      /* Number of loop iterations.  */
	mtctr	r6            /* Setup the counter.  */

	/* Main loop to look for BYTE in the string.  Since
	   it's a small loop (8 instructions), align it to 32-bytes.  */
	.align	5
L(loop):
	/* Load two words, compare and merge in a
	   single register for speed.  This is an attempt
	   to speed up the byte-checking process for bigger strings.  */
	lwz	r12,4(r8)
	lwzu	r11,8(r8)
	cmpb	r3,r12,r4
	cmpb	r9,r11,r4
	or	r6,r9,r3      /* Merge everything in one word.  */
	cmplwi	cr7,r6,0
	bne	cr7,L(found)
	bdnz	L(loop)

	/* We may have one more dword to read.  */
	cmplw	r8,r7
	beqlr

	lwzu	r12,4(r8)
	cmpb	r3,r12,r4
	cmplwi	cr6,r3,0
	bne	cr6,L(done)
	blr

	.align	4
L(found):
	/* OK, one (or both) of the words contains BYTE.  Check
	   the first word and decrement the address in case the first
	   word really contains BYTE.  */
	cmplwi	cr6,r3,0
	addi	r8,r8,-4
	bne	cr6,L(done)

	/* BYTE must be in the second word.  Adjust the address
	   again and move the result of cmpb to r3 so we can calculate the
	   pointer.  */

	mr	r3,r9
	addi	r8,r8,4

	/* r3 has the output of the cmpb instruction, that is, it contains
	   0xff in the same position as BYTE in the original
	   word from the string.  Use that to calculate the pointer.
	   We need to make sure BYTE is *before* the end of the range.  */
L(done):
#ifdef __LITTLE_ENDIAN__
	addi    r0,r3,-1
	andc    r0,r0,r3
	popcntw	r0,r0	      /* Count trailing zeros.  */
#else
	cntlzw	r0,r3	      /* Count leading zeros before the match.  */
#endif
	cmplw	r8,r7         /* Are we on the last word?  */
	srwi	r0,r0,3	      /* Convert leading/trailing zeros to bytes.  */
	add	r3,r8,r0
	cmplw	cr7,r0,r5     /* If on the last dword, check byte offset.  */
	bnelr
	blelr	cr7
	li	r3,0
	blr

	.align	4
L(null):
	li	r3,0
	blr

/* Deals with size <= 16.  */
	.align	4
L(small_range):
	cmplwi	r5,0
	beq	L(null)
	lwz	r12,0(r8)     /* Load word from memory.  */
	cmpb	r3,r12,r4     /* Check for BYTE in DWORD1.  */
	and	r3,r3,r9
	cmplwi	cr7,r3,0
	clrlwi	r5,r7,30      /* Byte count - 1 in last word.  */
	clrrwi	r7,r7,2       /* Address of last word.  */
	cmplw	r8,r7         /* Are we done already?  */
	bne	cr7,L(done)
	beqlr

	lwzu	r12,4(r8)
	cmpb	r3,r12,r4
	cmplwi	cr6,r3,0
	cmplw	r8,r7
	bne	cr6,L(done)
	beqlr

	lwzu	r12,4(r8)
	cmpb	r3,r12,r4
	cmplwi	cr6,r3,0
	cmplw	r8,r7
	bne	cr6,L(done)
	beqlr

	lwzu	r12,4(r8)
	cmpb	r3,r12,r4
	cmplwi	cr6,r3,0
	cmplw	r8,r7
	bne	cr6,L(done)
	beqlr

	lwzu	r12,4(r8)
	cmpb	r3,r12,r4
	cmplwi	cr6,r3,0
	bne	cr6,L(done)
	blr

END (__memchr)
weak_alias (__memchr, memchr)
libc_hidden_builtin_def (memchr)
