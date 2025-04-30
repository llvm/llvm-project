/* Optimized strlen implementation for PowerPC32/POWER7 using cmpb insn.
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

/* int [r3] strlen (char *s [r3])  */
	.machine  power7
ENTRY (strlen)
	CALL_MCOUNT
	dcbt	0,r3
	clrrwi	r4,r3,2	      /* Align the address to word boundary.  */
	rlwinm	r6,r3,3,27,28 /* Calculate padding.  */
	li	r0,0	      /* Word with null chars to use with cmpb.  */
	li	r5,-1	      /* MASK = 0xffffffffffffffff.  */
	lwz	r12,0(r4)     /* Load word from memory.  */
#ifdef __LITTLE_ENDIAN__
	slw	r5,r5,r6
#else
	srw	r5,r5,r6      /* MASK = MASK >> padding.  */
#endif
	orc	r9,r12,r5     /* Mask bits that are not part of the string.  */
	cmpb	r10,r9,r0     /* Check for null bytes in WORD1.  */
	cmpwi	cr7,r10,0     /* If r10 == 0, no null's have been found.  */
	bne	cr7,L(done)

	mtcrf   0x01,r4

	/* Are we now aligned to a doubleword boundary?  If so, skip to
	   the main loop.  Otherwise, go through the alignment code.  */

	bt	29,L(loop)

	/* Handle WORD2 of pair.  */
	lwzu	r12,4(r4)
	cmpb	r10,r12,r0
	cmpwi	cr7,r10,0
	bne	cr7,L(done)

	/* Main loop to look for the end of the string.  Since it's a
	   small loop (< 8 instructions), align it to 32-bytes.  */
	.p2align  5
L(loop):
	/* Load two words, compare and merge in a
	   single register for speed.  This is an attempt
	   to speed up the null-checking process for bigger strings.  */

	lwz	r12, 4(r4)
	lwzu	r11, 8(r4)
	cmpb	r10,r12,r0
	cmpb	r9,r11,r0
	or	r8,r9,r10     /* Merge everything in one word.  */
	cmpwi	cr7,r8,0
	beq	cr7,L(loop)

	/* OK, one (or both) of the words contains a null byte.  Check
	   the first word and decrement the address in case the first
	   word really contains a null byte.  */

	cmpwi	cr6,r10,0
	addi	r4,r4,-4
	bne	cr6,L(done)

	/* The null byte must be in the second word.  Adjust the address
	   again and move the result of cmpb to r10 so we can calculate the
	   length.  */

	mr	r10,r9
	addi	r4,r4,4

	/* r10 has the output of the cmpb instruction, that is, it contains
	   0xff in the same position as the null byte in the original
	   word from the string.  Use that to calculate the length.  */
L(done):
#ifdef __LITTLE_ENDIAN__
	addi	r9, r10, -1   /* Form a mask from trailing zeros.  */
	andc	r9, r9, r10
	popcntw r0, r9	      /* Count the bits in the mask.  */
#else
	cntlzw	r0,r10	      /* Count leading zeros before the match.  */
#endif
	subf	r5,r3,r4
	srwi	r0,r0,3	      /* Convert leading zeros to bytes.  */
	add	r3,r5,r0      /* Compute final length.  */
	blr
END (strlen)
libc_hidden_builtin_def (strlen)
