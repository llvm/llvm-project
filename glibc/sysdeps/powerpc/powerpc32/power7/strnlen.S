/* Optimized strnlen implementation for PowerPC32/POWER7 using cmpb insn.
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

/* int [r3] strnlen (char *s [r3], int size [r4])  */
	.machine  power7
ENTRY (__strnlen)
	CALL_MCOUNT
	dcbt	0,r3
	clrrwi	r8,r3,2	      /* Align the address to word boundary.  */
	add	r7,r3,r4      /* Calculate the last acceptable address.  */
	cmplwi	r4,16
	li	r0,0	      /* Word with null chars.  */
	addi	r7,r7,-1
	ble	L(small_range)

	rlwinm	r6,r3,3,27,28 /* Calculate padding.  */
	lwz	r12,0(r8)     /* Load word from memory.  */
	cmpb	r10,r12,r0    /* Check for null bytes in DWORD1.  */
#ifdef __LITTLE_ENDIAN__
	srw	r10,r10,r6
	slw	r10,r10,r6
#else
	slw	r10,r10,r6
	srw	r10,r10,r6
#endif
	cmplwi	cr7,r10,0     /* If r10 == 0, no null's have been found.  */
	bne	cr7,L(done)

	clrrwi	r7,r7,2       /* Address of last word.  */
	mtcrf   0x01,r8
	/* Are we now aligned to a doubleword boundary?  If so, skip to
	   the main loop.  Otherwise, go through the alignment code.  */

	bt	29,L(loop_setup)

	/* Handle WORD2 of pair.  */
	lwzu	r12,4(r8)
	cmpb	r10,r12,r0
	cmplwi	cr7,r10,0
	bne	cr7,L(done)

L(loop_setup):
	/* The last word we want to read in the loop below is the one
	   containing the last byte of the string, ie. the word at
	   (s + size - 1) & ~3, or r7.  The first word read is at
	   r8 + 4, we read 2 * cnt words, so the last word read will
	   be at r8 + 4 + 8 * cnt - 4.  Solving for cnt gives
	   cnt = (r7 - r8) / 8  */
	sub	r5,r7,r8
	srwi	r6,r5,3	      /* Number of loop iterations.  */
	mtctr	r6	      /* Setup the counter.  */

	/* Main loop to look for the null byte in the string.  Since
	   it's a small loop (< 8 instructions), align it to 32-bytes.  */
	.p2align  5
L(loop):
	/* Load two words, compare and merge in a
	   single register for speed.  This is an attempt
	   to speed up the null-checking process for bigger strings.  */

	lwz	r12,4(r8)
	lwzu	r11,8(r8)
	cmpb	r10,r12,r0
	cmpb	r9,r11,r0
	or	r5,r9,r10     /* Merge everything in one word.  */
	cmplwi	cr7,r5,0
	bne	cr7,L(found)
	bdnz	L(loop)

	/* We may have one more word to read.  */
	cmplw	cr6,r8,r7
	beq	cr6,L(end_max)

	lwzu	r12,4(r8)
	cmpb	r10,r12,r0
	cmplwi	cr6,r10,0
	bne	cr6,L(done)

L(end_max):
	mr	r3,r4
	blr

	/* OK, one (or both) of the words contains a null byte.  Check
	   the first word and decrement the address in case the first
	   word really contains a null byte.  */
	.align	4
L(found):
	cmplwi	cr6,r10,0
	addi	r8,r8,-4
	bne	cr6,L(done)

	/* The null byte must be in the second word.  Adjust the address
	   again and move the result of cmpb to r10 so we can calculate the
	   length.  */

	mr	r10,r9
	addi	r8,r8,4

	/* r10 has the output of the cmpb instruction, that is, it contains
	   0xff in the same position as the null byte in the original
	   word from the string.  Use that to calculate the length.
	   We need to make sure the null char is *before* the end of the
	   range.  */
L(done):
#ifdef __LITTLE_ENDIAN__
	addi	r0,r10,-1
	andc	r0,r0,r10
	popcntw	r0,r0
#else
	cntlzw	r0,r10	      /* Count leading zeros before the match.  */
#endif
	sub	r3,r8,r3
	srwi	r0,r0,3	      /* Convert leading/trailing zeros to bytes.  */
	add	r3,r3,r0      /* Length until the match.  */
	cmplw	r3,r4
	blelr
	mr	r3,r4
	blr

/* Deals with size <= 16.  */
	.align	4
L(small_range):
	cmplwi	r4,0
	beq	L(end_max)

	clrrwi	r7,r7,2       /* Address of last word.  */

	rlwinm	r6,r3,3,27,28 /* Calculate padding.  */
	lwz	r12,0(r8)     /* Load word from memory.  */
	cmpb	r10,r12,r0    /* Check for null bytes in WORD1.  */
#ifdef __LITTLE_ENDIAN__
	srw	r10,r10,r6
	slw	r10,r10,r6
#else
	slw	r10,r10,r6
	srw	r10,r10,r6
#endif
	cmplwi	cr7,r10,0
	bne	cr7,L(done)

	cmplw	r8,r7
	beq	L(end_max)

	.p2align  5
L(loop_small):
	lwzu	r12,4(r8)
	cmpb	r10,r12,r0
	cmplwi	cr6,r10,0
	bne	cr6,L(done)
	cmplw	r8,r7
	bne	L(loop_small)
	mr	r3,r4
	blr

END (__strnlen)
libc_hidden_def (__strnlen)
weak_alias (__strnlen, strnlen)
libc_hidden_builtin_def (strnlen)
