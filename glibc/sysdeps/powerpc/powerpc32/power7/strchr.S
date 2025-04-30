/* Optimized strchr implementation for PowerPC32/POWER7 using cmpb insn.
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

/* int [r3] strchr (char *s [r3], int c [r4])  */
	.machine  power7
ENTRY (strchr)
	CALL_MCOUNT
	dcbt	0,r3
	clrrwi	r8,r3,2	      /* Align the address to word boundary.  */
	cmpwi	cr7,r4,0
	lwz	r12,0(r8)     /* Load word from memory.  */
	li	r0,0	      /* Word with null chars to use
				 with cmpb.  */

	rlwinm	r6,r3,3,27,28 /* Calculate padding.  */

	beq	cr7,L(null_match)

	/* Replicate byte to word.  */
	insrwi	r4,r4,8,16
	insrwi	r4,r4,16,0

	/* Now r4 has a word of c bytes and r0 has
	   a word of null bytes.  */

	cmpb	r10,r12,r4     /* Compare each byte against c byte.  */
	cmpb	r11,r12,r0     /* Compare each byte against null byte.  */

	/* Move the words left and right to discard the bits that are
	   not part of the string and to bring them back as zeros.  */
#ifdef __LITTLE_ENDIAN__
	srw	r10,r10,r6
	srw	r11,r11,r6
	slw	r10,r10,r6
	slw	r11,r11,r6
#else
	slw	r10,r10,r6
	slw	r11,r11,r6
	srw	r10,r10,r6
	srw	r11,r11,r6
#endif
	or	r5,r10,r11    /* OR the results to speed things up.  */
	cmpwi	cr7,r5,0      /* If r5 == 0, no c or null bytes
				 have been found.  */
	bne	cr7,L(done)

	mtcrf   0x01,r8

	/* Are we now aligned to a doubleword boundary?  If so, skip to
	   the main loop.  Otherwise, go through the alignment code.  */

	bt	29,L(loop)

	/* Handle WORD2 of pair.  */
	lwzu	r12,4(r8)
	cmpb	r10,r12,r4
	cmpb	r11,r12,r0
	or	r5,r10,r11
	cmpwi	cr7,r5,0
	bne	cr7,L(done)
	b	L(loop)	      /* We branch here (rather than falling through)
				 to skip the nops due to heavy alignment
				 of the loop below.  */

	.p2align  5
L(loop):
	/* Load two words, compare and merge in a
	   single register for speed.  This is an attempt
	   to speed up the null-checking process for bigger strings.  */
	lwz	r12,4(r8)
	lwzu	r9,8(r8)
	cmpb	r10,r12,r4
	cmpb	r11,r12,r0
	cmpb	r6,r9,r4
	cmpb	r7,r9,r0
	or	r12,r10,r11
	or	r9,r6,r7
	or	r5,r12,r9
	cmpwi	cr7,r5,0
	beq	cr7,L(loop)

	/* OK, one (or both) of the words contains a c/null byte.  Check
	   the first word and decrement the address in case the first
	   word really contains a c/null byte.  */

	cmpwi	cr6,r12,0
	addi	r8,r8,-4
	bne	cr6,L(done)

	/* The c/null byte must be in the second word.  Adjust the address
	   again and move the result of cmpb to r10/r11 so we can calculate
	   the pointer.  */

	mr	r10,r6
	mr	r11,r7
	addi	r8,r8,4

	/* r10/r11 have the output of the cmpb instructions, that is,
	   0xff in the same position as the c/null byte in the original
	   word from the string.  Use that to calculate the pointer.  */
L(done):
#ifdef __LITTLE_ENDIAN__
	addi    r3,r10,-1
	andc    r3,r3,r10
	popcntw	r0,r3
	addi    r4,r11,-1
	andc    r4,r4,r11
	cmplw	cr7,r3,r4
	bgt	cr7,L(no_match)
#else
	cntlzw	r0,r10	      /* Count leading zeros before c matches.  */
	cmplw	cr7,r11,r10
	bgt	cr7,L(no_match)
#endif
	srwi	r0,r0,3	      /* Convert leading zeros to bytes.  */
	add	r3,r8,r0      /* Return address of the matching c byte
				 or null in case c was not found.  */
	blr

	.align	4
L(no_match):
	li	r3,0
	blr

/* We are here because strchr was called with a null byte.  */
	.align	4
L(null_match):
	/* r0 has a word of null bytes.  */

	cmpb	r5,r12,r0     /* Compare each byte against null bytes.  */

	/* Move the words left and right to discard the bits that are
	   not part of the string and bring them back as zeros.  */
#ifdef __LITTLE_ENDIAN__
	srw	r5,r5,r6
	slw	r5,r5,r6
#else
	slw	r5,r5,r6
	srw	r5,r5,r6
#endif
	cmpwi	cr7,r5,0      /* If r10 == 0, no c or null bytes
				 have been found.  */
	bne	cr7,L(done_null)

	mtcrf   0x01,r8

	/* Are we now aligned to a doubleword boundary?  If so, skip to
	   the main loop.  Otherwise, go through the alignment code.  */

	bt	29,L(loop_null)

	/* Handle WORD2 of pair.  */
	lwzu	r12,4(r8)
	cmpb    r5,r12,r0
	cmpwi	cr7,r5,0
	bne	cr7,L(done_null)
	b	L(loop_null)  /* We branch here (rather than falling through)
				 to skip the nops due to heavy alignment
				 of the loop below.  */

	/* Main loop to look for the end of the string.  Since it's a
	   small loop (< 8 instructions), align it to 32-bytes.  */
	.p2align  5
L(loop_null):
	/* Load two words, compare and merge in a
	   single register for speed.  This is an attempt
	   to speed up the null-checking process for bigger strings.  */
	lwz	r12,4(r8)
	lwzu    r11,8(r8)
	cmpb	r5,r12,r0
	cmpb	r10,r11,r0
	or	r6,r5,r10
	cmpwi	cr7,r6,0
	beq	cr7,L(loop_null)

	/* OK, one (or both) of the words contains a null byte.  Check
	   the first word and decrement the address in case the first
	   word really contains a null byte.  */

	cmpwi	cr6,r5,0
	addi	r8,r8,-4
	bne	cr6,L(done_null)

	/* The null byte must be in the second word.  Adjust the address
	   again and move the result of cmpb to r10 so we can calculate the
	   pointer.  */

	mr	r5,r10
	addi	r8,r8,4

	/* r5 has the output of the cmpb instruction, that is, it contains
	   0xff in the same position as the null byte in the original
	   word from the string.  Use that to calculate the pointer.  */
L(done_null):
#ifdef __LITTLE_ENDIAN__
	addi    r0,r5,-1
	andc    r0,r0,r5
	popcntw	r0,r0
#else
	cntlzw	r0,r5	      /* Count leading zeros before the match.  */
#endif
	srwi	r0,r0,3	      /* Convert leading zeros to bytes.  */
	add	r3,r8,r0      /* Return address of the matching null byte.  */
	blr
END (strchr)
weak_alias (strchr, index)
libc_hidden_builtin_def (strchr)
