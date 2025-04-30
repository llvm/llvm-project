/* Optimized strrchr implementation for PowerPC64/POWER7 using cmpb insn.
   Copyright (C) 2014-2021 Free Software Foundation, Inc.
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

/* int [r3] strrchr (char *s [r3], int c [r4])  */

#ifndef STRRCHR
# define STRRCHR strrchr
#endif

	.machine  power7
ENTRY_TOCLESS (STRRCHR)
	CALL_MCOUNT 2
	dcbt	0,r3
	clrrdi	r8,r3,3	      /* Align the address to doubleword boundary.  */
	cmpdi	cr7,r4,0
	ld	r12,0(r8)     /* Load doubleword from memory.  */
	li	r9,0	      /* used to store last occurence */
	li	r0,0	      /* Doubleword with null chars to use
				 with cmpb.  */

	rlwinm	r6,r3,3,26,28 /* Calculate padding.  */

	beq	cr7,L(null_match)

	/* Replicate byte to doubleword.  */
	insrdi	r4,r4,8,48
	insrdi	r4,r4,16,32
	insrdi	r4,r4,32,0

	/* r4 is changed now ,if its passed as more chars
	   check for null again */
	cmpdi	cr7,r4,0
	beq	cr7,L(null_match)
	/* Now r4 has a doubleword of c bytes and r0 has
	   a doubleword of null bytes.  */

	cmpb	r10,r12,r4     /* Compare each byte against c byte.  */
	cmpb	r11,r12,r0     /* Compare each byte against null byte.  */

	/* Move the doublewords left and right to discard the bits that are
	   not part of the string and bring them back as zeros.  */
#ifdef __LITTLE_ENDIAN__
	srd	r10,r10,r6
	srd	r11,r11,r6
	sld	r10,r10,r6
	sld	r11,r11,r6
#else
	sld	r10,r10,r6
	sld	r11,r11,r6
	srd	r10,r10,r6
	srd	r11,r11,r6
#endif
	or	r5,r10,r11    /* OR the results to speed things up.  */
	cmpdi	cr7,r5,0      /* If r5 == 0, no c or null bytes
				 have been found.  */
	bne	cr7,L(done)

L(align):
	mtcrf	0x01,r8

	/* Are we now aligned to a doubleword boundary?  If so, skip to
	   the main loop.  Otherwise, go through the alignment code.  */

	bt	28,L(loop)

	/* Handle WORD2 of pair.  */
	ldu	r12,8(r8)
	cmpb	r10,r12,r4
	cmpb	r11,r12,r0
	or	r5,r10,r11
	cmpdi	cr7,r5,0
	bne	cr7,L(done)
	b	L(loop)	      /* We branch here (rather than falling through)
				 to skip the nops due to heavy alignment
				 of the loop below.  */
	.p2align  5
L(loop):
	/* Load two doublewords, compare and merge in a
	   single register for speed.  This is an attempt
	   to speed up the null-checking process for bigger strings.  */
	ld	r12,8(r8)
	ldu	r7,16(r8)
	cmpb	r10,r12,r4
	cmpb	r11,r12,r0
	cmpb	r6,r7,r4
	cmpb	r7,r7,r0
	or	r12,r10,r11
	or	r5,r6,r7
	or	r5,r12,r5
	cmpdi	cr7,r5,0
	beq	cr7,L(loop)

	/* OK, one (or both) of the doublewords contains a c/null byte.  Check
	   the first doubleword and decrement the address in case the first
	   doubleword really contains a c/null byte.  */
	cmpdi	cr6,r12,0
	addi	r8,r8,-8
	bne	cr6,L(done)

	/* The c/null byte must be in the second doubleword.  Adjust the
	   address again and move the result of cmpb to r10 so we can calculate
	   the pointer.  */

	mr	r10,r6
	mr	r11,r7
	addi	r8,r8,8

	/* r10/r11 have the output of the cmpb instructions, that is,
	   0xff in the same position as the c/null byte in the original
	   doubleword from the string.  Use that to calculate the pointer.  */

L(done):
	/* if there are more than one 0xff in r11, find the first pos of ff
	   in r11 and fill r10 with 0 from that position */
	cmpdi	cr7,r11,0
	beq	cr7,L(no_null)
#ifdef __LITTLE_ENDIAN__
	addi	r3,r11,-1
	andc	r3,r3,r11
	popcntd r0,r3
#else
	cntlzd	r0,r11
#endif
	subfic	r0,r0,63
	li	r6,-1
#ifdef __LITTLE_ENDIAN__
	srd	r0,r6,r0
#else
	sld	r0,r6,r0
#endif
	and	r10,r0,r10
L(no_null):
#ifdef __LITTLE_ENDIAN__
	cntlzd	r0,r10		/* Count leading zeros before c matches.  */
	addi	r3,r10,-1
	andc	r3,r3,r10
	addi	r10,r11,-1
	andc	r10,r10,r11
	cmpld	cr7,r3,r10
	bgt	cr7,L(no_match)
#else
	addi	r3,r10,-1	/* Count trailing zeros before c matches.  */
	andc	r3,r3,r10
	popcntd	r0,r3
	cmpld	cr7,r11,r10
	bgt	cr7,L(no_match)
#endif
	srdi	r0,r0,3		/* Convert trailing zeros to bytes.  */
	subfic	r0,r0,7
	add	r9,r8,r0      /* Return address of the matching c byte
				 or null in case c was not found.  */
	li	r0,0
	cmpdi	cr7,r11,0     /* If r11 == 0, no null's have been found.  */
	beq	cr7,L(align)

	.align	4
L(no_match):
	mr	r3,r9
	blr

/* We are here because strrchr was called with a null byte.  */
	.align	4
L(null_match):
	/* r0 has a doubleword of null bytes.  */

	cmpb	r5,r12,r0     /* Compare each byte against null bytes.  */

	/* Move the doublewords left and right to discard the bits that are
	   not part of the string and bring them back as zeros.  */
#ifdef __LITTLE_ENDIAN__
	srd	r5,r5,r6
	sld	r5,r5,r6
#else
	sld	r5,r5,r6
	srd	r5,r5,r6
#endif
	cmpdi	cr7,r5,0      /* If r10 == 0, no c or null bytes
				 have been found.  */
	bne	cr7,L(done_null)

	mtcrf	0x01,r8

	/* Are we now aligned to a quadword boundary?  If so, skip to
	   the main loop.  Otherwise, go through the alignment code.  */

	bt	28,L(loop_null)

	/* Handle WORD2 of pair.  */
	ldu	r12,8(r8)
	cmpb	r5,r12,r0
	cmpdi	cr7,r5,0
	bne	cr7,L(done_null)
	b	L(loop_null)  /* We branch here (rather than falling through)
				 to skip the nops due to heavy alignment
				 of the loop below.  */

	/* Main loop to look for the end of the string.  Since it's a
	   small loop (< 8 instructions), align it to 32-bytes.  */
	.p2align  5
L(loop_null):
	/* Load two doublewords, compare and merge in a
	   single register for speed.  This is an attempt
	   to speed up the null-checking process for bigger strings.  */
	ld	r12,8(r8)
	ldu	r11,16(r8)
	cmpb	r5,r12,r0
	cmpb	r10,r11,r0
	or	r6,r5,r10
	cmpdi	cr7,r6,0
	beq	cr7,L(loop_null)

	/* OK, one (or both) of the doublewords contains a null byte.  Check
	   the first doubleword and decrement the address in case the first
	   doubleword really contains a null byte.  */

	cmpdi	cr6,r5,0
	addi	r8,r8,-8
	bne	cr6,L(done_null)

	/* The null byte must be in the second doubleword.  Adjust the address
	   again and move the result of cmpb to r10 so we can calculate the
	   pointer.  */

	mr	r5,r10
	addi	r8,r8,8

	/* r5 has the output of the cmpb instruction, that is, it contains
	   0xff in the same position as the null byte in the original
	   doubleword from the string.  Use that to calculate the pointer.  */
L(done_null):
#ifdef __LITTLE_ENDIAN__
	addi	r0,r5,-1
	andc	r0,r0,r5
	popcntd	r0,r0
#else
	cntlzd	r0,r5	      /* Count leading zeros before the match.  */
#endif
	srdi	r0,r0,3	      /* Convert trailing zeros to bytes.  */
	add	r3,r8,r0      /* Return address of the matching null byte.  */
	blr
END (STRRCHR)
weak_alias (strrchr, rindex)
libc_hidden_builtin_def (strrchr)
