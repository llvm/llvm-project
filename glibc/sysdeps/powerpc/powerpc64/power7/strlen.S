/* Optimized strlen implementation for PowerPC64/POWER7 using cmpb insn.
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

#ifndef STRLEN
# define STRLEN strlen
#endif
	.machine  power7
ENTRY_TOCLESS (STRLEN)
	CALL_MCOUNT 1
	dcbt	0,r3
	clrrdi	r4,r3,3	      /* Align the address to doubleword boundary.  */
	rlwinm	r6,r3,3,26,28 /* Calculate padding.  */
	li	r0,0	      /* Doubleword with null chars to use
				 with cmpb.  */
	li	r5,-1	      /* MASK = 0xffffffffffffffff.  */
	ld	r12,0(r4)     /* Load doubleword from memory.  */
#ifdef __LITTLE_ENDIAN__
	sld	r5,r5,r6
#else
	srd	r5,r5,r6      /* MASK = MASK >> padding.  */
#endif
	orc	r9,r12,r5     /* Mask bits that are not part of the string.  */
	cmpb	r10,r9,r0     /* Check for null bytes in DWORD1.  */
	cmpdi	cr7,r10,0     /* If r10 == 0, no null's have been found.  */
	bne	cr7,L(done)

	mtcrf   0x01,r4

	/* Are we now aligned to a quadword boundary?  If so, skip to
	   the main loop.  Otherwise, go through the alignment code.  */

	bt	28,L(loop)

	/* Handle DWORD2 of pair.  */
	ldu	r12,8(r4)
	cmpb	r10,r12,r0
	cmpdi	cr7,r10,0
	bne	cr7,L(done)

	/* Main loop to look for the end of the string.  Since it's a
	   small loop (< 8 instructions), align it to 32-bytes.  */
	.p2align  5
L(loop):
	/* Load two doublewords, compare and merge in a
	   single register for speed.  This is an attempt
	   to speed up the null-checking process for bigger strings.  */

	ld	r12, 8(r4)
	ldu	r11, 16(r4)
	cmpb	r10,r12,r0
	cmpb	r9,r11,r0
	or	r8,r9,r10     /* Merge everything in one doubleword.  */
	cmpdi	cr7,r8,0
	beq	cr7,L(loop)

	/* OK, one (or both) of the doublewords contains a null byte.  Check
	   the first doubleword and decrement the address in case the first
	   doubleword really contains a null byte.  */

	cmpdi	cr6,r10,0
	addi	r4,r4,-8
	bne	cr6,L(done)

	/* The null byte must be in the second doubleword.  Adjust the address
	   again and move the result of cmpb to r10 so we can calculate the
	   length.  */

	mr	r10,r9
	addi	r4,r4,8

	/* r10 has the output of the cmpb instruction, that is, it contains
	   0xff in the same position as the null byte in the original
	   doubleword from the string.  Use that to calculate the length.  */
L(done):
#ifdef __LITTLE_ENDIAN__
	addi	r9, r10, -1   /* Form a mask from trailing zeros.  */
	andc	r9, r9, r10
	popcntd r0, r9	      /* Count the bits in the mask.  */
#else
	cntlzd	r0,r10	      /* Count leading zeros before the match.  */
#endif
	subf	r5,r3,r4
	srdi	r0,r0,3	      /* Convert leading/trailing zeros to bytes.  */
	add	r3,r5,r0      /* Compute final length.  */
	blr
END (STRLEN)
libc_hidden_builtin_def (strlen)
