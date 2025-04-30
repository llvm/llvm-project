/* Optimized rawmemchr implementation for PowerPC64/POWER7 using cmpb insn.
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

/* int [r3] rawmemchr (void *s [r3], int c [r4])  */

#ifndef RAWMEMCHR
# define RAWMEMCHR __rawmemchr
#endif
	.machine  power7
ENTRY_TOCLESS (RAWMEMCHR)
	CALL_MCOUNT 2
	dcbt	0,r3
	clrrdi	r8,r3,3	      /* Align the address to doubleword boundary.  */

	/* Replicate byte to doubleword.  */
	insrdi	r4,r4,8,48
	insrdi	r4,r4,16,32
	insrdi	r4,r4,32,0

	/* Now r4 has a doubleword of c bytes.  */

	rlwinm	r6,r3,3,26,28 /* Calculate padding.  */
	ld	r12,0(r8)     /* Load doubleword from memory.  */
	cmpb	r5,r12,r4     /* Compare each byte against c byte.  */
#ifdef __LITTLE_ENDIAN__
	srd	r5,r5,r6
	sld	r5,r5,r6
#else
	sld	r5,r5,r6      /* Move left to discard ignored bits.  */
	srd	r5,r5,r6      /* Bring the bits back as zeros.  */
#endif
	cmpdi	cr7,r5,0      /* If r5 == 0, no c bytes have been found.  */
	bne	cr7,L(done)

	mtcrf   0x01,r8

	/* Are we now aligned to a quadword boundary?  If so, skip to
	   the main loop.  Otherwise, go through the alignment code.  */

	bt	28,L(loop)

	/* Handle DWORD2 of pair.  */
	ldu	r12,8(r8)
	cmpb	r5,r12,r4
	cmpdi	cr7,r5,0
	bne	cr7,L(done)
	b	L(loop)	      /* We branch here (rather than falling through)
				 to skip the nops due to heavy alignment
				 of the loop below.  */

	/* Main loop to look for the end of the string.  Since it's a
	   small loop (< 8 instructions), align it to 32-bytes.  */
	.p2align  5
L(loop):
	/* Load two doublewords, compare and merge in a
	   single register for speed.  This is an attempt
	   to speed up the byte-checking process for bigger strings.  */
	ld	r12,8(r8)
	ldu     r11,16(r8)
	cmpb	r5,r12,r4
	cmpb	r6,r11,r4
	or	r7,r5,r6
	cmpdi	cr7,r7,0
	beq	cr7,L(loop)

	/* OK, one (or both) of the doublewords contains a 'c' byte.  Check
	   the first doubleword and decrement the address in case the first
	   doubleword really contains a c byte.  */

	cmpdi	cr6,r5,0
	addi	r8,r8,-8
	bne	cr6,L(done)

	/* The 'c' byte must be in the second doubleword.  Adjust the address
	   again and move the result of cmpb to r10 so we can calculate the
	   pointer.  */
	mr	r5,r6
	addi	r8,r8,8

	/* r5 has the output of the cmpb instruction, that is, it contains
	   0xff in the same position as the 'c' byte in the original
	   doubleword from the string.  Use that fact to find out what is
	   the position of the byte inside the string.  */
L(done):
#ifdef __LITTLE_ENDIAN__
	addi    r0,r5,-1
	andc    r0,r0,r5
	popcntd	r0,r0	      /* Count trailing zeros.  */
#else
	cntlzd	r0,r5	      /* Count leading zeros before the match.  */
#endif
	srdi	r0,r0,3	      /* Convert leading zeros to bytes.  */
	add	r3,r8,r0      /* Return address of the matching char.  */
	blr
END (RAWMEMCHR)
weak_alias (__rawmemchr,rawmemchr)
libc_hidden_builtin_def (__rawmemchr)
