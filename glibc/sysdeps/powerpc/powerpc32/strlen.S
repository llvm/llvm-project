/* Optimized strlen implementation for PowerPC.
   Copyright (C) 1997-2021 Free Software Foundation, Inc.
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

/* The algorithm here uses the following techniques:

   1) Given a word 'x', we can test to see if it contains any 0 bytes
      by subtracting 0x01010101, and seeing if any of the high bits of each
      byte changed from 0 to 1. This works because the least significant
      0 byte must have had no incoming carry (otherwise it's not the least
      significant), so it is 0x00 - 0x01 == 0xff. For all other
      byte values, either they have the high bit set initially, or when
      1 is subtracted you get a value in the range 0x00-0x7f, none of which
      have their high bit set. The expression here is
      (x + 0xfefefeff) & ~(x | 0x7f7f7f7f), which gives 0x00000000 when
      there were no 0x00 bytes in the word.  You get 0x80 in bytes that
      match, but possibly false 0x80 matches in the next more significant
      byte to a true match due to carries.  For little-endian this is
      of no consequence since the least significant match is the one
      we're interested in, but big-endian needs method 2 to find which
      byte matches.

   2) Given a word 'x', we can test to see _which_ byte was zero by
      calculating ~(((x & 0x7f7f7f7f) + 0x7f7f7f7f) | x | 0x7f7f7f7f).
      This produces 0x80 in each byte that was zero, and 0x00 in all
      the other bytes. The '| 0x7f7f7f7f' clears the low 7 bits in each
      byte, and the '| x' part ensures that bytes with the high bit set
      produce 0x00. The addition will carry into the high bit of each byte
      iff that byte had one of its low 7 bits set. We can then just see
      which was the most significant bit set and divide by 8 to find how
      many to add to the index.
      This is from the book 'The PowerPC Compiler Writer's Guide',
      by Steve Hoxey, Faraydon Karim, Bill Hay and Hank Warren.

   We deal with strings not aligned to a word boundary by taking the
   first word and ensuring that bytes not part of the string
   are treated as nonzero. To allow for memory latency, we unroll the
   loop a few times, being careful to ensure that we do not read ahead
   across cache line boundaries.

   Questions to answer:
   1) How long are strings passed to strlen? If they're often really long,
   we should probably use cache management instructions and/or unroll the
   loop more. If they're often quite short, it might be better to use
   fact (2) in the inner loop than have to recalculate it.
   2) How popular are bytes with the high bit set? If they are very rare,
   on some processors it might be useful to use the simpler expression
   ~((x - 0x01010101) | 0x7f7f7f7f) (that is, on processors with only one
   ALU), but this fails when any character has its high bit set.  */

/* Some notes on register usage: Under the SVR4 ABI, we can use registers
   0 and 3 through 12 (so long as we don't call any procedures) without
   saving them. We can also use registers 14 through 31 if we save them.
   We can't use r1 (it's the stack pointer), r2 nor r13 because the user
   program may expect them to hold their usual value if we get sent
   a signal. Integer parameters are passed in r3 through r10.
   We can use condition registers cr0, cr1, cr5, cr6, and cr7 without saving
   them, the others we must save.  */

/* int [r3] strlen (char *s [r3])  */

ENTRY (strlen)

#define rTMP4	r0
#define rRTN	r3	/* incoming STR arg, outgoing result */
#define rSTR	r4	/* current string position */
#define rPADN	r5	/* number of padding bits we prepend to the
			   string to make it start at a word boundary */
#define rFEFE	r6	/* constant 0xfefefeff (-0x01010101) */
#define r7F7F	r7	/* constant 0x7f7f7f7f */
#define rWORD1	r8	/* current string word */
#define rWORD2	r9	/* next string word */
#define rMASK	r9	/* mask for first string word */
#define rTMP1	r10
#define rTMP2	r11
#define rTMP3	r12


	clrrwi	rSTR, rRTN, 2
	lis	r7F7F, 0x7f7f
	rlwinm	rPADN, rRTN, 3, 27, 28
	lwz	rWORD1, 0(rSTR)
	li	rMASK, -1
	addi	r7F7F, r7F7F, 0x7f7f
/* We use method (2) on the first two words, because rFEFE isn't
   required which reduces setup overhead.  Also gives a faster return
   for small strings on big-endian due to needing to recalculate with
   method (2) anyway.  */
#ifdef __LITTLE_ENDIAN__
	slw	rMASK, rMASK, rPADN
#else
	srw	rMASK, rMASK, rPADN
#endif
	and	rTMP1, r7F7F, rWORD1
	or	rTMP2, r7F7F, rWORD1
	add	rTMP1, rTMP1, r7F7F
	nor	rTMP3, rTMP2, rTMP1
	and.	rTMP3, rTMP3, rMASK
	mtcrf	0x01, rRTN
	bne	L(done0)
	lis	rFEFE, -0x101
	addi	rFEFE, rFEFE, -0x101
/* Are we now aligned to a doubleword boundary?  */
	bt	29, L(loop)

/* Handle second word of pair.  */
/* Perhaps use method (1) here for little-endian, saving one instruction?  */
	lwzu	rWORD1, 4(rSTR)
	and	rTMP1, r7F7F, rWORD1
	or	rTMP2, r7F7F, rWORD1
	add	rTMP1, rTMP1, r7F7F
	nor.	rTMP3, rTMP2, rTMP1
	bne	L(done0)

/* The loop.  */

L(loop):
	lwz	rWORD1, 4(rSTR)
	lwzu	rWORD2, 8(rSTR)
	add	rTMP1, rFEFE, rWORD1
	nor	rTMP2, r7F7F, rWORD1
	and.	rTMP1, rTMP1, rTMP2
	add	rTMP3, rFEFE, rWORD2
	nor	rTMP4, r7F7F, rWORD2
	bne	L(done1)
	and.	rTMP3, rTMP3, rTMP4
	beq	L(loop)

#ifndef __LITTLE_ENDIAN__
	and	rTMP1, r7F7F, rWORD2
	add	rTMP1, rTMP1, r7F7F
	andc	rTMP3, rTMP4, rTMP1
	b	L(done0)

L(done1):
	and	rTMP1, r7F7F, rWORD1
	subi	rSTR, rSTR, 4
	add	rTMP1, rTMP1, r7F7F
	andc	rTMP3, rTMP2, rTMP1

/* When we get to here, rSTR points to the first word in the string that
   contains a zero byte, and rTMP3 has 0x80 for bytes that are zero,
   and 0x00 otherwise.  */
L(done0):
	cntlzw	rTMP3, rTMP3
	subf	rTMP1, rRTN, rSTR
	srwi	rTMP3, rTMP3, 3
	add	rRTN, rTMP1, rTMP3
	blr
#else

L(done0):
	addi	rTMP1, rTMP3, -1	/* Form a mask from trailing zeros.  */
	andc	rTMP1, rTMP1, rTMP3
	cntlzw	rTMP1, rTMP1		/* Count bits not in the mask.  */
	subf	rTMP3, rRTN, rSTR
	subfic	rTMP1, rTMP1, 32-7
	srwi	rTMP1, rTMP1, 3
	add	rRTN, rTMP1, rTMP3
	blr

L(done1):
	addi	rTMP3, rTMP1, -1
	andc	rTMP3, rTMP3, rTMP1
	cntlzw	rTMP3, rTMP3
	subf	rTMP1, rRTN, rSTR
	subfic	rTMP3, rTMP3, 32-7-32
	srawi	rTMP3, rTMP3, 3
	add	rRTN, rTMP1, rTMP3
	blr
#endif

END (strlen)
libc_hidden_builtin_def (strlen)
