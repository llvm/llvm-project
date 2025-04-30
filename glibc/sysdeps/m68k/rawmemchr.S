/* rawmemchr (str, ch) -- Return pointer to first occurrence of CH in STR.
   For Motorola 68000.
   Copyright (C) 1999-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Andreas Schwab <schwab@gnu.org>.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library.  If not, see
   <https://www.gnu.org/licenses/>.  */

#include <sysdep.h>
#include "asm-syntax.h"

	TEXT
ENTRY(__rawmemchr)
	/* Save the callee-saved registers we use.  */
	movel	R(d2),MEM_PREDEC(sp)
	cfi_adjust_cfa_offset (4)
	movel	R(d3),MEM_PREDEC(sp)
	cfi_adjust_cfa_offset (4)
	cfi_rel_offset (R(d2), 4)
	cfi_rel_offset (R(d3), 0)

	/* Get string pointer and character.  */
	movel	MEM_DISP(sp,12),R(a0)
	moveb	MEM_DISP(sp,19),R(d0)

	/* Distribute the character to all bytes of a longword.  */
	movel	R(d0),R(d1)
	lsll	#8,R(d1)
	moveb	R(d0),R(d1)
	movel	R(d1),R(d0)
	swap	R(d0)
	movew	R(d1),R(d0)

	/* First search for the character one byte at a time until the
	   pointer is aligned to a longword boundary.  */
	movel	R(a0),R(d1)
#ifdef __mcoldfire__
	andl	#3,R(d1)
#else
	andw	#3,R(d1)
#endif
	beq	L(L1)
	cmpb	MEM(a0),R(d0)
	beq	L(L9)
	addql	#1,R(a0)

#ifdef __mcoldfire__
	subql	#3,R(d1)
#else
	subqw	#3,R(d1)
#endif
	beq	L(L1)
	cmpb	MEM(a0),R(d0)
	beq	L(L9)
	addql	#1,R(a0)

#ifdef __mcoldfire__
	addql	#1,R(d1)
#else
	addqw	#1,R(d1)
#endif
	beq	L(L1)
	cmpb	MEM(a0),R(d0)
	beq	L(L9)
	addql	#1,R(a0)

L(L1:)
	/* Load the magic bits.  Unlike the generic implementation we can
	   use the carry bit as the fourth hole.  */
	movel	#0xfefefeff,R(d3)

      /* We exit the loop if adding MAGIC_BITS to LONGWORD fails to
	 change any of the hole bits of LONGWORD.

	 1) Is this safe?  Will it catch all the zero bytes?
	 Suppose there is a byte with all zeros.  Any carry bits
	 propagating from its left will fall into the hole at its
	 least significant bit and stop.  Since there will be no
	 carry from its most significant bit, the LSB of the
	 byte to the left will be unchanged, and the zero will be
	 detected.

	 2) Is this worthwhile?  Will it ignore everything except
	 zero bytes?  Suppose every byte of LONGWORD has a bit set
	 somewhere.  There will be a carry into bit 8.	If bit 8
	 is set, this will carry into bit 16.  If bit 8 is clear,
	 one of bits 9-15 must be set, so there will be a carry
	 into bit 16.  Similarly, there will be a carry into bit
	 24.  If one of bits 24-31 is set, there will be a carry
	 into bit 32 (=carry flag), so all of the hole bits will
	 be changed.

	 3) But wait!  Aren't we looking for C, not zero?
	 Good point.  So what we do is XOR LONGWORD with a longword,
	 each of whose bytes is C.  This turns each byte that is C
	 into a zero.  */

L(L2:)
	/* Get the longword in question.  */
	movel	MEM_POSTINC(a0),R(d1)
	/* XOR with the byte we search for.  */
	eorl	R(d0),R(d1)

	/* Add the magic value.  We get carry bits reported for each byte
	   which is not C.  */
	movel	R(d3),R(d2)
	addl	R(d1),R(d2)

	/* Check the fourth carry bit before it is clobbered by the next
	   XOR.  If it is not set we have a hit.  */
	bcc	L(L8)

	/* We are only interested in carry bits that change due to the
	   previous add, so remove original bits.  */
	eorl	R(d1),R(d2)

	/* Now test for the other three overflow bits.
	   Set all non-carry bits.  */
	orl	R(d3),R(d2)
	/* Add 1 to get zero if all carry bits were set.  */
	addql	#1,R(d2)

	/* If we don't get zero then at least one byte of the word equals
	   C.  */
	bne	L(L8)

	/* Get the longword in question.  */
	movel	MEM_POSTINC(a0),R(d1)
	/* XOR with the byte we search for.  */
	eorl	R(d0),R(d1)

	/* Add the magic value.  We get carry bits reported for each byte
	   which is not C.  */
	movel	R(d3),R(d2)
	addl	R(d1),R(d2)

	/* Check the fourth carry bit before it is clobbered by the next
	   XOR.  If it is not set we have a hit.  */
	bcc	L(L8)

	/* We are only interested in carry bits that change due to the
	   previous add, so remove original bits */
	eorl	R(d1),R(d2)

	/* Now test for the other three overflow bits.
	   Set all non-carry bits.  */
	orl	R(d3),R(d2)
	/* Add 1 to get zero if all carry bits were set.  */
	addql	#1,R(d2)

	/* If we don't get zero then at least one byte of the word equals
	   C.  */
	beq	L(L2)

L(L8:)
	/* We have a hit.  Check to see which byte it was.  First
	   compensate for the autoincrement in the loop.  */
	subql	#4,R(a0)

	cmpb	MEM(a0),R(d0)
	beq	L(L9)
	addql	#1,R(a0)

	cmpb	MEM(a0),R(d0)
	beq	L(L9)
	addql	#1,R(a0)

	cmpb	MEM(a0),R(d0)
	beq	L(L9)
	addql	#1,R(a0)

	/* Otherwise the fourth byte must equal C.  */
L(L9:)
	movel	R(a0),R(d0)
	movel	MEM_POSTINC(sp),R(d3)
	cfi_adjust_cfa_offset (-4)
	cfi_restore (R(d3))
	movel	MEM_POSTINC(sp),R(d2)
	cfi_adjust_cfa_offset (-4)
	cfi_restore (R(d2))
	rts
END(__rawmemchr)

libc_hidden_def (__rawmemchr)
weak_alias (__rawmemchr, rawmemchr)
