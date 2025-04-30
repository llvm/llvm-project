/* strlen -- Compute length of NUL terminated string.
   Highly optimized version for ix86, x>=5.
   Copyright (C) 1995-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper, <drepper@gnu.ai.mit.edu>.

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
#include "asm-syntax.h"

/* This version is especially optimized for the i586 (and following?)
   processors.  This is mainly done by using the two pipelines.  The
   version optimized for i486 is weak in this aspect because to get
   as much parallelism we have to execute some *more* instructions.

   The code below is structured to reflect the pairing of the instructions
   as *I think* it is.  I have no processor data book to verify this.
   If you find something you think is incorrect let me know.  */


/* The magic value which is used throughout in the whole code.  */
#define magic 0xfefefeff

#define PARMS	4		/* no space for saved regs */
#define STR	PARMS

	.text
ENTRY (strlen)

	movl STR(%esp), %eax
	movl $3, %edx		/* load mask (= 3) */

	andl %eax, %edx		/* separate last two bits of address */

	jz L(1)			/* aligned => start loop */
	jp L(0)			/* exactly two bits set */

	cmpb %dh, (%eax)	/* is byte NUL? */
	je L(2)			/* yes => return */

	incl %eax		/* increment pointer */
	cmpb %dh, (%eax)	/* is byte NUL? */

	je L(2)			/* yes => return */

	incl %eax		/* increment pointer */
	xorl $2, %edx

	jz L(1)

L(0):	cmpb %dh, (%eax)	/* is byte NUL? */
	je L(2)			/* yes => return */

	incl %eax		/* increment pointer */
	xorl %edx, %edx		/* We need %edx == 0 for later */

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

	 Note: %edx == 0 in any case here.  */

L(1):
	movl (%eax), %ecx	/* get word (= 4 bytes) in question */
	addl $4, %eax		/* adjust pointer for *next* word */

	subl %ecx, %edx		/* first step to negate word */
	addl $magic, %ecx	/* add magic word */

	decl %edx		/* complete negation of word */
	jnc L(3)		/* previous addl caused overflow? */

	xorl %ecx, %edx		/* (word+magic)^word */

	andl $~magic, %edx	/* any of the carry flags set? */

	jne L(3)		/* yes => determine byte */


	movl (%eax), %ecx	/* get word (= 4 bytes) in question */
	addl $4, %eax		/* adjust pointer for *next* word */

	subl %ecx, %edx		/* first step to negate word */
	addl $magic, %ecx	/* add magic word */

	decl %edx		/* complete negation of word */
	jnc L(3)		/* previous addl caused overflow? */

	xorl %ecx, %edx		/* (word+magic)^word */

	andl $~magic, %edx	/* any of the carry flags set? */

	jne L(3)		/* yes => determine byte */


	movl (%eax), %ecx	/* get word (= 4 bytes) in question */
	addl $4, %eax		/* adjust pointer for *next* word */

	subl %ecx, %edx		/* first step to negate word */
	addl $magic, %ecx	/* add magic word */

	decl %edx		/* complete negation of word */
	jnc L(3)		/* previous addl caused overflow? */

	xorl %ecx, %edx		/* (word+magic)^word */

	andl $~magic, %edx	/* any of the carry flags set? */

	jne L(3)		/* yes => determine byte */


	movl (%eax), %ecx	/* get word (= 4 bytes) in question */
	addl $4, %eax		/* adjust pointer for *next* word */

	subl %ecx, %edx		/* first step to negate word */
	addl $magic, %ecx	/* add magic word */

	decl %edx		/* complete negation of word */
	jnc L(3)		/* previous addl caused overflow? */

	xorl %ecx, %edx		/* (word+magic)^word */

	andl $~magic, %edx	/* any of the carry flags set? */

	je L(1)			/* no => start loop again */


L(3):	subl $4, %eax		/* correct too early pointer increment */
	subl $magic, %ecx

	cmpb $0, %cl		/* lowest byte NUL? */
	jz L(2)			/* yes => return */

	inc %eax		/* increment pointer */
	testb %ch, %ch		/* second byte NUL? */

	jz L(2)			/* yes => return */

	shrl $16, %ecx		/* make upper bytes accessible */
	incl %eax		/* increment pointer */

	cmpb $0, %cl		/* is third byte NUL? */
	jz L(2)			/* yes => return */

	incl %eax		/* increment pointer */

L(2):	subl STR(%esp), %eax	/* now compute the length as difference
				   between start and terminating NUL
				   character */
	ret
END (strlen)
libc_hidden_builtin_def (strlen)
