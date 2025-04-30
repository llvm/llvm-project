/* strchr (str, ch) -- Return pointer to first occurrence of CH in STR.
   For Intel 80x86, x>=3.
   Copyright (C) 1994-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@gnu.ai.mit.edu>
   Some optimisations by Alan Modra <Alan@SPRI.Levels.UniSA.Edu.Au>

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

#define PARMS	4+4		/* space for 1 saved reg */
#define RTN	PARMS
#define STR	RTN
#define CHR	STR+4

	.text
ENTRY (strchr)

	pushl %edi		/* Save callee-safe registers used here.  */
	cfi_adjust_cfa_offset (4)
	cfi_rel_offset (edi, 0)
	movl STR(%esp), %eax
	movl CHR(%esp), %edx

	/* At the moment %edx contains C.  What we need for the
	   algorithm is C in all bytes of the dword.  Avoid
	   operations on 16 bit words because these require an
	   prefix byte (and one more cycle).  */
	movb %dl, %dh		/* now it is 0|0|c|c */
	movl %edx, %ecx
	shll $16, %edx		/* now it is c|c|0|0 */
	movw %cx, %dx		/* and finally c|c|c|c */

	/* Before we start with the main loop we process single bytes
	   until the source pointer is aligned.  This has two reasons:
	   1. aligned 32-bit memory access is faster
	   and (more important)
	   2. we process in the main loop 32 bit in one step although
	      we don't know the end of the string.  But accessing at
	      4-byte alignment guarantees that we never access illegal
	      memory if this would not also be done by the trivial
	      implementation (this is because all processor inherent
	      boundaries are multiples of 4.  */

	testb $3, %al		/* correctly aligned ? */
	jz L(11)		/* yes => begin loop */
	movb (%eax), %cl	/* load byte in question (we need it twice) */
	cmpb %cl, %dl		/* compare byte */
	je L(6)			/* target found => return */
	testb %cl, %cl		/* is NUL? */
	jz L(2)			/* yes => return NULL */
	incl %eax		/* increment pointer */

	testb $3, %al		/* correctly aligned ? */
	jz L(11)		/* yes => begin loop */
	movb (%eax), %cl	/* load byte in question (we need it twice) */
	cmpb %cl, %dl		/* compare byte */
	je L(6)			/* target found => return */
	testb %cl, %cl		/* is NUL? */
	jz L(2)			/* yes => return NULL */
	incl %eax		/* increment pointer */

	testb $3, %al		/* correctly aligned ? */
	jz L(11)		/* yes => begin loop */
	movb (%eax), %cl	/* load byte in question (we need it twice) */
	cmpb %cl, %dl		/* compare byte */
	je L(6)			/* target found => return */
	testb %cl, %cl		/* is NUL? */
	jz L(2)			/* yes => return NULL */
	incl %eax		/* increment pointer */

	/* No we have reached alignment.  */
	jmp L(11)		/* begin loop */

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

	/* Each round the main loop processes 16 bytes.  */

	ALIGN(4)

L(1):	addl $16, %eax		/* adjust pointer for whole round */

L(11):	movl (%eax), %ecx	/* get word (= 4 bytes) in question */
	xorl %edx, %ecx		/* XOR with word c|c|c|c => bytes of str == c
				   are now 0 */
	movl $0xfefefeff, %edi	/* magic value */
	addl %ecx, %edi		/* add the magic value to the word.  We get
				   carry bits reported for each byte which
				   is *not* C */

	/* According to the algorithm we had to reverse the effect of the
	   XOR first and then test the overflow bits.  But because the
	   following XOR would destroy the carry flag and it would (in a
	   representation with more than 32 bits) not alter then last
	   overflow, we can now test this condition.  If no carry is signaled
	   no overflow must have occurred in the last byte => it was 0.	*/
	jnc L(7)

	/* We are only interested in carry bits that change due to the
	   previous add, so remove original bits */
	xorl %ecx, %edi		/* ((word^charmask)+magic)^(word^charmask) */

	/* Now test for the other three overflow bits.  */
	orl $0xfefefeff, %edi	/* set all non-carry bits */
	incl %edi		/* add 1: if one carry bit was *not* set
				   the addition will not result in 0.  */

	/* If at least one byte of the word is C we don't get 0 in %edi.  */
	jnz L(7)		/* found it => return pointer */

	/* Now we made sure the dword does not contain the character we are
	   looking for.  But because we deal with strings we have to check
	   for the end of string before testing the next dword.  */

	xorl %edx, %ecx		/* restore original dword without reload */
	movl $0xfefefeff, %edi	/* magic value */
	addl %ecx, %edi		/* add the magic value to the word.  We get
				   carry bits reported for each byte which
				   is *not* 0 */
	jnc L(2)		/* highest byte is NUL => return NULL */
	xorl %ecx, %edi		/* (word+magic)^word */
	orl $0xfefefeff, %edi	/* set all non-carry bits */
	incl %edi		/* add 1: if one carry bit was *not* set
				   the addition will not result in 0.  */
	jnz L(2)		/* found NUL => return NULL */

	movl 4(%eax), %ecx	/* get word (= 4 bytes) in question */
	xorl %edx, %ecx		/* XOR with word c|c|c|c => bytes of str == c
				   are now 0 */
	movl $0xfefefeff, %edi	/* magic value */
	addl %ecx, %edi		/* add the magic value to the word.  We get
				   carry bits reported for each byte which
				   is *not* C */
	jnc L(71)		/* highest byte is C => return pointer */
	xorl %ecx, %edi		/* ((word^charmask)+magic)^(word^charmask) */
	orl $0xfefefeff, %edi	/* set all non-carry bits */
	incl %edi		/* add 1: if one carry bit was *not* set
				   the addition will not result in 0.  */
	jnz L(71)		/* found it => return pointer */
	xorl %edx, %ecx		/* restore original dword without reload */
	movl $0xfefefeff, %edi	/* magic value */
	addl %ecx, %edi		/* add the magic value to the word.  We get
				   carry bits reported for each byte which
				   is *not* 0 */
	jnc L(2)		/* highest byte is NUL => return NULL */
	xorl %ecx, %edi		/* (word+magic)^word */
	orl $0xfefefeff, %edi	/* set all non-carry bits */
	incl %edi		/* add 1: if one carry bit was *not* set
				   the addition will not result in 0.  */
	jnz L(2)		/* found NUL => return NULL */

	movl 8(%eax), %ecx	/* get word (= 4 bytes) in question */
	xorl %edx, %ecx		/* XOR with word c|c|c|c => bytes of str == c
				   are now 0 */
	movl $0xfefefeff, %edi	/* magic value */
	addl %ecx, %edi		/* add the magic value to the word.  We get
				   carry bits reported for each byte which
				   is *not* C */
	jnc L(72)		/* highest byte is C => return pointer */
	xorl %ecx, %edi		/* ((word^charmask)+magic)^(word^charmask) */
	orl $0xfefefeff, %edi	/* set all non-carry bits */
	incl %edi		/* add 1: if one carry bit was *not* set
				   the addition will not result in 0.  */
	jnz L(72)		/* found it => return pointer */
	xorl %edx, %ecx		/* restore original dword without reload */
	movl $0xfefefeff, %edi	/* magic value */
	addl %ecx, %edi		/* add the magic value to the word.  We get
				   carry bits reported for each byte which
				   is *not* 0 */
	jnc L(2)		/* highest byte is NUL => return NULL */
	xorl %ecx, %edi		/* (word+magic)^word */
	orl $0xfefefeff, %edi	/* set all non-carry bits */
	incl %edi		/* add 1: if one carry bit was *not* set
				   the addition will not result in 0.  */
	jnz L(2)		/* found NUL => return NULL */

	movl 12(%eax), %ecx	/* get word (= 4 bytes) in question */
	xorl %edx, %ecx		/* XOR with word c|c|c|c => bytes of str == c
				   are now 0 */
	movl $0xfefefeff, %edi	/* magic value */
	addl %ecx, %edi		/* add the magic value to the word.  We get
				   carry bits reported for each byte which
				   is *not* C */
	jnc L(73)		/* highest byte is C => return pointer */
	xorl %ecx, %edi		/* ((word^charmask)+magic)^(word^charmask) */
	orl $0xfefefeff, %edi	/* set all non-carry bits */
	incl %edi		/* add 1: if one carry bit was *not* set
				   the addition will not result in 0.  */
	jnz L(73)		/* found it => return pointer */
	xorl %edx, %ecx		/* restore original dword without reload */
	movl $0xfefefeff, %edi	/* magic value */
	addl %ecx, %edi		/* add the magic value to the word.  We get
				   carry bits reported for each byte which
				   is *not* 0 */
	jnc L(2)		/* highest byte is NUL => return NULL */
	xorl %ecx, %edi		/* (word+magic)^word */
	orl $0xfefefeff, %edi	/* set all non-carry bits */
	incl %edi		/* add 1: if one carry bit was *not* set
				   the addition will not result in 0.  */
	jz L(1)			/* no NUL found => restart loop */

L(2):	/* Return NULL.  */
	xorl %eax, %eax
	popl %edi		/* restore saved register content */
	cfi_adjust_cfa_offset (-4)
	cfi_restore (edi)

	ret

	cfi_adjust_cfa_offset (4)
	cfi_rel_offset (edi, 0)
L(73):	addl $4, %eax		/* adjust pointer */
L(72):	addl $4, %eax
L(71):	addl $4, %eax

	/* We now scan for the byte in which the character was matched.
	   But we have to take care of the case that a NUL char is
	   found before this in the dword.  Note that we XORed %ecx
	   with the byte we're looking for, therefore the tests below look
	   reversed.  */

L(7):	testb %cl, %cl		/* is first byte C? */
	jz L(6)			/* yes => return pointer */
	cmpb %dl, %cl		/* is first byte NUL? */
	je L(2)			/* yes => return NULL */
	incl %eax		/* it's not in the first byte */

	testb %ch, %ch		/* is second byte C? */
	jz L(6)			/* yes => return pointer */
	cmpb %dl, %ch		/* is second byte NUL? */
	je L(2)			/* yes => return NULL? */
	incl %eax		/* it's not in the second byte */

	shrl $16, %ecx		/* make upper byte accessible */
	testb %cl, %cl		/* is third byte C? */
	jz L(6)			/* yes => return pointer */
	cmpb %dl, %cl		/* is third byte NUL? */
	je L(2)			/* yes => return NULL */

	/* It must be in the fourth byte and it cannot be NUL.  */
	incl %eax

L(6):
	popl %edi		/* restore saved register content */
	cfi_adjust_cfa_offset (-4)
	cfi_restore (edi)

	ret
END (strchr)

weak_alias (strchr, index)
libc_hidden_builtin_def (strchr)
