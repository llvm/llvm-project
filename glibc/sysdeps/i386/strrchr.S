/* strrchr (str, ch) -- Return pointer to last occurrence of CH in STR.
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

#define PARMS	4+8	/* space for 2 saved regs */
#define RTN	PARMS
#define STR	RTN
#define CHR	STR+4

	.text
ENTRY (strrchr)

	pushl %edi		/* Save callee-safe registers used here.  */
	cfi_adjust_cfa_offset (4)
	cfi_rel_offset (edi, 0)
	pushl %esi
	cfi_adjust_cfa_offset (4)

	xorl %eax, %eax
	movl STR(%esp), %esi
	cfi_rel_offset (esi, 0)
	movl CHR(%esp), %ecx

	/* At the moment %ecx contains C.  What we need for the
	   algorithm is C in all bytes of the dword.  Avoid
	   operations on 16 bit words because these require an
	   prefix byte (and one more cycle).  */
	movb %cl, %ch		/* now it is 0|0|c|c */
	movl %ecx, %edx
	shll $16, %ecx		/* now it is c|c|0|0 */
	movw %dx, %cx		/* and finally c|c|c|c */

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

	testl $3, %esi		/* correctly aligned ? */
	jz L(19)		/* yes => begin loop */
	movb (%esi), %dl	/* load byte in question (we need it twice) */
	cmpb %dl, %cl		/* compare byte */
	jne L(11)			/* target found => return */
	movl %esi, %eax		/* remember pointer as possible result */
L(11):	orb %dl, %dl		/* is NUL? */
	jz L(2)			/* yes => return NULL */
	incl %esi		/* increment pointer */

	testl $3, %esi		/* correctly aligned ? */
	jz L(19)		/* yes => begin loop */
	movb (%esi), %dl	/* load byte in question (we need it twice) */
	cmpb %dl, %cl		/* compare byte */
	jne L(12)			/* target found => return */
	movl %esi, %eax		/* remember pointer as result */
L(12):	orb %dl, %dl		/* is NUL? */
	jz L(2)			/* yes => return NULL */
	incl %esi		/* increment pointer */

	testl $3, %esi		/* correctly aligned ? */
	jz L(19)		/* yes => begin loop */
	movb (%esi), %dl	/* load byte in question (we need it twice) */
	cmpb %dl, %cl		/* compare byte */
	jne L(13)			/* target found => return */
	movl %esi, %eax		/* remember pointer as result */
L(13):	orb %dl, %dl		/* is NUL? */
	jz L(2)			/* yes => return NULL */
	incl %esi		/* increment pointer */

	/* No we have reached alignment.  */
	jmp L(19)		/* begin loop */

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

	/* Jump to here when the character is detected.  We chose this
	   way around because the character one is looking for is not
	   as frequent as the rest and taking a conditional jump is more
	   expensive than ignoring it.

	   Some more words to the code below: it might not be obvious why
	   we decrement the source pointer here.  In the loop the pointer
	   is not pre-incremented and so it still points before the word
	   we are looking at.  But you should take a look at the instruction
	   which gets executed before we get into the loop: `addl $16, %esi'.
	   This makes the following subs into adds.  */

	/* These fill bytes make the main loop be correctly aligned.
	   We cannot use align because it is not the following instruction
	   which should be aligned.  */
	.byte 0, 0
#ifndef	PROF
	/* Profiling adds some code and so changes the alignment.  */
	.byte 0
#endif

L(4):	subl $4, %esi		/* adjust pointer */
L(41):	subl $4, %esi
L(42):	subl $4, %esi
L(43):	testl $0xff000000, %edx	/* is highest byte == C? */
	jnz L(33)		/* no => try other bytes */
	leal 15(%esi), %eax	/* store address as result */
	jmp L(1)		/* and start loop again */

L(3):	subl $4, %esi		/* adjust pointer */
L(31):	subl $4, %esi
L(32):	subl $4, %esi
L(33):	testl $0xff0000, %edx	/* is C in third byte? */
	jnz L(51)		/* no => try other bytes */
	leal 14(%esi), %eax	/* store address as result */
	jmp L(1)		/* and start loop again */

L(51):
	/* At this point we know that the byte is in one of the lower bytes.
	   We make a guess and correct it if necessary.  This reduces the
	   number of necessary jumps.  */
	leal 12(%esi), %eax	/* guess address of lowest byte as result */
	testb %dh, %dh		/* is guess correct? */
	jnz L(1)		/* yes => start loop */
	leal 13(%esi), %eax	/* correct guess to second byte */

L(1):	addl $16, %esi		/* increment pointer for full round */

L(19):	movl (%esi), %edx	/* get word (= 4 bytes) in question */
	movl $0xfefefeff, %edi	/* magic value */
	addl %edx, %edi		/* add the magic value to the word.  We get
				   carry bits reported for each byte which
				   is *not* 0 */

	/* According to the algorithm we had to reverse the effect of the
	   XOR first and then test the overflow bits.  But because the
	   following XOR would destroy the carry flag and it would (in a
	   representation with more than 32 bits) not alter then last
	   overflow, we can now test this condition.  If no carry is signaled
	   no overflow must have occurred in the last byte => it was 0.	*/

	jnc L(20)			/* found NUL => check last word */

	/* We are only interested in carry bits that change due to the
	   previous add, so remove original bits */
	xorl %edx, %edi		/* (word+magic)^word */

	/* Now test for the other three overflow bits.  */
	orl $0xfefefeff, %edi	/* set all non-carry bits */
	incl %edi		/* add 1: if one carry bit was *not* set
				   the addition will not result in 0.  */

	/* If at least one byte of the word is C we don't get 0 in %edi.  */
	jnz L(20)			/* found NUL => check last word */

	/* Now we made sure the dword does not contain the character we are
	   looking for.  But because we deal with strings we have to check
	   for the end of string before testing the next dword.  */

	xorl %ecx, %edx		/* XOR with word c|c|c|c => bytes of str == c
				   are now 0 */
	movl $0xfefefeff, %edi	/* magic value */
	addl %edx, %edi		/* add the magic value to the word.  We get
				   carry bits reported for each byte which
				   is *not* 0 */
	jnc L(4)		/* highest byte is C => examine dword */
	xorl %edx, %edi		/* ((word^charmask)+magic)^(word^charmask) */
	orl $0xfefefeff, %edi	/* set all non-carry bits */
	incl %edi		/* add 1: if one carry bit was *not* set
				   the addition will not result in 0.  */
	jnz L(3)		/* C is detected in the word => examine it */

	movl 4(%esi), %edx	/* get word (= 4 bytes) in question */
	movl $0xfefefeff, %edi	/* magic value */
	addl %edx, %edi		/* add the magic value to the word.  We get
				   carry bits reported for each byte which
				   is *not* 0 */
	jnc L(21)		/* found NUL => check last word */
	xorl %edx, %edi		/* (word+magic)^word */
	orl $0xfefefeff, %edi	/* set all non-carry bits */
	incl %edi		/* add 1: if one carry bit was *not* set
				   the addition will not result in 0.  */
	jnz L(21)		/* found NUL => check last word */
	xorl %ecx, %edx		/* XOR with word c|c|c|c => bytes of str == c
				   are now 0 */
	movl $0xfefefeff, %edi	/* magic value */
	addl %edx, %edi		/* add the magic value to the word.  We get
				   carry bits reported for each byte which
				   is *not* 0 */
	jnc L(41)		/* highest byte is C => examine dword */
	xorl %edx, %edi		/* ((word^charmask)+magic)^(word^charmask) */
	orl $0xfefefeff, %edi	/* set all non-carry bits */
	incl %edi		/* add 1: if one carry bit was *not* set
				   the addition will not result in 0.  */
	jnz L(31)		/* C is detected in the word => examine it */

	movl 8(%esi), %edx	/* get word (= 4 bytes) in question */
	movl $0xfefefeff, %edi	/* magic value */
	addl %edx, %edi		/* add the magic value to the word.  We get
				   carry bits reported for each byte which
				   is *not* 0 */
	jnc L(22)		/* found NUL => check last word */
	xorl %edx, %edi		/* (word+magic)^word */
	orl $0xfefefeff, %edi	/* set all non-carry bits */
	incl %edi		/* add 1: if one carry bit was *not* set
				   the addition will not result in 0.  */
	jnz L(22)		/* found NUL => check last word */
	xorl %ecx, %edx		/* XOR with word c|c|c|c => bytes of str == c
				   are now 0 */
	movl $0xfefefeff, %edi	/* magic value */
	addl %edx, %edi		/* add the magic value to the word.  We get
				   carry bits reported for each byte which
				   is *not* 0 */
	jnc L(42)		/* highest byte is C => examine dword */
	xorl %edx, %edi		/* ((word^charmask)+magic)^(word^charmask) */
	orl $0xfefefeff, %edi	/* set all non-carry bits */
	incl %edi		/* add 1: if one carry bit was *not* set
				   the addition will not result in 0.  */
	jnz L(32)		/* C is detected in the word => examine it */

	movl 12(%esi), %edx	/* get word (= 4 bytes) in question */
	movl $0xfefefeff, %edi	/* magic value */
	addl %edx, %edi		/* add the magic value to the word.  We get
				   carry bits reported for each byte which
				   is *not* 0 */
	jnc L(23)		/* found NUL => check last word */
	xorl %edx, %edi		/* (word+magic)^word */
	orl $0xfefefeff, %edi	/* set all non-carry bits */
	incl %edi		/* add 1: if one carry bit was *not* set
				   the addition will not result in 0.  */
	jnz L(23)		/* found NUL => check last word */
	xorl %ecx, %edx		/* XOR with word c|c|c|c => bytes of str == c
				   are now 0 */
	movl $0xfefefeff, %edi	/* magic value */
	addl %edx, %edi		/* add the magic value to the word.  We get
				   carry bits reported for each byte which
				   is *not* 0 */
	jnc L(43)		/* highest byte is C => examine dword */
	xorl %edx, %edi		/* ((word^charmask)+magic)^(word^charmask) */
	orl $0xfefefeff, %edi	/* set all non-carry bits */
	incl %edi		/* add 1: if one carry bit was *not* set
				   the addition will not result in 0.  */
	jz L(1)			/* C is not detected => restart loop */
	jmp L(33)		/* examine word */

L(23):	addl $4, %esi		/* adjust pointer */
L(22):	addl $4, %esi
L(21):	addl $4, %esi

	/* What remains to do is to test which byte the NUL char is and
	   whether the searched character appears in one of the bytes
	   before.  A special case is that the searched byte maybe NUL.
	   In this case a pointer to the terminating NUL char has to be
	   returned.  */

L(20):	cmpb %cl, %dl		/* is first byte == C? */
	jne L(24)		/* no => skip */
	movl %esi, %eax		/* store address as result */
L(24):	testb %dl, %dl		/* is first byte == NUL? */
	jz L(2)			/* yes => return */

	cmpb %cl, %dh		/* is second byte == C? */
	jne L(25)		/* no => skip */
	leal 1(%esi), %eax	/* store address as result */
L(25):	testb %dh, %dh		/* is second byte == NUL? */
	jz L(2)			/* yes => return */

	shrl $16,%edx		/* make upper bytes accessible */
	cmpb %cl, %dl		/* is third byte == C */
	jne L(26)		/* no => skip */
	leal 2(%esi), %eax	/* store address as result */
L(26):	testb %dl, %dl		/* is third byte == NUL */
	jz L(2)			/* yes => return */

	cmpb %cl, %dh		/* is fourth byte == C */
	jne L(2)		/* no => skip */
	leal 3(%esi), %eax	/* store address as result */

L(2):	popl %esi		/* restore saved register content */
	cfi_adjust_cfa_offset (-4)
	cfi_restore (esi)
	popl %edi
	cfi_adjust_cfa_offset (-4)
	cfi_restore (edi)

	ret
END (strrchr)

weak_alias (strrchr, rindex)
libc_hidden_builtin_def (strrchr)
