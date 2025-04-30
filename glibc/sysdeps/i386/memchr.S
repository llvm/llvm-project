/* memchr (str, chr, len) -- Return pointer to first occurrence of CHR in STR
	 less than LEN.  For Intel 80x86, x>=3.
   Copyright (C) 1994-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@gnu.ai.mit.edu>
   Optimised a little by Alan Modra <Alan@SPRI.Levels.UniSA.Edu.Au>
   This version is developed using the same algorithm as the fast C
   version which carries the following introduction:
   Based on strlen implementation by Torbjorn Granlund (tege@sics.se),
   with help from Dan Sahlin (dan@sics.se) and
   commentary by Jim Blandy (jimb@ai.mit.edu);
   adaptation to memchr suggested by Dick Karpinski (dick@cca.ucsf.edu),
   and implemented by Roland McGrath (roland@ai.mit.edu).

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

#define PARMS	4+8		/* space for 2 saved regs */
#define RTN	PARMS
#define STR	RTN
#define CHR	STR+4
#define LEN	CHR+4

	.text
ENTRY (__memchr)

	/* Save callee-safe registers used in this function.  */
	pushl %esi
	cfi_adjust_cfa_offset (4)
	pushl %edi
	cfi_adjust_cfa_offset (4)
	cfi_rel_offset (edi, 0)

	/* Load parameters into registers.  */
	movl STR(%esp), %eax	/* str: pointer to memory block.  */
	movl CHR(%esp), %edx	/* c: byte we are looking for.  */
	movl LEN(%esp), %esi	/* len: length of memory block.  */
	cfi_rel_offset (esi, 4)

	/* If my must not test more than three characters test
	   them one by one.  This is especially true for 0.  */
	cmpl $4, %esi
	jb L(3)

	/* At the moment %edx contains CHR.  What we need for the
	   algorithm is CHR in all bytes of the dword.  Avoid
	   operations on 16 bit words because these require an
	   prefix byte (and one more cycle).  */
	movb %dl, %dh		/* Now it is 0|0|c|c */
	movl %edx, %ecx
	shll $16, %edx		/* Now c|c|0|0 */
	movw %cx, %dx		/* And finally c|c|c|c */

	/* Better performance can be achieved if the word (32
	   bit) memory access is aligned on a four-byte-boundary.
	   So process first bytes one by one until boundary is
	   reached. Don't use a loop for better performance.  */

	testb $3, %al		/* correctly aligned ? */
	je L(2)			/* yes => begin loop */
	cmpb %dl, (%eax)	/* compare byte */
	je L(9)			/* target found => return */
	incl %eax		/* increment source pointer */
	decl %esi		/* decrement length counter */
	je L(4)			/* len==0 => return NULL */

	testb $3, %al		/* correctly aligned ? */
	je L(2)			/* yes => begin loop */
	cmpb %dl, (%eax)	/* compare byte */
	je L(9)			/* target found => return */
	incl %eax		/* increment source pointer */
	decl %esi		/* decrement length counter */
	je L(4)			/* len==0 => return NULL */

	testb $3, %al		/* correctly aligned ? */
	je L(2)			/* yes => begin loop */
	cmpb %dl, (%eax)	/* compare byte */
	je L(9)			/* target found => return */
	incl %eax		/* increment source pointer */
	decl %esi		/* decrement length counter */
	/* no test for len==0 here, because this is done in the
	   loop head */
	jmp L(2)

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

	 3) But wait!  Aren't we looking for CHR, not zero?
	 Good point.  So what we do is XOR LONGWORD with a longword,
	 each of whose bytes is CHR.  This turns each byte that is CHR
	 into a zero.  */


	/* Each round the main loop processes 16 bytes.  */

	ALIGN (4)

L(1):	movl (%eax), %ecx	/* get word (= 4 bytes) in question */
	movl $0xfefefeff, %edi	/* magic value */
	xorl %edx, %ecx		/* XOR with word c|c|c|c => bytes of str == c
				   are now 0 */
	addl %ecx, %edi		/* add the magic value to the word.  We get
				   carry bits reported for each byte which
				   is *not* 0 */

	/* According to the algorithm we had to reverse the effect of the
	   XOR first and then test the overflow bits.  But because the
	   following XOR would destroy the carry flag and it would (in a
	   representation with more than 32 bits) not alter then last
	   overflow, we can now test this condition.  If no carry is signaled
	   no overflow must have occurred in the last byte => it was 0.	*/
	jnc L(8)

	/* We are only interested in carry bits that change due to the
	   previous add, so remove original bits */
	xorl %ecx, %edi		/* ((word^charmask)+magic)^(word^charmask) */

	/* Now test for the other three overflow bits.  */
	orl $0xfefefeff, %edi	/* set all non-carry bits */
	incl %edi		/* add 1: if one carry bit was *not* set
				   the addition will not result in 0.  */

	/* If at least one byte of the word is CHR we don't get 0 in %edi.  */
	jnz L(8)		/* found it => return pointer */

	/* This process is unfolded four times for better performance.
	   we don't increment the source pointer each time.  Instead we
	   use offsets and increment by 16 in each run of the loop.  But
	   before probing for the matching byte we need some extra code
	   (following LL(13) below).  Even the len can be compared with
	   constants instead of decrementing each time.  */

	movl 4(%eax), %ecx	/* get word (= 4 bytes) in question */
	movl $0xfefefeff, %edi	/* magic value */
	xorl %edx, %ecx		/* XOR with word c|c|c|c => bytes of str == c
				   are now 0 */
	addl %ecx, %edi		/* add the magic value to the word.  We get
				   carry bits reported for each byte which
				   is *not* 0 */
	jnc L(7)		/* highest byte is CHR => return pointer */
	xorl %ecx, %edi		/* ((word^charmask)+magic)^(word^charmask) */
	orl $0xfefefeff, %edi	/* set all non-carry bits */
	incl %edi		/* add 1: if one carry bit was *not* set
				   the addition will not result in 0.  */
	jnz L(7)		/* found it => return pointer */

	movl 8(%eax), %ecx	/* get word (= 4 bytes) in question */
	movl $0xfefefeff, %edi	/* magic value */
	xorl %edx, %ecx		/* XOR with word c|c|c|c => bytes of str == c
				   are now 0 */
	addl %ecx, %edi		/* add the magic value to the word.  We get
				   carry bits reported for each byte which
				   is *not* 0 */
	jnc L(6)		/* highest byte is CHR => return pointer */
	xorl %ecx, %edi		/* ((word^charmask)+magic)^(word^charmask) */
	orl $0xfefefeff, %edi	/* set all non-carry bits */
	incl %edi		/* add 1: if one carry bit was *not* set
				   the addition will not result in 0.  */
	jnz L(6)		/* found it => return pointer */

	movl 12(%eax), %ecx	/* get word (= 4 bytes) in question */
	movl $0xfefefeff, %edi	/* magic value */
	xorl %edx, %ecx		/* XOR with word c|c|c|c => bytes of str == c
				   are now 0 */
	addl %ecx, %edi		/* add the magic value to the word.  We get
				   carry bits reported for each byte which
				   is *not* 0 */
	jnc L(5)		/* highest byte is CHR => return pointer */
	xorl %ecx, %edi		/* ((word^charmask)+magic)^(word^charmask) */
	orl $0xfefefeff, %edi	/* set all non-carry bits */
	incl %edi		/* add 1: if one carry bit was *not* set
				   the addition will not result in 0.  */
	jnz L(5)		/* found it => return pointer */

	/* Adjust both counters for a full round, i.e. 16 bytes.  */
	addl $16, %eax
L(2):	subl $16, %esi
	jae L(1)		/* Still more than 16 bytes remaining */

	/* Process remaining bytes separately.  */
	cmpl $4-16, %esi	/* rest < 4 bytes? */
	jb L(3)			/* yes, than test byte by byte */

	movl (%eax), %ecx	/* get word (= 4 bytes) in question */
	movl $0xfefefeff, %edi	/* magic value */
	xorl %edx, %ecx		/* XOR with word c|c|c|c => bytes of str == c
				   are now 0 */
	addl %ecx, %edi		/* add the magic value to the word.  We get
				   carry bits reported for each byte which
				   is *not* 0 */
	jnc L(8)		/* highest byte is CHR => return pointer */
	xorl %ecx, %edi		/* ((word^charmask)+magic)^(word^charmask) */
	orl $0xfefefeff, %edi	/* set all non-carry bits */
	incl %edi		/* add 1: if one carry bit was *not* set
				   the addition will not result in 0.  */
	jne L(8)		/* found it => return pointer */
	addl $4, %eax		/* adjust source pointer */

	cmpl $8-16, %esi	/* rest < 8 bytes? */
	jb L(3)			/* yes, than test byte by byte */

	movl (%eax), %ecx	/* get word (= 4 bytes) in question */
	movl $0xfefefeff, %edi	/* magic value */
	xorl %edx, %ecx		/* XOR with word c|c|c|c => bytes of str == c
				   are now 0 */
	addl %ecx, %edi		/* add the magic value to the word.  We get
				   carry bits reported for each byte which
				   is *not* 0 */
	jnc L(8)		/* highest byte is CHR => return pointer */
	xorl %ecx, %edi		/* ((word^charmask)+magic)^(word^charmask) */
	orl $0xfefefeff, %edi	/* set all non-carry bits */
	incl %edi		/* add 1: if one carry bit was *not* set
				   the addition will not result in 0.  */
	jne L(8)		/* found it => return pointer */
	addl $4, %eax		/* adjust source pointer */

	cmpl $12-16, %esi	/* rest < 12 bytes? */
	jb L(3)			/* yes, than test byte by byte */

	movl (%eax), %ecx	/* get word (= 4 bytes) in question */
	movl $0xfefefeff, %edi	/* magic value */
	xorl %edx, %ecx		/* XOR with word c|c|c|c => bytes of str == c
				   are now 0 */
	addl %ecx, %edi		/* add the magic value to the word.  We get
				   carry bits reported for each byte which
				   is *not* 0 */
	jnc L(8)		/* highest byte is CHR => return pointer */
	xorl %ecx, %edi		/* ((word^charmask)+magic)^(word^charmask) */
	orl $0xfefefeff, %edi	/* set all non-carry bits */
	incl %edi		/* add 1: if one carry bit was *not* set
				   the addition will not result in 0.  */
	jne L(8)		/* found it => return pointer */
	addl $4, %eax		/* adjust source pointer */

	/* Check the remaining bytes one by one.  */
L(3):	andl $3, %esi		/* mask out uninteresting bytes */
	jz L(4)			/* no remaining bytes => return NULL */

	cmpb %dl, (%eax)	/* compare byte with CHR */
	je L(9)			/* equal, than return pointer */
	incl %eax		/* increment source pointer */
	decl %esi		/* decrement length */
	jz L(4)			/* no remaining bytes => return NULL */

	cmpb %dl, (%eax)	/* compare byte with CHR */
	je L(9)			/* equal, than return pointer */
	incl %eax		/* increment source pointer */
	decl %esi		/* decrement length */
	jz L(4)			/* no remaining bytes => return NULL */

	cmpb %dl, (%eax)	/* compare byte with CHR */
	je L(9)			/* equal, than return pointer */

L(4):	/* no byte found => return NULL */
	xorl %eax, %eax
	jmp L(9)

	/* add missing source pointer increments */
L(5):	addl $4, %eax
L(6):	addl $4, %eax
L(7):	addl $4, %eax

	/* Test for the matching byte in the word.  %ecx contains a NUL
	   char in the byte which originally was the byte we are looking
	   at.  */
L(8):	testb %cl, %cl		/* test first byte in dword */
	jz L(9)			/* if zero => return pointer */
	incl %eax		/* increment source pointer */

	testb %ch, %ch		/* test second byte in dword */
	jz L(9)			/* if zero => return pointer */
	incl %eax		/* increment source pointer */

	testl $0xff0000, %ecx	/* test third byte in dword */
	jz L(9)			/* if zero => return pointer */
	incl %eax		/* increment source pointer */

	/* No further test needed we we know it is one of the four bytes.  */
L(9):	popl %edi		/* pop saved registers */
	cfi_adjust_cfa_offset (-4)
	cfi_restore (edi)
	popl %esi
	cfi_adjust_cfa_offset (-4)
	cfi_restore (esi)

	ret
END (__memchr)

weak_alias (__memchr, memchr)
libc_hidden_builtin_def (memchr)
