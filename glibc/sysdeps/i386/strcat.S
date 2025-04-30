/* strcat(dest, src) -- Append SRC on the end of DEST.
   For Intel 80x86, x>=4.
   Copyright (C) 1994-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@ipd.info.uni-karlsruhe.de>.
   Optimised a little by Alan Modra <Alan@SPRI.Levels.UniSA.Edu.Au>

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

#define PARMS	4+4	/* space for 1 saved reg */
#define RTN	PARMS
#define DEST	RTN
#define SRC	DEST+4

	.text
ENTRY (strcat)

	pushl %edi		/* Save callee-safe register.  */
	cfi_adjust_cfa_offset (4)

	movl DEST(%esp), %edx
	movl SRC(%esp), %ecx

	testb $0xff, (%ecx)	/* Is source string empty? */
	jz L(8)			/* yes => return */

	/* Test the first bytes separately until destination is aligned.  */
	testl $3, %edx		/* destination pointer aligned? */
	jz L(1)			/* yes => begin scan loop */
	testb $0xff, (%edx)	/* is end of string? */
	jz L(2)			/* yes => start appending */
	incl %edx		/* increment source pointer */

	testl $3, %edx		/* destination pointer aligned? */
	jz L(1)			/* yes => begin scan loop */
	testb $0xff, (%edx)	/* is end of string? */
	jz L(2)			/* yes => start appending */
	incl %edx		/* increment source pointer */

	testl $3, %edx		/* destination pointer aligned? */
	jz L(1)			/* yes => begin scan loop */
	testb $0xff, (%edx)	/* is end of string? */
	jz L(2)			/* yes => start appending */
	incl %edx		/* increment source pointer */

	/* Now we are aligned.  Begin scan loop.  */
	jmp L(1)

	cfi_rel_offset (edi, 0)
	ALIGN(4)

L(4):	addl $16,%edx		/* increment destination pointer for round */

L(1):	movl (%edx), %eax	/* get word (= 4 bytes) in question */
	movl $0xfefefeff, %edi	/* magic value */

	/* If you compare this with the algorithm in memchr.S you will
	   notice that here is an `xorl' statement missing.  But you must
	   not forget that we are looking for C == 0 and `xorl $0, %eax'
	   is a no-op.  */

	addl %eax, %edi		/* add the magic value to the word.  We get
				   carry bits reported for each byte which
				   is *not* 0 */

	/* According to the algorithm we had to reverse the effect of the
	   XOR first and then test the overflow bits.  But because the
	   following XOR would destroy the carry flag and it would (in a
	   representation with more than 32 bits) not alter then last
	   overflow, we can now test this condition.  If no carry is signaled
	   no overflow must have occurred in the last byte => it was 0.	*/
	jnc L(3)

	/* We are only interested in carry bits that change due to the
	   previous add, so remove original bits */
	xorl %eax, %edi		/* ((word^charmask)+magic)^(word^charmask) */

	/* Now test for the other three overflow bits.  */
	orl $0xfefefeff, %edi	/* set all non-carry bits */
	incl %edi		/* add 1: if one carry bit was *not* set
				   the addition will not result in 0.  */

	/* If at least one byte of the word is C we don't get 0 in %ecx.  */
	jnz L(3)

	movl 4(%edx), %eax	/* get word from source */
	movl $0xfefefeff, %edi	/* magic value */
	addl %eax, %edi		/* add the magic value to the word.  We get
				   carry bits reported for each byte which
				   is *not* 0 */
	jnc L(5)		/* highest byte is C => stop copying */
	xorl %eax, %edi		/* ((word^charmask)+magic)^(word^charmask) */
	orl $0xfefefeff, %edi	/* set all non-carry bits */
	incl %edi		/* add 1: if one carry bit was *not* set
				   the addition will not result in 0.  */
	jnz L(5)		/* one byte is NUL => stop copying */

	movl 8(%edx), %eax	/* get word from source */
	movl $0xfefefeff, %edi	/* magic value */
	addl %eax, %edi		/* add the magic value to the word.  We get
				   carry bits reported for each byte which
				   is *not* 0 */
	jnc L(6)		/* highest byte is C => stop copying */
	xorl %eax, %edi		/* ((word^charmask)+magic)^(word^charmask) */
	orl $0xfefefeff, %edi	/* set all non-carry bits */
	incl %edi		/* add 1: if one carry bit was *not* set
				   the addition will not result in 0.  */
	jnz L(6)		/* one byte is NUL => stop copying */

	movl 12(%edx), %eax	/* get word from source */
	movl $0xfefefeff, %edi	/* magic value */
	addl %eax, %edi		/* add the magic value to the word.  We get
				   carry bits reported for each byte which
				   is *not* 0 */
	jnc L(7)		/* highest byte is C => stop copying */
	xorl %eax, %edi		/* ((word^charmask)+magic)^(word^charmask) */
	orl $0xfefefeff, %edi	/* set all non-carry bits */
	incl %edi		/* add 1: if one carry bit was *not* set
				   the addition will not result in 0.  */
	jz L(4)			/* no byte is NUL => carry on copying */

L(7):	addl $4, %edx		/* adjust source pointer */
L(6):	addl $4, %edx
L(5):	addl $4, %edx

L(3):	testb %al, %al		/* is first byte NUL? */
	jz L(2)			/* yes => start copying */
	incl %edx		/* increment source pointer */

	testb %ah, %ah		/* is second byte NUL? */
	jz L(2)			/* yes => start copying */
	incl %edx		/* increment source pointer */

	testl $0xff0000, %eax	/* is third byte NUL? */
	jz L(2)			/* yes => start copying */
	incl %edx		/* increment source pointer */

L(2):	subl %ecx, %edx		/* reduce number of loop variants */

	/* Now we have to align the source pointer.  */
	testl $3, %ecx		/* pointer correctly aligned? */
	jz L(29)		/* yes => start copy loop */
	movb (%ecx), %al	/* get first byte */
	movb %al, (%ecx,%edx)	/* and store it */
	andb %al, %al		/* is byte NUL? */
	jz L(8)			/* yes => return */
	incl %ecx		/* increment pointer */

	testl $3, %ecx		/* pointer correctly aligned? */
	jz L(29)		/* yes => start copy loop */
	movb (%ecx), %al	/* get first byte */
	movb %al, (%ecx,%edx)	/* and store it */
	andb %al, %al		/* is byte NUL? */
	jz L(8)			/* yes => return */
	incl %ecx		/* increment pointer */

	testl $3, %ecx		/* pointer correctly aligned? */
	jz L(29)		/* yes => start copy loop */
	movb (%ecx), %al	/* get first byte */
	movb %al, (%ecx,%edx)	/* and store it */
	andb %al, %al		/* is byte NUL? */
	jz L(8)			/* yes => return */
	incl %ecx		/* increment pointer */

	/* Now we are aligned.  */
	jmp L(29)		/* start copy loop */

	ALIGN(4)

L(28):	movl %eax, 12(%ecx,%edx)/* store word at destination */
	addl $16, %ecx		/* adjust pointer for full round */

L(29):	movl (%ecx), %eax	/* get word from source */
	movl $0xfefefeff, %edi	/* magic value */
	addl %eax, %edi		/* add the magic value to the word.  We get
				   carry bits reported for each byte which
				   is *not* 0 */
	jnc L(9)		/* highest byte is C => stop copying */
	xorl %eax, %edi		/* ((word^charmask)+magic)^(word^charmask) */
	orl $0xfefefeff, %edi	/* set all non-carry bits */
	incl %edi		/* add 1: if one carry bit was *not* set
				   the addition will not result in 0.  */
	jnz L(9)		/* one byte is NUL => stop copying */
	movl %eax, (%ecx,%edx)	/* store word to destination */

	movl 4(%ecx), %eax	/* get word from source */
	movl $0xfefefeff, %edi	/* magic value */
	addl %eax, %edi		/* add the magic value to the word.  We get
				   carry bits reported for each byte which
				   is *not* 0 */
	jnc L(91)		/* highest byte is C => stop copying */
	xorl %eax, %edi		/* ((word^charmask)+magic)^(word^charmask) */
	orl $0xfefefeff, %edi	/* set all non-carry bits */
	incl %edi		/* add 1: if one carry bit was *not* set
				   the addition will not result in 0.  */
	jnz L(91)		/* one byte is NUL => stop copying */
	movl %eax, 4(%ecx,%edx)	/* store word to destination */

	movl 8(%ecx), %eax	/* get word from source */
	movl $0xfefefeff, %edi	/* magic value */
	addl %eax, %edi		/* add the magic value to the word.  We get
				   carry bits reported for each byte which
				   is *not* 0 */
	jnc L(92)		/* highest byte is C => stop copying */
	xorl %eax, %edi		/* ((word^charmask)+magic)^(word^charmask) */
	orl $0xfefefeff, %edi	/* set all non-carry bits */
	incl %edi		/* add 1: if one carry bit was *not* set
				   the addition will not result in 0.  */
	jnz L(92)		/* one byte is NUL => stop copying */
	movl %eax, 8(%ecx,%edx)	/* store word to destination */

	movl 12(%ecx), %eax	/* get word from source */
	movl $0xfefefeff, %edi	/* magic value */
	addl %eax, %edi		/* add the magic value to the word.  We get
				   carry bits reported for each byte which
				   is *not* 0 */
	jnc L(93)		/* highest byte is C => stop copying */
	xorl %eax, %edi		/* ((word^charmask)+magic)^(word^charmask) */
	orl $0xfefefeff, %edi	/* set all non-carry bits */
	incl %edi		/* add 1: if one carry bit was *not* set
				   the addition will not result in 0.  */
	jz L(28)		/* no is NUL => carry on copying */

L(93):	addl $4, %ecx		/* adjust pointer */
L(92):	addl $4, %ecx
L(91):	addl $4, %ecx

L(9):	movb %al, (%ecx,%edx)	/* store first byte of last word */
	orb %al, %al		/* is it NUL? */
	jz L(8)			/* yes => return */

	movb %ah, 1(%ecx,%edx)	/* store second byte of last word */
	orb %ah, %ah		/* is it NUL? */
	jz L(8)			/* yes => return */

	shrl $16, %eax		/* make upper bytes accessible */
	movb %al, 2(%ecx,%edx)	/* store third byte of last word */
	orb %al, %al		/* is it NUL? */
	jz L(8)			/* yes => return */

	movb %ah, 3(%ecx,%edx)	/* store fourth byte of last word */

L(8):	movl DEST(%esp), %eax	/* start address of destination is result */
	popl %edi		/* restore saved register */
	cfi_adjust_cfa_offset (-4)
	cfi_restore (edi)

	ret
END (strcat)
libc_hidden_builtin_def (strcat)
