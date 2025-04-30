/* strlen(str) -- determine the length of the string STR.
   Optimized for Intel 80x86, x>=4.
   Copyright (C) 1991-2021 Free Software Foundation, Inc.
   Contributed by Ulrich Drepper <drepper@gnu.ai.mit.edu>.
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
#include "asm-syntax.h"

#define PARMS	4		/* no space for saved regs */
#define STR	PARMS

	.text
ENTRY (strlen)

	movl STR(%esp), %ecx
	movl %ecx, %eax		/* duplicate it */

	andl $3, %ecx		/* mask alignment bits */
	jz L(1)			/* aligned => start loop */
	cmpb %ch, (%eax)	/* is byte NUL? */
	je L(2)			/* yes => return */
	incl %eax		/* increment pointer */

	xorl $3, %ecx		/* was alignment = 3? */
	jz L(1)			/* yes => now it is aligned and start loop */
	cmpb %ch, (%eax)	/* is byte NUL? */
	je L(2)			/* yes => return */
	addl $1, %eax		/* increment pointer */

	subl $1, %ecx		/* was alignment = 2? */
	jz L(1)			/* yes => now it is aligned and start loop */
	cmpb %ch, (%eax)	/* is byte NUL? */
	je L(2)			/* yes => return */

/* Don't change the above `addl $1,%eax' and `subl $1, %ecx' into `incl %eax'
   and `decl %ecx' resp.  The additional two byte per instruction make the
   label 4 to be aligned on a 16 byte boundary with nops.

   The following `sub $15, %eax' is part of this trick, too.  Together with
   the next instruction (`addl $16, %eax') it is in fact a `incl %eax', just
   as expected from the algorithm.  But doing so has the advantage that
   no jump to label 1 is necessary and so the pipeline is not flushed.  */

	subl $15, %eax		/* effectively +1 */


L(4):	addl $16, %eax		/* adjust pointer for full loop */

L(1):	movl (%eax), %ecx	/* get word (= 4 bytes) in question */
	movl $0xfefefeff, %edx	/* magic value */
	addl %ecx, %edx		/* add the magic value to the word.  We get
				   carry bits reported for each byte which
				   is *not* 0 */
	jnc L(3)		/* highest byte is NUL => return pointer */
	xorl %ecx, %edx		/* (word+magic)^word */
	orl $0xfefefeff, %edx	/* set all non-carry bits */
	incl %edx		/* add 1: if one carry bit was *not* set
				   the addition will not result in 0.  */
	jnz L(3)		/* found NUL => return pointer */

	movl 4(%eax), %ecx	/* get word (= 4 bytes) in question */
	movl $0xfefefeff, %edx	/* magic value */
	addl %ecx, %edx		/* add the magic value to the word.  We get
				   carry bits reported for each byte which
				   is *not* 0 */
	jnc L(5)		/* highest byte is NUL => return pointer */
	xorl %ecx, %edx		/* (word+magic)^word */
	orl $0xfefefeff, %edx	/* set all non-carry bits */
	incl %edx		/* add 1: if one carry bit was *not* set
				   the addition will not result in 0.  */
	jnz L(5)		/* found NUL => return pointer */

	movl 8(%eax), %ecx	/* get word (= 4 bytes) in question */
	movl $0xfefefeff, %edx	/* magic value */
	addl %ecx, %edx		/* add the magic value to the word.  We get
				   carry bits reported for each byte which
				   is *not* 0 */
	jnc L(6)		/* highest byte is NUL => return pointer */
	xorl %ecx, %edx		/* (word+magic)^word */
	orl $0xfefefeff, %edx	/* set all non-carry bits */
	incl %edx		/* add 1: if one carry bit was *not* set
				   the addition will not result in 0.  */
	jnz L(6)		/* found NUL => return pointer */

	movl 12(%eax), %ecx	/* get word (= 4 bytes) in question */
	movl $0xfefefeff, %edx	/* magic value */
	addl %ecx, %edx		/* add the magic value to the word.  We get
				   carry bits reported for each byte which
				   is *not* 0 */
	jnc L(7)		/* highest byte is NUL => return pointer */
	xorl %ecx, %edx		/* (word+magic)^word */
	orl $0xfefefeff, %edx	/* set all non-carry bits */
	incl %edx		/* add 1: if one carry bit was *not* set
				   the addition will not result in 0.  */
	jz L(4)			/* no NUL found => continue loop */

L(7):	addl $4, %eax		/* adjust pointer */
L(6):	addl $4, %eax
L(5):	addl $4, %eax

L(3):	testb %cl, %cl		/* is first byte NUL? */
	jz L(2)			/* yes => return */
	incl %eax		/* increment pointer */

	testb %ch, %ch		/* is second byte NUL? */
	jz L(2)			/* yes => return */
	incl %eax		/* increment pointer */

	testl $0xff0000, %ecx	/* is third byte NUL? */
	jz L(2)			/* yes => return pointer */
	incl %eax		/* increment pointer */

L(2):	subl STR(%esp), %eax	/* compute difference to string start */

	ret
END (strlen)
libc_hidden_builtin_def (strlen)
