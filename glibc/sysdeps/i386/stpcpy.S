/* Copy SRC to DEST returning the address of the terminating '\0' in DEST.
   For Intel 80x86, x>=3.
   Copyright (C) 1994-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper (drepper@gnu.ai.mit.edu).

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

/* This function is defined neither in ANSI nor POSIX standards but is
   also not invented here.  */

#include <sysdep.h>
#include "asm-syntax.h"

#define PARMS	4		/* no space for saved regs */
#define RTN	PARMS
#define DEST	RTN
#define SRC	DEST+4

	.text
ENTRY (__stpcpy)

	movl DEST(%esp), %eax
	movl SRC(%esp), %ecx
	subl %eax, %ecx		/* magic: reduce number of loop variants
				   to one using addressing mode */

	/* Here we would like to write

	subl $4, %eax
	ALIGN (4)

	but the assembler is too smart and optimizes for the shortest
	form where the number only needs one byte.  But if we could
	have the long form we would not need the alignment.  */

	.byte 0x81, 0xe8	/* This is `subl $0x00000004, %eax' */
	.long 0x00000004

	/* Four times unfolded loop with only one loop counter.  This
	   is achieved by the use of index+base addressing mode.  As the
	   loop counter we use the destination address because this is
	   also the result.  */
L(1):	addl $4, %eax		/* increment loop counter */

	movb (%eax,%ecx), %dl	/* load current char */
	movb %dl, (%eax)	/* and store it */
	testb %dl, %dl		/* was it NUL? */
	jz L(2)			/* yes, then exit */

	movb 1(%eax,%ecx), %dl	/* load current char */
	movb %dl, 1(%eax)	/* and store it */
	testb %dl, %dl		/* was it NUL? */
	jz L(3)			/* yes, then exit */

	movb 2(%eax,%ecx), %dl	/* load current char */
	movb %dl, 2(%eax)	/* and store it */
	testb %dl, %dl		/* was it NUL? */
	jz L(4)			/* yes, then exit */

	movb 3(%eax,%ecx), %dl	/* load current char */
	movb %dl, 3(%eax)	/* and store it */
	testb %dl, %dl		/* was it NUL? */
	jnz L(1)		/* no, then continue loop */

	incl %eax		/* correct loop counter */
L(4):	incl %eax
L(3):	incl %eax
L(2):

	ret
END (__stpcpy)

weak_alias (__stpcpy, stpcpy)
libc_hidden_def (__stpcpy)
libc_hidden_builtin_def (stpcpy)
