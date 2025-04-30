/* strcspn (str, ss) -- Return the length of the initial segment of STR
			which contains no characters from SS.
   For AMD x86-64.
   Copyright (C) 1994-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@gnu.ai.mit.edu>.
   Bug fixes by Alan Modra <Alan@SPRI.Levels.UniSA.Edu.Au>.
   Adopted for x86-64 by Andreas Jaeger <aj@suse.de>.

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

	.text
ENTRY (strcspn)

	movq %rdi, %rdx		/* Save SRC.  */

	/* First we create a table with flags for all possible characters.
	   For the ASCII (7bit/8bit) or ISO-8859-X character sets which are
	   supported by the C string functions we have 256 characters.
	   Before inserting marks for the stop characters we clear the whole
	   table.  */
	movq %rdi, %r8			/* Save value.  */
	subq $256, %rsp			/* Make space for 256 bytes.  */
	cfi_adjust_cfa_offset(256)
	movl $32,  %ecx			/* 32*8 bytes = 256 bytes.  */
	movq %rsp, %rdi
	xorl %eax, %eax			/* We store 0s.  */
	cld
	rep
	stosq

	movq %rsi, %rax			/* Setup skipset.  */

/* For understanding the following code remember that %rcx == 0 now.
   Although all the following instruction only modify %cl we always
   have a correct zero-extended 64-bit value in %rcx.  */

	.p2align 4
L(2):	movb (%rax), %cl	/* get byte from skipset */
	testb %cl, %cl		/* is NUL char? */
	jz L(1)			/* yes => start compare loop */
	movb %cl, (%rsp,%rcx)	/* set corresponding byte in skipset table */

	movb 1(%rax), %cl	/* get byte from skipset */
	testb $0xff, %cl	/* is NUL char? */
	jz L(1)			/* yes => start compare loop */
	movb %cl, (%rsp,%rcx)	/* set corresponding byte in skipset table */

	movb 2(%rax), %cl	/* get byte from skipset */
	testb $0xff, %cl	/* is NUL char? */
	jz L(1)			/* yes => start compare loop */
	movb %cl, (%rsp,%rcx)	/* set corresponding byte in skipset table */

	movb 3(%rax), %cl	/* get byte from skipset */
	addq $4, %rax		/* increment skipset pointer */
	movb %cl, (%rsp,%rcx)	/* set corresponding byte in skipset table */
	testb $0xff, %cl	/* is NUL char? */
	jnz L(2)		/* no => process next dword from skipset */

L(1):	leaq -4(%rdx), %rax	/* prepare loop */

	/* We use a neat trick for the following loop.  Normally we would
	   have to test for two termination conditions
	   1. a character in the skipset was found
	   and
	   2. the end of the string was found
	   But as a sign that the character is in the skipset we store its
	   value in the table.  But the value of NUL is NUL so the loop
	   terminates for NUL in every case.  */

	.p2align 4
L(3):	addq $4, %rax		/* adjust pointer for full loop round */

	movb (%rax), %cl	/* get byte from string */
	cmpb %cl, (%rsp,%rcx)	/* is it contained in skipset? */
	je L(4)			/* yes => return */

	movb 1(%rax), %cl	/* get byte from string */
	cmpb %cl, (%rsp,%rcx)	/* is it contained in skipset? */
	je L(5)			/* yes => return */

	movb 2(%rax), %cl	/* get byte from string */
	cmpb %cl, (%rsp,%rcx)	/* is it contained in skipset? */
	jz L(6)			/* yes => return */

	movb 3(%rax), %cl	/* get byte from string */
	cmpb %cl, (%rsp,%rcx)	/* is it contained in skipset? */
	jne L(3)		/* no => start loop again */

	incq %rax		/* adjust pointer */
L(6):	incq %rax
L(5):	incq %rax

L(4):	addq $256, %rsp		/* remove skipset */
	cfi_adjust_cfa_offset(-256)
#ifdef USE_AS_STRPBRK
	xorl %edx,%edx
	orb %cl, %cl		/* was last character NUL? */
	cmovzq %rdx, %rax	/* Yes:	return NULL */
#else
	subq %rdx, %rax		/* we have to return the number of valid
				   characters, so compute distance to first
				   non-valid character */
#endif
	ret
END (strcspn)
libc_hidden_builtin_def (strcspn)
