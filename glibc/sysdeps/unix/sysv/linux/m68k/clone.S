/* Copyright (C) 1996-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Andreas Schwab (schwab@issan.informatik.uni-dortmund.de)

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

/* clone() is even more special than fork() as it mucks with stacks
   and invokes a function in the right context after its all over.  */

#include <sysdep.h>
#define _ERRNO_H	1
#include <bits/errno.h>
#include <tls.h>

/* int clone(int (*fn)(void *arg), void *child_stack, int flags, void *arg,
	     void *parent_tidptr, void *tls, void *child_tidptr) */

        .text
ENTRY (__clone)

	/* Sanity check arguments.  */
	movel	#-EINVAL, %d0
	movel	4(%sp), %a0		/* no NULL function pointers */
	tstl	%a0
	jeq	SYSCALL_ERROR_LABEL
	movel	8(%sp), %a1		/* no NULL stack pointers */
	tstl	%a1
	jeq	SYSCALL_ERROR_LABEL

	/* Allocate space and copy the argument onto the new stack.  */
	movel	16(%sp), -(%a1)

	/* Do the system call */
	movel	12+0(%sp), %d1		/* get flags */
	movel	%d3, -(%a1)             /* save %d3 and get parent_tidptr */
	movel	%d3, -(%sp)
	cfi_adjust_cfa_offset (4)
	cfi_rel_offset (%d3, 0)
	movel	20+4(%sp), %d3
	movel	%d4, -(%a1)		/* save %d4 and get child_tidptr */
	movel	%d4, -(%sp)
	cfi_adjust_cfa_offset (4)
	cfi_rel_offset (%d4, 0)
	movel	28+8(%sp), %d4
	movel	%d5, -(%a1)             /* save %d5 and get tls */
	movel	%d5, -(%sp)
	cfi_adjust_cfa_offset (4)
	cfi_rel_offset (%d5, 0)
	movel	24+12(%sp), %d5
	/* save %d2 and get stack pointer */
#ifdef __mcoldfire__
	movel	%d2, -(%a1)
	movel	%d2, -(%sp)
	cfi_adjust_cfa_offset (4)
	cfi_rel_offset (%d2, 0)
	movel	%a1, %d2
#else
	exg	%d2, %a1		/* save %d2 and get stack pointer */
	cfi_register (%d2, %a1)
#endif
	movel	#SYS_ify (clone), %d0

	/* End FDE now, because in the child the unwind info will be
	   wrong.  */
	cfi_endproc

	trap	#0
#ifdef __mcoldfire__
	movel	(%sp)+, %d2
#else
	exg	%d2, %a1		/* restore %d2 */
#endif
	movel	(%sp)+, %d5             /* restore %d5, %d4 and %d3 */
	movel	(%sp)+, %d4
	movel	(%sp)+, %d3

	tstl	%d0
	jmi	SYSCALL_ERROR_LABEL
	jeq	1f

	rts

1:
	cfi_startproc
	cfi_undefined (pc)	/* Mark end of stack */
	subl	%fp, %fp	/* terminate the stack frame */
	jsr	(%a0)
	movel	%d0, %d1
	movel	#SYS_ify (exit), %d0
	trap	#0
	cfi_endproc

	cfi_startproc
PSEUDO_END (__clone)

libc_hidden_def (__clone)
weak_alias (__clone, clone)
