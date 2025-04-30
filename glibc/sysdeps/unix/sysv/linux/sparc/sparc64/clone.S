/* Copyright (C) 1997-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Richard Henderson (rth@tamu.edu).

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

/* clone() is even more special than fork() as it mucks with stacks
   and invokes a function in the right context after its all over.  */

#include <asm/errno.h>
#include <tcb-offsets.h>
#include <sysdep.h>

/* int clone(int (*fn)(void *arg), void *child_stack, int flags, void *arg,
	     pid_t *ptid, void *tls, pid_t *ctid); */

	.register	%g2,#scratch
	.register	%g3,#scratch

	.text

ENTRY (__clone)
	save	%sp, -192, %sp
	cfi_def_cfa_register(%fp)
	cfi_window_save
	cfi_register(%o7, %i7)

	/* sanity check arguments */
	brz,pn	%i0, 99f		/* fn non-NULL? */
	 mov	%i0, %g2
	brz,pn	%i1, 99f		/* child_stack non-NULL? */
	 mov	%i2, %o0		/* clone flags */

	/* The child_stack is the top of the stack, allocate one
	   whole stack frame from that as this is what the kernel
	   expects.  Also, subtract STACK_BIAS.  */
	sub	%i1, 192 + 0x7ff, %o1
	mov	%i3, %g3

	mov	%i4,%o2			/* PTID */
	mov	%i5,%o3			/* TLS */
	ldx	[%fp+0x7ff+176],%o4	/* CTID */

	/* Do the system call */
	set	__NR_clone, %g1
	ta	0x6d
	bcs,pn	%xcc, 98f
	 nop
	brnz,pn	%o1, __thread_start
	 nop
	jmpl	%i7 + 8, %g0
	 restore %o0, %g0, %o0
99:	mov	EINVAL, %o0
98:	call	HIDDEN_JUMPTARGET(__errno_location)
	 mov	%o0, %i0
	st	%i0, [%o0]
	jmpl	%i7 + 8, %g0
	 restore %g0,-1,%o0
END(__clone)

	.type __thread_start,@function
__thread_start:
	mov	%g0, %fp	/* terminate backtrace */
	call	%g2
	 mov	%g3,%o0
	set	__NR_exit, %g1
	ta	0x6d
	 nop

	.size	__thread_start, .-__thread_start

libc_hidden_def (__clone)
weak_alias (__clone, clone)
