/* Copyright (C) 1996-2021 Free Software Foundation, Inc.
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

	.text
ENTRY (__clone)
	save	%sp,-96,%sp
	cfi_def_cfa_register(%fp)
	cfi_window_save
	cfi_register(%o7, %i7)

	/* sanity check arguments */
	orcc	%i0,%g0,%g2
	be	.Leinval
	 orcc	%i1,%g0,%o1
	be	.Leinval
	 mov	%i2,%o0

	/* The child_stack is the top of the stack, allocate one
	   whole stack frame from that as this is what the kernel
	   expects.  */
	sub	%o1, 96, %o1
	mov	%i3, %g3

	/* ptid */
	mov	%i4,%o2
	/* tls */
	mov	%i5,%o3
	/* ctid */
	ld	[%fp+92],%o4

	/* Do the system call */
	set	__NR_clone,%g1
	ta	0x10
	bcs	.Lerror
	 tst	%o1
	bne	__thread_start
	 nop
	jmpl	%i7 + 8, %g0
	 restore %o0,%g0,%o0

.Leinval:
	mov	EINVAL, %o0
.Lerror:
	call	HIDDEN_JUMPTARGET(__errno_location)
	 mov	%o0, %i0
	st	%i0,[%o0]
	jmpl	%i7 + 8, %g0
	 restore %g0,-1,%o0
END(__clone)

	.type	__thread_start,@function
__thread_start:
	mov	%g0, %fp	/* terminate backtrace */
	call	%g2
	 mov	%g3,%o0
	set	__NR_exit, %g1
	ta	0x10
	 nop

	.size	__thread_start, .-__thread_start

libc_hidden_def (__clone)
weak_alias (__clone, clone)
