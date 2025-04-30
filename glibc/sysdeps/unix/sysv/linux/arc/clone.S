/* clone() implementation for ARC.
   Copyright (C) 2020-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Andrew Jenner <andrew@codesourcery.com>, 2008.

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
#define _ERRNO_H	1
#include <bits/errno.h>
#include <tcb-offsets.h>

#define CLONE_SETTLS		0x00080000

/* int clone(int (*fn)(void *), void *child_stack,
           int flags, void *arg, ...
           < pid_t *ptid, struct user_desc *tls, pid_t *ctid > );

 NOTE: I'm assuming that the last 3 args are NOT var-args and in case all
	3 are not relevant, caller will nevertheless pass those as NULL.

 clone syscall in kernel (ABI: CONFIG_CLONE_BACKWARDS)

  int sys_clone(unsigned long int clone_flags,
	        unsigned long int newsp,
		int __user *parent_tidptr,
		void *tls,
		int __user *child_tidptr).  */

ENTRY (__clone)
	cmp	r0, 0		/* @fn can't be NULL.  */
	cmp.ne	r1, 0		/* @child_stack can't be NULL.  */
	bz	L (__sys_err)

	/* save some of the orig args
	   r0 containg @fn will be clobbered AFTER syscall (with ret val)
	   rest are clobbered BEFORE syscall due to different arg ordering.  */
	mov	r10, r0		/* @fn.  */
	mov	r11, r3		/* @args.  */
	mov	r12, r2		/* @clone_flags.  */
	mov	r9,  r5		/* @tls.  */

	/* adjust libc args for syscall.  */

	mov 	r0, r2		/* libc @flags is 1st syscall arg.  */
	mov	r2, r4		/* libc @ptid.  */
	mov	r3, r5		/* libc @tls.  */
	mov	r4, r6		/* libc @ctid.  */
	mov	r8, __NR_clone
	ARC_TRAP_INSN

	cmp	r0, 0		/* return code : 0 new process, !0 parent.  */
	blt	L (__sys_err2)	/* < 0 (signed) error.  */
	jnz	[blink]		/* Parent returns.  */

	/* child jumps off to @fn with @arg as argument
           TP register already set by kernel.  */
	jl.d	[r10]
	mov	r0, r11

	/* exit() with result from @fn (already in r0).  */
	mov	r8, __NR_exit
	ARC_TRAP_INSN
	/* In case it ever came back.  */
	flag	1

L (__sys_err):
	mov	r0, -EINVAL
L (__sys_err2):
	/* (1) No need to make -ve kernel error code as positive errno
	       __syscall_error expects the -ve error code returned by kernel
	   (2) r0 still had orig -ve kernel error code
	   (3) Tail call to __syscall_error so we dont have to come back
	       here hence instead of jmp-n-link (reg push/pop) we do jmp
	   (4) No need to route __syscall_error via PLT, B is inherently
	       position independent.  */
	b   __syscall_error
PSEUDO_END (__clone)
libc_hidden_def (__clone)
weak_alias (__clone, clone)
