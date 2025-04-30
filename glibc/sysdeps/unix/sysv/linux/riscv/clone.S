/* Wrapper around clone system call.  RISC-V version.
   Copyright (C) 1996-2021 Free Software Foundation, Inc.
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
   License along with the GNU C Library.  If not, see
   <https://www.gnu.org/licenses/>.  */

/* clone() is even more special than fork() as it mucks with stacks
   and invokes a function in the right context after its all over.  */

#include <sys/asm.h>
#include <sysdep.h>
#define _ERRNO_H	1
#include <bits/errno.h>
#include <tls.h>
#include "tcb-offsets.h"

/* int clone(int (*fn)(void *arg), void *child_stack, int flags, void *arg,
	     void *parent_tidptr, void *tls, void *child_tidptr) */

	.text
LEAF (__clone)

	/* Sanity check arguments.  */
	beqz		a0,L (invalid)	/* No NULL function pointers.  */
	beqz		a1,L (invalid)	/* No NULL stack pointers.  */

	addi		a1,a1,-16	/* Reserve argument save space.  */
	REG_S		a0,0(a1)	/* Save function pointer.  */
	REG_S		a3,SZREG(a1)	/* Save argument pointer.  */

	/* The syscall expects the args to be in different slots.  */
	mv		a0,a2
	mv		a2,a4
	mv		a3,a5
	mv		a4,a6

	/* Do the system call.  */
	li		a7,__NR_clone
	scall

	bltz		a0,L (error)
	beqz		a0,L (thread_start)

	/* Successful return from the parent.  */
	ret

L (invalid):
	li		a0, -EINVAL
	/* Something bad happened -- no child created.  */
L (error):
	j		__syscall_error
	END (__clone)

/* Load up the arguments to the function.  Put this block of code in
   its own function so that we can terminate the stack trace with our
   debug info.  */

ENTRY (__thread_start)
L (thread_start):
	/* Terminate call stack by noting ra is undefined.  Use a dummy
	   .cfi_label to force starting the FDE.  */
	.cfi_label .Ldummy
	cfi_undefined (ra)

	/* Restore the arg for user's function.  */
	REG_L		a1,0(sp)	/* Function pointer.  */
	REG_L		a0,SZREG(sp)	/* Argument pointer.  */

	/* Call the user's function.  */
	jalr		a1

	/* Call exit with the function's return value.  */
	li		a7, __NR_exit
	scall

	END (__thread_start)

libc_hidden_def (__clone)
weak_alias (__clone, clone)
