/* Copyright (C) 2001-2021 Free Software Foundation, Inc.
   Contributed by Martin Schwidefsky (schwidefsky@de.ibm.com).
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

/* Please consult the file sysdeps/unix/sysv/linux/s390/sysdep.h for
   more information about the value -4095 used below.*/

ENTRY (syscall)
	stmg   %r6,%r7,48(%r15)    /* Save registers.  */
	cfi_offset (%r7,-104)
	cfi_offset (%r6,-112)

	lgr    %r1,%r2		   /* Move syscall number.  */
	lgr    %r2,%r3		   /* First parameter.	*/
	lgr    %r3,%r4		   /* Second parameter.	 */
	lgr    %r4,%r5		   /* Third parameter.	*/
	lgr    %r5,%r6		   /* Fourth parameter.	 */
	lmg    %r6,%r7,160(%r15)   /* Fifth / Sixth parameter.	*/
	svc    0

	lmg    %r6,%r7,48(%r15)    /* Load registers.  */

	lghi   %r0,-4095
	clgr   %r2,%r0		   /* Check R2 for error.  */
	jgnl   SYSCALL_ERROR_LABEL
	br     %r14		   /* Return to caller.	 */
PSEUDO_END (syscall)
