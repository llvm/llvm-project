/* Copyright (C) 2000-2021 Free Software Foundation, Inc.
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
	stm    %r6,%r7,24(%r15)    /* save registers */
	cfi_offset (%r7, -68)
	cfi_offset (%r6, -72)

	lr     %r1,%r2             /* move syscall number */
	lr     %r2,%r3             /* first parameter  */
	lr     %r3,%r4             /* second parameter */
	lr     %r4,%r5             /* third parameter  */
	lr     %r5,%r6             /* fourth parameter */
	lm     %r6,%r7,96(%r15)    /* fifth / sixth parameter  */
	svc    0

	lm     %r6,%r7,24(%r15)    /* load registers.  */

	lhi    %r0,-4095
	clr    %r2,%r0             /* check R2 for error */
	jnl    SYSCALL_ERROR_LABEL
	br     %r14                /* return to caller */
PSEUDO_END (syscall)
