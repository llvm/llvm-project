/* Static library error handling code fragment for Nios II.
   Copyright (C) 2015-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public License as
   published by the Free Software Foundation; either version 2.1 of the
   License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; if not, see
   <https://www.gnu.org/licenses/>.  */

#include <sysdep.h>

/* The following code is only used in the static library. In the shared
   library, the error handling code is at the end of each function.  */

#ifndef	PIC

/* In the static library, the syscall stubs jump here when they detect
   an error.  */

# undef CALL_MCOUNT
# define CALL_MCOUNT /* Don't insert the profiling call, it clobbers r2.  */

# if IS_IN (libc)
#  define SYSCALL_ERROR_ERRNO __libc_errno
# else
#  define SYSCALL_ERROR_ERRNO errno
# endif
	.text
ENTRY (__syscall_error)
	nextpc	r3
1:
	movhi	r8, %hiadj(_gp_got - 1b)
	addi	r8, r8, %lo(_gp_got - 1b)
	add	r3, r3, r8
	ldw	r3, %tls_ie(SYSCALL_ERROR_ERRNO)(r3)
	add	r3, r23, r3
	stw	r2, 0(r3)
	movi	r2, -1
	ret
END (__syscall_error)
#endif
