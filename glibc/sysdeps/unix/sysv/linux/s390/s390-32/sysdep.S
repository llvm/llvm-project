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
#include <tls.h>

/* The following code is only used in the shared library when we
   compile the reentrant version.  Otherwise each system call defines
   each own version.  */

/* The syscall stubs jump here when they detect an error.  */

#undef CALL_MCOUNT
#define CALL_MCOUNT

	.text
ENTRY(__syscall_error)
#ifndef PIC
# if IS_IN (libc)
#  define SYSCALL_ERROR_ERRNO __libc_errno
# else
#  define SYSCALL_ERROR_ERRNO errno
# endif
	basr	%r1,0
0:	l	%r1,1f-0b(%r1)
	ear	%r3,%a0
	lcr	%r2,%r2
	st	%r2,0(%r1,%r3)
	lhi	%r2,-1
	br	%r14
1:	.long	SYSCALL_ERROR_ERRNO@ntpoff
#else
# if RTLD_PRIVATE_ERRNO
	basr	%r1,0
0:	al	%r1,1f-0b(%r1)
	lcr	%r2,%r2
	st	%r2,0(%r1)
	lhi	%r2,-1
	br	%r14
1:	.long	rtld_errno - 0b
# else
#  if IS_IN (libc)
#   define SYSCALL_ERROR_ERRNO __libc_errno
#  else
#   define SYSCALL_ERROR_ERRNO errno
#  endif
	basr	%r1,0
0:	al	%r1,1f-0b(%r1)
	ear	%r3,%a0
	l	%r1,SYSCALL_ERROR_ERRNO@gotntpoff(%r1)
	lcr	%r2,%r2
	st	%r2,0(%r1,%r3)
	lhi	%r2,-1
	br	%r14
1:	.long	_GLOBAL_OFFSET_TABLE_-0b
# endif
#endif

END (__syscall_error)
