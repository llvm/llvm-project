/* Copyright (C) 1999-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by David Mosberger-Tang <davidm@hpl.hp.com>.

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
#include <features.h>
#include <tls.h>

ENTRY(__syscall_error)
#if RTLD_PRIVATE_ERRNO
	/*
	 * Note that the gp has to be set properly for this to work.
	 * As long as all syscalls are in the same load unit
	 * (executable or shared library) as this routine, we should
	 * be fine.  Otherwise, we would have to first load the global
	 * pointer register from __gp.
	 */
	addl	r2=@gprel(rtld_errno),gp
	;;
	st4	[r2]=r8
	mov	r8=-1
#else
# if IS_IN (libc)
#  define SYSCALL_ERROR_ERRNO __libc_errno
# else
#  define SYSCALL_ERROR_ERRNO errno
# endif
	addl	r2=@ltoff(@tprel(SYSCALL_ERROR_ERRNO)), gp;;
	ld8	r2=[r2]
	mov	r3=r8;;
	mov	r8=-1
	add	r2=r2,r13;;
	st4	[r2]=r3
#endif
	ret			// ret is #define'd in syscall.h!
END(__syscall_error)

ENTRY(__ia64_syscall)
	mov r15=r37		/* syscall number */
	break __IA64_BREAK_SYSCALL
	cmp.eq p6,p0=-1,r10	/* r10 = -1 on error */
(p6)	br.cond.spnt.few __syscall_error
	ret
PSEUDO_END(__ia64_syscall)
