/* Copyright (C) 2002-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@redhat.com>, 2002.

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
#include <errno.h>
#include <shlib-compat.h>

ENTRY (__pthread_spin_trylock)
	movl	4(%esp), %edx
	movl	$1, %eax
	xorl	%ecx, %ecx
	lock
	cmpxchgl %ecx, (%edx)
	movl	$EBUSY, %eax
#ifdef HAVE_CMOV
	cmovel	%ecx, %eax
#else
	jne	0f
	movl	%ecx, %eax
0:
#endif
	ret
END (__pthread_spin_trylock)
versioned_symbol (libc, __pthread_spin_trylock, pthread_spin_trylock,
		  GLIBC_2_34)

#if OTHER_SHLIB_COMPAT (libpthread, GLIBC_2_2, GLIBC_2_34)
compat_symbol (libpthread, __pthread_spin_trylock, pthread_spin_trylock,
	       GLIBC_2_2)
#endif
