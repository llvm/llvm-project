/* Copyright (C) 2003-2021 Free Software Foundation, Inc.
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

#include <shlib-compat.h>

	.globl	__pthread_spin_unlock
	.type	__pthread_spin_unlock,@function
	.align	5
__pthread_spin_unlock:
	mov	#0,r0
	rts
	 mov.l	r0,@r4
	.size	__pthread_spin_unlock,.-__pthread_spin_unlock
versioned_symbol (libc, __pthread_spin_unlock, pthread_spin_unlock, GLIBC_2_34)
/* The implementation of pthread_spin_init is identical.  */
versioned_symbol (libc, __pthread_spin_unlock, pthread_spin_init, GLIBC_2_34)

#if OTHER_SHLIB_COMPAT (libpthread, GLIBC_2_2, GLIBC_2_34)
compat_symbol (libpthread, __pthread_spin_unlock, pthread_spin_unlock,
	       GLIBC_2_2)
compat_symbol (libpthread, __pthread_spin_unlock, pthread_spin_init, GLIBC_2_2)
#endif
