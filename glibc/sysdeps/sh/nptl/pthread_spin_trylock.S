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

#include <errno.h>
#include <shlib-compat.h>

	.globl	__pthread_spin_trylock
	.type	__pthread_spin_trylock,@function
	.align	5
__pthread_spin_trylock:
	tas.b	@r4
	bf/s	1f
	mov	#EBUSY, r0
	mov	#0, r0
1:
	rts
	 nop
	.size	__pthread_spin_trylock,.-__pthread_spin_trylock
versioned_symbol (libc, __pthread_spin_trylock, pthread_spin_trylock,
		  GLIBC_2_34)

#if OTHER_SHLIB_COMPAT (libpthread, GLIBC_2_2, GLIBC_2_34)
compat_symbol (libpthread, __pthread_spin_trylock, pthread_spin_trylock,
	       GLIBC_2_2)
#endif
