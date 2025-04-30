/* Copyright (C) 2012-2021 Free Software Foundation, Inc.
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
#include <shlib-compat.h>

	.text
ENTRY(__pthread_spin_lock)
1:	ldstub		[%o0], %g1
	brnz,pn		%g1, 2f
	 membar		#StoreLoad | #StoreStore
	retl
	 mov		0, %o0
2:	ldub		[%o0], %g1
	brnz,pt		%g1, 2b
	 membar		#LoadLoad
	ba,a,pt		%xcc, 1b
END(__pthread_spin_lock)
versioned_symbol (libc, __pthread_spin_lock, pthread_spin_lock, GLIBC_2_34)

#if OTHER_SHLIB_COMPAT (libpthread, GLIBC_2_2, GLIBC_2_34)
compat_symbol (libpthread, __pthread_spin_lock, pthread_spin_lock, GLIBC_2_2)
#endif
