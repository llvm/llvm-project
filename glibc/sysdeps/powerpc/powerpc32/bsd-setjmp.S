/* BSD `setjmp' entry point to `sigsetjmp (..., 1)'.  PowerPC32/64 version.
   Copyright (C) 1994-2021 Free Software Foundation, Inc.
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
#include <libc-symbols.h>
#include <sysdep.h>

#if defined SHARED && SHLIB_COMPAT (libc, GLIBC_2_0, GLIBC_2_3_4)

ENTRY (__novmxsetjmp)
	li r4,1			/* Set second argument to 1.  */
	b __novmx__sigsetjmp@local
END (__novmxsetjmp)
strong_alias (__novmxsetjmp, __novmx__setjmp)
compat_symbol (libc, __novmxsetjmp, setjmp, GLIBC_2_0)

#endif  /* defined SHARED && SHLIB_COMPAT (libc, GLIBC_2_0, GLIBC_2_3_4) ) */

ENTRY (__vmxsetjmp)
	li r4,1			/* Set second argument to 1.  */
	b __vmx__sigsetjmp@local
END (__vmxsetjmp)
strong_alias (__vmxsetjmp, __vmx__setjmp)
strong_alias (__vmx__setjmp, __setjmp)
versioned_symbol (libc, __vmxsetjmp, setjmp, GLIBC_2_3_4)
