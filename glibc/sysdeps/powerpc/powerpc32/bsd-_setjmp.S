/* BSD `_setjmp' entry point to `sigsetjmp (..., 0)'.  PowerPC32/64 version.
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

#if !IS_IN (libc)
/* Build a non-versioned object for rtld-*.  */
ENTRY (_setjmp)
	li r4,0			/* Set second argument to 0.  */
	b __sigsetjmp@local
END (_setjmp)
libc_hidden_def (_setjmp)
#else
/* Build a versioned object for libc.  */

# if defined SHARED && SHLIB_COMPAT (libc, GLIBC_2_0, GLIBC_2_3_4)
compat_symbol (libc, __novmx_setjmp, _setjmp, GLIBC_2_0);

ENTRY (__novmx_setjmp)
	li r4,0			/* Set second argument to 0.  */
	b __novmx__sigsetjmp@local
END (__novmx_setjmp)
libc_hidden_def (__novmx_setjmp)
# endif /* defined SHARED  && SHLIB_COMPAT (libc, GLIBC_2_0, GLIBC_2_3_4) */

versioned_symbol (libc, __vmx_setjmp, _setjmp, GLIBC_2_3_4)
/* __GI__setjmp prototype is needed for ntpl i.e. _setjmp is defined
   as a libc_hidden_proto & is used in sysdeps/generic/libc-start.c
   if HAVE_CLEANUP_JMP_BUF is defined */
ENTRY (__GI__setjmp)
	li r4,0			/* Set second argument to 0.  */
	b __vmx__sigsetjmp@local
END (__GI__setjmp)

ENTRY (__vmx_setjmp)
	li r4,0			/* Set second argument to 0.  */
	b __vmx__sigsetjmp@local
END (__vmx_setjmp)
libc_hidden_def (__vmx_setjmp)
#endif /* IS_IN (libc) */
