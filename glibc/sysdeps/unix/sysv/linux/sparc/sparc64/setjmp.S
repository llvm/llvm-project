/* Copyright (C) 1997-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Richard Henderson (rth@tamu.edu).

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

/* __sigsetjmp is implemented in terms of the getcontext trap on
   Linux/Sparc64.  */

#include <sysdep.h>

/* Offsets into the jmp_buf structure.  */

#define O_mask_was_saved	512
#define O_gregs			32
#define O_g1			(O_gregs + 4*8)

/* int _setjmp(jmp_buf) */

ENTRY(_setjmp)
	ba	__sigsetjmp_local
	 set	0, %o1
END(_setjmp)
libc_hidden_def (_setjmp)

/* int setjmp(jmp_buf) */

ENTRY(setjmp)
	ba,pt	%xcc, __sigsetjmp_local
	 set	1, %o1
END(setjmp)

/* int __sigsetjmp(jmp_buf, savemask)  */

ENTRY(__sigsetjmp)
__sigsetjmp_local:

	/* Record whether the user is intending to save the sigmask.  */
	st	%o1, [%o0 + O_mask_was_saved]

	/* Load up our return value, as longjmp is going to override
	   the jmp_buf on its way back.  */
	mov	%g0, %g1

	/* And call getcontext!  */
	ta	0x6e

	retl
	 mov	%g1, %o0

END(__sigsetjmp)
hidden_def (__sigsetjmp)

weak_extern(_setjmp)
weak_extern(setjmp)
