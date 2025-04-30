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

/* longjmp is implemented in terms of the setcontext trap on Linux/Sparc64.  */

#include <sysdep.h>

/* Offsets into the jmp_buf structure.  */

#define O_mask_was_saved	512
#define O_gregs			32
#define O_g1			(O_gregs + 4*8)


/* void longjmp (sigjmp_buf env, int val) */

ENTRY(__libc_siglongjmp)

	/* Modify the context with the value we want to return.  */
	movre	%o1, 1, %o1
	stx	%o1, [%o0 + O_g1]

	/* Let setcontext know if we want to modify the current sigmask. */
	ld	[%o0 + O_mask_was_saved], %o1

	/* And bamf back to where we belong!  */
	ta	0x6f

END(__libc_siglongjmp)

strong_alias(__libc_siglongjmp, __longjmp)
strong_alias(__libc_siglongjmp, __libc_longjmp)
weak_alias (__libc_siglongjmp, longjmp)
weak_alias (__libc_siglongjmp, _longjmp)
weak_alias (__libc_siglongjmp, siglongjmp)
