/* Copyright (C) 1991-2021 Free Software Foundation, Inc.
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
#include <sys/trap.h>

#include <jmpbuf-offsets.h>

ENTRY(_setjmp)
	b	1f
	 set	0, %o1
END(_setjmp)
libc_hidden_def (_setjmp)

ENTRY(setjmp)
	set	1, %o1
END(setjmp)

ENTRY (__sigsetjmp)
1:
	/* Save our PC, SP and FP.  Save the signal mask if requested with
	   a tail-call for simplicity; it always returns zero.  */
	ta	ST_FLUSH_WINDOWS

#ifdef PTR_MANGLE
	PTR_MANGLE (%g1, %o7, %g4)
	PTR_MANGLE2 (%g2, %sp, %g4)
	PTR_MANGLE2 (%g3, %fp, %g4)
	st	%g1, [%o0 + (JB_PC * 4)]
	st	%g2, [%o0 + (JB_SP * 4)]
	st	%g3, [%o0 + (JB_FP * 4)]
#else
	st	%o7, [%o0 + (JB_PC * 4)]
	st	%sp, [%o0 + (JB_SP * 4)]
	st	%fp, [%o0 + (JB_FP * 4)]
#endif

	mov	%o7, %g1
	call	__sigjmp_save
	 mov	%g1, %o7
END(__sigsetjmp)
hidden_def (__sigsetjmp)

weak_extern(_setjmp)
weak_extern(setjmp)
