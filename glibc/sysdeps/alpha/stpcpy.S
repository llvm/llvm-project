/* Copyright (C) 1996-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Richard Henderson <rth@tamu.edu>, 1996.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library.  If not, see
   <https://www.gnu.org/licenses/>.  */

/* Copy a null-terminated string from SRC to DST.  Return a pointer
   to the null-terminator in the source.  */

#include <sysdep.h>

	.text

ENTRY(__stpcpy)
	ldgp	gp, 0(pv)
#ifdef PROF
	.set noat
	lda	AT, _mcount
	jsr	AT, (AT), _mcount
	.set at
#endif
	.prologue 1

	jsr	t9, __stxcpy	# do the work of the copy

	and	t8, 0xf0, t2	# binary search for byte offset of the
	and	t8, 0xcc, t1	# last byte written.
	and	t8, 0xaa, t0
	andnot	a0, 7, a0
	cmovne	t2, 4, t2
	cmovne	t1, 2, t1
	cmovne	t0, 1, t0
	addq	a0, t2, v0
	addq	t0, t1, t0
	addq	v0, t0, v0

	ret

	END(__stpcpy)

weak_alias (__stpcpy, stpcpy)
libc_hidden_def (__stpcpy)
libc_hidden_builtin_def (stpcpy)
