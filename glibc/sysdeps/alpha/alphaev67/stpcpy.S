/* Copyright (C) 2000-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Richard Henderson <rth@redhat.com>.

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

/*  Copy SRC to DEST returning the address of the terminating 0 in DEST.  */

#include <sysdep.h>

	.arch ev6
	.set noreorder
	.set noat
	.text

ENTRY(__stpcpy)
	ldgp	gp, 0(pv)
#ifdef PROF
	lda	AT, _mcount
	jsr	AT, (AT), _mcount
#endif
	.prologue 1

	.align 4
	mov	a0, v0
	nop
	jsr	t9, __stxcpy

        # t8  = bitmask (with one bit set) indicating the last byte written
        # a0  = unaligned address of the last *word* written

	cttz	t8, t8
	andnot	a0, 7, a0
        addq    a0, t8, v0
	ret

	END(__stpcpy)

weak_alias (__stpcpy, stpcpy)
libc_hidden_def (__stpcpy)
libc_hidden_builtin_def (stpcpy)
