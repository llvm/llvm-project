/* ffs -- find first set bit in an int, from least significant end.
   Copyright (C) 2013-2021 Free Software Foundation, Inc.
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
   License along with the GNU C Library.  If not, see
   <https://www.gnu.org/licenses/>.  */

#include <sysdep.h>

	.syntax unified
	.text

ENTRY (__ffs)
	cmp	r0, #0
	rbit	r0, r0
	itt	ne
	clzne	r0, r0
	addne	r0, r0, #1
	bx	lr
END (__ffs)

weak_alias (__ffs, ffs)
weak_alias (__ffs, ffsl)
libc_hidden_def (__ffs)
libc_hidden_builtin_def (ffs)
