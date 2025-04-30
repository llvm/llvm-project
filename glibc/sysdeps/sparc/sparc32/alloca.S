/* Copyright (C) 1994-2021 Free Software Foundation, Inc.
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

/* Code produced by Sun's C compiler calls this function with two extra
   arguments which it makes relocatable symbols but seem always to be
   the constant 96; I have no idea what they are for.  */

ENTRY (__builtin_alloca)
	sub %sp, %o0, %sp	/* Push some stack space.  */
	retl			/* Return; the returned buffer leaves 96 */
	add %sp, 96, %o0	/* bytes of register save area at the top. */
END (__builtin_alloca)
