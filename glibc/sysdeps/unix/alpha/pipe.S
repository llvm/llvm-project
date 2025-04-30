/* Copyright (C) 1993-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by David Mosberger (davidm@cs.arizona.edu).

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

/* __pipe is a special syscall since it returns two values.  */

#include <sysdep.h>

PSEUDO (__pipe, pipe, 0)
	stl	r0, 0(a0)
	stl	r1, 4(a0)
	mov	zero, v0
	ret
PSEUDO_END(__pipe)

libc_hidden_def (__pipe)
weak_alias (__pipe, pipe)
