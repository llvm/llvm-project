/* rawmemchr - find a character in a memory zone

   Copyright (C) 2015-2021 Free Software Foundation, Inc.

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

/* Special case rawmemchr (s, 0) as strlen, otherwise tailcall memchr.
   Call strlen without setting up a full frame - it preserves x14/x15.
*/

ENTRY (__rawmemchr)
	cbz	w1, L(do_strlen)
	mov	x2, -1
	b	__memchr

L(do_strlen):
	mov	x15, x30
	cfi_return_column (x15)
	mov	x14, x0
	bl	__strlen
	add	x0, x14, x0
	ret	x15

END (__rawmemchr)
weak_alias (__rawmemchr, rawmemchr)
libc_hidden_builtin_def (__rawmemchr)
