/* Copyright (C) 1996-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@gnu.ai.mit.edu>, 1996.

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

#include <wchar.h>

#define __need_ptrdiff_t
#include <stddef.h>


/* Copy SRC to DEST, returning the address of the terminating L'\0' in
   DEST.  Check for overflows.  */
wchar_t *
__wcpcpy_chk (wchar_t *dest, const wchar_t *src, size_t destlen)
{
  wchar_t *wcp = (wchar_t *) dest - 1;
  wint_t c;
  const ptrdiff_t off = src - dest + 1;

  do
    {
      if (__glibc_unlikely (destlen-- == 0))
	__chk_fail ();
      c = wcp[off];
      *++wcp = c;
    }
  while (c != L'\0');

  return wcp;
}
