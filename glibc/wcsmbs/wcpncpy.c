/* Copyright (C) 1996-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@gnu.ai.mit.edu>, 1995.

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

#ifdef WCPNCPY
# define __wcpncpy WCPNCPY
#endif

/* Copy no more than N wide-characters of SRC to DEST, returning the
   address of the last character written into DEST.  */
wchar_t *
__wcpncpy (wchar_t *dest, const wchar_t *src, size_t n)
{
  size_t size = __wcsnlen (src, n);
  __wmemcpy (dest, src, size);
  dest += size;
  if (size == n)
    return dest;
  return wmemset (dest, L'\0', (n - size));
}

#ifndef WCPNCPY
weak_alias (__wcpncpy, wcpncpy)
#endif
