/* Copyright (C) 1996-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@gnu.org>, 1996.

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

#ifdef WMEMSET
# define __wmemset WMEMSET
#endif

wchar_t *
__wmemset (wchar_t *s, wchar_t c, size_t n)
{
  wchar_t *wp = s;

  while (n >= 4)
    {
      wp[0] = c;
      wp[1] = c;
      wp[2] = c;
      wp[3] = c;
      wp += 4;
      n -= 4;
    }

  if (n > 0)
    {
      wp[0] = c;

      if (n > 1)
	{
	  wp[1] = c;

	  if (n > 2)
	    wp[2] = c;
	}
    }

  return s;
}
libc_hidden_def (__wmemset)
weak_alias (__wmemset, wmemset)
libc_hidden_weak (wmemset)
