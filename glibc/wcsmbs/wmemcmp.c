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

#ifdef WMEMCMP
# define __wmemcmp WMEMCMP
#endif

int
__wmemcmp (const wchar_t *s1, const wchar_t *s2, size_t n)
{
  wchar_t c1;
  wchar_t c2;

  while (n >= 4)
    {
      c1 = s1[0];
      c2 = s2[0];
      if (c1 - c2 != 0)
	return c1 > c2 ? 1 : -1;
      c1 = s1[1];
      c2 = s2[1];
      if (c1 - c2 != 0)
	return c1 > c2 ? 1 : -1;
      c1 = s1[2];
      c2 = s2[2];
      if (c1 - c2 != 0)
	return c1 > c2 ? 1 : -1;
      c1 = s1[3];
      c2 = s2[3];
      if (c1 - c2 != 0)
	return c1 > c2 ? 1 : -1;
      s1 += 4;
      s2 += 4;
      n -= 4;
    }

  if (n > 0)
    {
      c1 = s1[0];
      c2 = s2[0];
      if (c1 - c2 != 0)
	return c1 > c2 ? 1 : -1;
      ++s1;
      ++s2;
      --n;
    }
  if (n > 0)
    {
      c1 = s1[0];
      c2 = s2[0];
      if (c1 - c2 != 0)
	return c1 > c2 ? 1 : -1;
      ++s1;
      ++s2;
      --n;
    }
  if (n > 0)
    {
      c1 = s1[0];
      c2 = s2[0];
      if (c1 - c2 != 0)
	return c1 > c2 ? 1 : -1;
    }

  return 0;
}
#ifndef WMEMCMP
weak_alias (__wmemcmp, wmemcmp)
#endif
