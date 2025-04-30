/* Copyright (C) 1995-2021 Free Software Foundation, Inc.
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

#ifndef WCSCMP
# define WCSCMP __wcscmp
#endif

/* Compare S1 and S2, returning less than, equal to or
   greater than zero if S1 is lexicographically less than,
   equal to or greater than S2.	 */
int
WCSCMP (const wchar_t *s1, const wchar_t *s2)
{
  wchar_t c1, c2;

  do
    {
      c1 = *s1++;
      c2 = *s2++;
      if (c2 == L'\0')
	return c1 - c2;
    }
  while (c1 == c2);

  return c1 < c2 ? -1 : 1;
}
libc_hidden_def (WCSCMP)
weak_alias (WCSCMP, wcscmp)
