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

#ifdef WCSSPN
# define wcsspn WCSSPN
#endif

/* Return the length of the maximum initial segment
   of WCS which contains only wide-characters in ACCEPT.  */
size_t
wcsspn (const wchar_t *wcs, const wchar_t *accept)
{
  const wchar_t *p;
  const wchar_t *a;
  size_t count = 0;

  for (p = wcs; *p != L'\0'; ++p)
    {
      for (a = accept; *a != L'\0'; ++a)
	if (*p == *a)
	  break;
      if (*a == L'\0')
	return count;
      else
	++count;
    }

  return count;
}
libc_hidden_def (wcsspn)
