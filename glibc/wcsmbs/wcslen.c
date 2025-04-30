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

/* Return length of string S.  */
#ifdef WCSLEN
# define __wcslen WCSLEN
#endif

size_t
__wcslen (const wchar_t *s)
{
  size_t len = 0;

  while (s[len] != L'\0')
    {
      if (s[++len] == L'\0')
	return len;
      if (s[++len] == L'\0')
	return len;
      if (s[++len] == L'\0')
	return len;
      ++len;
    }

  return len;
}
#ifndef WCSLEN
weak_alias (__wcslen, wcslen)
#endif
