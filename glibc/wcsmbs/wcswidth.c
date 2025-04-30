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

#include "wcwidth.h"

/* Determine number of column positions required for first N wide
   characters (or fewer if S ends before this) in S.  */
int
wcswidth (const wchar_t *s, size_t n)
{
  int result = 0;

  while (n-- > 0 && *s != L'\0')
    {
      int now = internal_wcwidth (*s);
      if (now == -1)
	return -1;
      result += now;
      ++s;
    }

  return result;
}
