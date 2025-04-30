/* Convert multibyte to char32_t.
   Copyright (C) 2018-2021 Free Software Foundation, Inc.
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
   License along with the GNU C Library; if not, see
   <https://www.gnu.org/licenses/>.  */

#include <uchar.h>
#include <wchar.h>


/* This is the private state used if PS is NULL.  */
static mbstate_t state;

size_t
mbrtoc32 (char32_t *pc32, const char *s, size_t n, mbstate_t *ps)
{
  if (ps == NULL)
    ps = &state;
  return mbrtowc ((wchar_t *) pc32, s, n, ps);
}
