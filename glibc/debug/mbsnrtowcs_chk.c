/* Copyright (C) 2005-2021 Free Software Foundation, Inc.
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

#include <locale.h>
#include <wchar.h>


size_t
__mbsnrtowcs_chk (wchar_t *dst, const char **src, size_t nmc, size_t len,
		  mbstate_t *ps, size_t dstlen)
{
  if (__glibc_unlikely (dstlen < len))
    __chk_fail ();

  return __mbsnrtowcs (dst, src, nmc, len, ps);
}
