/* Measure stpncpy functions.
   Copyright (C) 2013-2021 Free Software Foundation, Inc.
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

#define STRNCPY_RESULT(dst, len, n) ((dst) + ((len) > (n) ? (n) : (len)))
#define TEST_MAIN
#ifndef WIDE
# define TEST_NAME "stpncpy"
#else
# define TEST_NAME "wcpncpy"
# define generic_stpncpy generic_wcpncpy
#endif /* WIDE */
#include "bench-string.h"

CHAR *
generic_stpncpy (CHAR *dst, const CHAR *src, size_t n)
{
  size_t nc = STRNLEN (src, n);
  MEMCPY (dst, src, nc);
  dst += nc;
  if (nc == n)
    return dst;
  return MEMSET (dst, 0, n - nc);
}

IMPL (STPNCPY, 1)
IMPL (generic_stpncpy, 0)

#include "bench-strncpy.c"
