/* Measure stpcpy functions.
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

#define STRCPY_RESULT(dst, len) ((dst) + (len))
#define TEST_MAIN
#ifndef WIDE
# define TEST_NAME "stpcpy"
#else
# define TEST_NAME "wcpcpy"
# define generic_stpcpy generic_wcpcpy
#endif /* WIDE */
#include "bench-string.h"

CHAR *
generic_stpcpy (CHAR *dst, const CHAR *src)
{
  size_t len = STRLEN (src);
  return (CHAR *) MEMCPY (dst, src, len + 1) + len;
}

IMPL (STPCPY, 1)
IMPL (generic_stpcpy, 0)

#include "bench-strcpy.c"
