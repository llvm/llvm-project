/* Measure memrchr functions.
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

#define TEST_MAIN
#define TEST_NAME "memrchr"
#include "bench-string.h"

typedef char *(*proto_t) (const char *, int, size_t);
char *simple_memrchr (const char *, int, size_t);

IMPL (simple_memrchr, 0)
IMPL (memrchr, 1)

char *
simple_memrchr (const char *s, int c, size_t n)
{
  s = s + n;
  while (n--)
    if (*--s == (char) c)
      return (char *) s;
  return NULL;
}

#define USE_AS_MEMRCHR
#include "bench-memchr.c"
