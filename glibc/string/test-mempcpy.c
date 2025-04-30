/* Test and measure mempcpy functions.
   Copyright (C) 1999-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Written by Jakub Jelinek <jakub@redhat.com>, 1999.

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

#define MEMCPY_RESULT(dst, len) (dst) + (len)
#define MIN_PAGE_SIZE 131072
#define TEST_MAIN
#define TEST_NAME "mempcpy"
#include "test-string.h"

char *simple_mempcpy (char *, const char *, size_t);

IMPL (simple_mempcpy, 0)
IMPL (mempcpy, 1)

char *
simple_mempcpy (char *dst, const char *src, size_t n)
{
  while (n--)
    *dst++ = *src++;
  return dst;
}

#include "test-memcpy.c"
