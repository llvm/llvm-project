/* Test and measure stpcpy checking functions.
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

#define STRCPY_RESULT(dst, len) ((dst) + (len))
#define TEST_MAIN
#define TEST_NAME "stpcpy_chk"
#include "../string/test-string.h"

extern void __attribute__ ((noreturn)) __chk_fail (void);
char *simple_stpcpy_chk (char *, const char *, size_t);
extern char *normal_stpcpy (char *, const char *, size_t)
  __asm ("stpcpy");
extern char *__stpcpy_chk (char *, const char *, size_t);

IMPL (simple_stpcpy_chk, 0)
IMPL (normal_stpcpy, 1)
IMPL (__stpcpy_chk, 2)

char *
simple_stpcpy_chk (char *dst, const char *src, size_t len)
{
  if (! len)
    __chk_fail ();
  while ((*dst++ = *src++) != '\0')
    if (--len == 0)
      __chk_fail ();
  return dst - 1;
}

#include "test-strcpy_chk.c"
