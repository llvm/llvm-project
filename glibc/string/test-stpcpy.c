/* Test stpcpy functions.
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
#ifndef WIDE
# define TEST_NAME "stpcpy"
#else
# define TEST_NAME "wcpcpy"
#endif /* !WIDE */
#include "test-string.h"
#ifndef WIDE
# define CHAR char
# define SIMPLE_STPCPY simple_stpcpy
# define STPCPY stpcpy
#else
# include <wchar.h>
# define CHAR wchar_t
# define SIMPLE_STPCPY simple_wcpcpy
# define STPCPY wcpcpy
#endif /* !WIDE */

CHAR *SIMPLE_STPCPY (CHAR *, const CHAR *);

IMPL (SIMPLE_STPCPY, 0)
IMPL (STPCPY, 1)

CHAR *
SIMPLE_STPCPY (CHAR *dst, const CHAR *src)
{
  while ((*dst++ = *src++) != '\0');
  return dst - 1;
}

#undef CHAR
#include "test-strcpy.c"
