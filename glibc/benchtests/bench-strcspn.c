/* Measure strcspn functions.
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

#define STRPBRK_RESULT(s, pos) (pos)
#define RES_TYPE size_t
#define TEST_MAIN
#ifndef WIDE
# define TEST_NAME "strcspn"
#else
# define TEST_NAME "wcscspn"
#endif /* WIDE */
#include "bench-string.h"

#ifndef WIDE
# define SIMPLE_STRCSPN simple_strcspn
#else
# define SIMPLE_STRCSPN simple_wcscspn
#endif /* WIDE */

typedef size_t (*proto_t) (const CHAR *, const CHAR *);
size_t SIMPLE_STRCSPN (const CHAR *, const CHAR *);

IMPL (SIMPLE_STRCSPN, 0)
IMPL (STRCSPN, 1)

size_t
SIMPLE_STRCSPN (const CHAR *s, const CHAR *rej)
{
  const CHAR *r, *str = s;
  CHAR c;

  while ((c = *s++) != '\0')
    for (r = rej; *r != '\0'; ++r)
      if (*r == c)
	return s - str - 1;
  return s - str - 1;
}

#include "bench-strpbrk.c"
