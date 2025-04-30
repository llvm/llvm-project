/* Test strcspn functions.
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

#define STRPBRK_RESULT(s, pos) (pos)
#define RES_TYPE size_t
#define TEST_MAIN
#ifndef WIDE
# define TEST_NAME "strcspn"
#else
# define TEST_NAME "wcscspn"
#endif /* WIDE */
#include "test-string.h"

#ifndef WIDE
# define STRCSPN strcspn
# define CHAR char
# define SIMPLE_STRCSPN simple_strcspn
# define STUPID_STRCSPN stupid_strcspn
# define STRLEN strlen
#else
# include <wchar.h>
# define STRCSPN wcscspn
# define CHAR wchar_t
# define SIMPLE_STRCSPN simple_wcscspn
# define STUPID_STRCSPN stupid_wcscspn
# define STRLEN wcslen
#endif /* WIDE */

typedef size_t (*proto_t) (const CHAR *, const CHAR *);
size_t SIMPLE_STRCSPN (const CHAR *, const CHAR *);
size_t STUPID_STRCSPN (const CHAR *, const CHAR *);

IMPL (STUPID_STRCSPN, 0)
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

size_t
STUPID_STRCSPN (const CHAR *s, const CHAR *rej)
{
  size_t ns = STRLEN (s), nrej = STRLEN (rej);
  size_t i, j;

  for (i = 0; i < ns; ++i)
    for (j = 0; j < nrej; ++j)
      if (s[i] == rej[j])
	return i;
  return i;
}

#undef CHAR
#undef STRLEN
#include "test-strpbrk.c"
