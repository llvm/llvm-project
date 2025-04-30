/* Test __p_secstodate compat symbol.
   Copyright (C) 2017-2021 Free Software Foundation, Inc.
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

#include <array_length.h>
#include <limits.h>
#include <resolv.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>

#include <shlib-compat.h>

char *__p_secstodate (unsigned long int);
compat_symbol_reference (libresolv, __p_secstodate, __p_secstodate, GLIBC_2_0);

struct test
{
  /* Argument to __p_secstodate.  */
  unsigned long int in;
  /* Expected output.  */
  const char *out;
};

static const struct test tests[] =
  {
    { 0UL, "19700101000000" },
    { 12345UL, "19700101032545" },
    { 999999999UL, "20010909014639" },
    { 2147483647UL, "20380119031407" },
    { 2147483648UL, "<overflow>" },
    { 4294967295UL, "<overflow>" },
# if ULONG_MAX > 0xffffffffUL
    { 4294967296UL, "<overflow>" },
    { 9999999999UL, "<overflow>" },
    { LONG_MAX, "<overflow>" },
    { ULONG_MAX, "<overflow>" },
# endif
  };

static int
do_test (void)
{
  int ret = 0;
  for (size_t i = 0; i < array_length (tests); i++)
    {
      char *p = __p_secstodate (tests[i].in);
      printf ("Test %zu: %lu -> %s\n", i, tests[i].in, p);
      if (strcmp (p, tests[i].out) != 0)
	{
	  printf ("test %zu failed", i);
	  ret = 1;
	}
    }
  return ret;
}

#include <support/test-driver.c>
