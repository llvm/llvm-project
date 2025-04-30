/* Test strtol functions work with all ASCII letters in Turkish
   locales (bug 19242).
   Copyright (C) 2015-2021 Free Software Foundation, Inc.
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
#include <stdio.h>
#include <stdlib.h>
#include <wchar.h>

#define STR_(X) #X
#define STR(X) STR_(X)
#define FNPFXS STR (FNPFX)
#define CONCAT_(X, Y) X ## Y
#define CONCAT(X, Y) CONCAT_ (X, Y)
#define FNX(FN) CONCAT (FNPFX, FN)

#define TEST(LOC, STR, EXP_VAL, FN, TYPE, FMT)				\
  do									\
    {									\
      CHAR *ep;								\
      TYPE val = FNX (FN) (STR, &ep, 36);				\
      printf ("%s: " FNPFXS #FN " (" SFMT ") == " FMT "\n", LOC, STR, val); \
      if (val == (TYPE) (EXP_VAL) && *ep == 0)				\
	printf ("PASS: %s: " FNPFXS #FN " (" SFMT ")\n", LOC, STR);	\
      else								\
	{								\
	  printf ("FAIL: %s: " FNPFXS #FN " (" SFMT ")\n", LOC, STR);	\
	  result = 1;							\
	}								\
    }									\
  while (0)

static int
test_one_locale (const char *loc)
{
  if (setlocale (LC_ALL, loc) == NULL)
    {
      printf ("setlocale (LC_ALL, \"%s\") failed\n", loc);
      return 1;
    }
  int result = 0;
  for (int i = 10; i < 36; i++)
    {
      CHAR s[2];
      s[0] = L_('A') + i - 10;
      s[1] = 0;
      TEST (loc, s, i, l, long int, "%ld");
      TEST (loc, s, i, ul, unsigned long int, "%lu");
      TEST (loc, s, i, ll, long long int, "%lld");
      TEST (loc, s, i, ull, unsigned long long int, "%llu");
      s[0] = L_('a') + i - 10;
      s[1] = 0;
      TEST (loc, s, i, l, long int, "%ld");
      TEST (loc, s, i, ul, unsigned long int, "%lu");
      TEST (loc, s, i, ll, long long int, "%lld");
      TEST (loc, s, i, ull, unsigned long long int, "%llu");
    }
  return result;
}

static int
do_test (void)
{
  int result = 0;
  result |= test_one_locale ("C");
  result |= test_one_locale ("tr_TR.UTF-8");
  result |= test_one_locale ("tr_TR.ISO-8859-9");
  return result;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
