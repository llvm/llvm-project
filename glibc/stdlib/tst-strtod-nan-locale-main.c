/* Test strtod functions work with all ASCII letters in NAN(...) in
   Turkish locales (bug 19266).
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
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <wchar.h>

#include <stdlib/tst-strtod.h>

#define STR_(X) #X
#define STR(X) STR_(X)
#define FNPFXS STR (FNPFX)
#define CONCAT_(X, Y) X ## Y
#define CONCAT(X, Y) CONCAT_ (X, Y)
#define FNX(FN) CONCAT (FNPFX, FN)

#define TEST_STRTOD(FSUF, FTYPE, FTOSTR, LSUF, CSUF)			\
static int								\
test_strto ## FSUF (const char * loc, CHAR * s)				\
{									\
  CHAR *ep;								\
  FTYPE val = FNX (FSUF) (s, &ep);					\
  if (isnan (val) && *ep == 0)						\
    printf ("PASS: %s: " FNPFXS #FSUF " (" SFMT ")\n", loc, s);		\
  else									\
    {									\
      printf ("FAIL: %s: " FNPFXS #FSUF " (" SFMT ")\n", loc, s);	\
      return 1;							        \
    }									\
  return 0;								\
}
GEN_TEST_STRTOD_FOREACH (TEST_STRTOD)

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
      CHAR s[7];
      s[0] = L_('N');
      s[1] = L_('A');
      s[2] = L_('N');
      s[3] = L_('(');
      s[4] = L_('A') + i - 10;
      s[5] = L_(')');
      s[6] = 0;
      result |= STRTOD_TEST_FOREACH (test_strto, loc, s);
      s[4] = L_('a') + i - 10;
      result |= STRTOD_TEST_FOREACH (test_strto, loc, s);
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
