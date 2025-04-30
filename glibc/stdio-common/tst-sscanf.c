/* Copyright (C) 2000-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Jakub Jelinek <jakub@redhat.com>, 2000.

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
#include <stdlib.h>
#include <stdio.h>
#include <locale.h>

#ifndef CHAR
# define CHAR char
# define L(str) str
# define SSCANF sscanf
#endif

const CHAR *str_double[] =
{
  L("-.10000E+020.20000E+020.25000E+010.40000E+010.50000E+010.12500E+01"),
  L("0.10000E+020.20000E+020.25000E+010.40000E+010.50000E+010.12500E+01"),
  L("-1234567E0198765432E0912345678901987654321091234567890198765432109"),
  L("-0.1000E+020.20000E+020.25000E+010.40000E+010.50000E+010.12500E+01")
};

const double val_double[] =
{
  -.10000E+02, 0.20000E+02, 0.25000E+01, 0.40000E+01, 0.50000E+01, 0.12500E+01,
  0.10000E+02, 0.20000E+02, 0.25000E+01, 0.40000E+01, 0.50000E+01, 0.12500E+01,
  -1234567E01, 98765432E09, 12345678901.0, 98765432109.0, 12345678901.0,
    98765432109.0,
  -0.1000E+02, 0.20000E+02, 0.25000E+01, 0.40000E+01, 0.50000E+01, 0.12500E+01
};

const CHAR *str_long[] =
{
  L("-12345678987654321123456789987654321123456789987654321"),
  L("-12345678987654321123456789987654321123456789987654321"),
  L("-12,345,678987,654,321123,456,789987,654,321123,456,789987,654,321"),
  L("-12,345,678987,654,321123,456,789987,654,321123,456,789987,654,321")
};

const CHAR *fmt_long[] =
{
  L("%9ld%9ld%9ld%9ld%9ld%9ld"),
  L("%I9ld%I9ld%I9ld%I9ld%I9ld%I9ld"),
  L("%'11ld%'11ld%'11ld%'11ld%'11ld%'11ld"),
  L("%I'11ld%I'11ld%I'11ld%I'11ld%I'11ld%I'11ld")
};

const long int val_long[] =
{
  -12345678, 987654321, 123456789, 987654321, 123456789, 987654321
};

struct test
{
  const CHAR *str;
  const CHAR *fmt;
  int retval;
} int_tests[] =
{
  { L("foo\n"), L("foo\nbar"), -1 },
  { L("foo\n"), L("foo bar"), -1 },
  { L("foo\n"), L("foo %d"), -1 },
  { L("foo\n"), L("foo\n%d"), -1 },
  { L("foon"), L("foonbar"), -1 },
  { L("foon"), L("foon%d"), -1 },
  { L("foo "), L("foo bar"), -1 },
  { L("foo "), L("foo %d"), -1 },
  { L("foo\t"), L("foo\tbar"), -1 },
  { L("foo\t"), L("foo bar"), -1 },
  { L("foo\t"), L("foo %d"), -1 },
  { L("foo\t"), L("foo\t%d"), -1 },
  { L("foo"), L("foo"), 0 },
  { L("foon"), L("foo bar"), 0 },
  { L("foon"), L("foo %d"), 0 },
  { L("foo "), L("fooxbar"), 0 },
  { L("foo "), L("foox%d"), 0 },
  { L("foo bar"), L("foon"), 0 },
  { L("foo bar"), L("foo bar"), 0 },
  { L("foo bar"), L("foo %d"), 0 },
  { L("foo bar"), L("foon%d"), 0 },
  { L("foo (nil)"), L("foo %p"), 1},
  { L("foo (nil)"), L("foo %4p"), 0},
  { L("foo "), L("foo %n"), 0 },
  { L("foo%bar1"), L("foo%%bar%d"), 1 },
  /* Some OSes skip whitespace here while others don't.  */
  { L("foo \t %bar1"), L("foo%%bar%d"), 1 }
};

struct test double_tests[] =
{
  { L("-1"), L("%1g"), 0 },
  { L("-.1"), L("%2g"), 0 },
  { L("-inf"), L("%3g"), 0 },
  { L("+0"), L("%1g"),  },
  { L("-0x1p0"), L("%2g"), 1 },
  { L("-..1"), L("%g"), 0 },
  { L("-inf"), L("%g"), 1 }
};

struct test2
{
  const CHAR *str;
  const CHAR *fmt;
  int retval;
  char residual;
} double_tests2[] =
{
  { L("0e+0"), L("%g%c"), 1, 0 },
  { L("0xe+0"), L("%g%c"), 2, '+' },
  { L("0x.e+0"), L("%g%c"), 2, '+' },
};

static int
do_test (void)
{
  double d[6];
  long l[6];
  int i, j;
  int tst_locale;
  int result = 0;

  tst_locale = 1;
  if (tst_locale)
    if (setlocale (LC_ALL, "en_US.ISO-8859-1") == NULL)
      {
	puts ("Failed to set en_US locale, skipping locale related tests");
	tst_locale = 0;
      }

  for (i = 0; i < 4; ++i)
    {
      if (SSCANF (str_double[i], L("%11lf%11lf%11lf%11lf%11lf%11lf"),
		  &d[0], &d[1], &d[2], &d[3], &d[4], &d[5]) != 6)
	{
	  printf ("Double sscanf test %d wrong number of "
		  "assigned inputs\n", i);
	  result = 1;
	}
      else
	for (j = 0; j < 6; ++j)
	  if (d[j] != val_double[6 * i + j])
	    {
	      printf ("Double sscanf test %d failed (%g instead of %g)\n",
		      i, d[j], val_double[6 * i + j]);
	      result = 1;
	      break;
	    }
    }

  for (i = 0; i < 4; ++i)
    {
      if (SSCANF (str_long[i], fmt_long[i],
		  &l[0], &l[1], &l[2], &l[3], &l[4], &l[5]) != 6)
	{
	  printf ("Integer sscanf test %d wrong number of "
		  "assigned inputs\n", i);
	  result = 1;
	}
      else
	for (j = 0; j < 6; ++j)
	  if (l[j] != val_long[j])
	    {
	      printf ("Integer sscanf test %d failed (%ld instead %ld)\n",
		      i, l[j], val_long[j]);
	      result = 1;
	      break;
	    }

      if (! tst_locale)
	break;
    }

  for (i = 0; i < array_length (int_tests); ++i)
    {
      long dummy;
      int ret;

      if ((ret = SSCANF (int_tests[i].str, int_tests[i].fmt,
			 &dummy)) != int_tests[i].retval)
	{
	  printf ("int_tests[%d] returned %d != %d\n",
		  i, ret, int_tests[i].retval);
	  result = 1;
	}
    }

  for (i = 0; i < array_length (double_tests); ++i)
    {
      double dummy;
      int ret;

      if ((ret = SSCANF (double_tests[i].str, double_tests[i].fmt,
			 &dummy)) != double_tests[i].retval)
	{
	  printf ("double_tests[%d] returned %d != %d\n",
		  i, ret, double_tests[i].retval);
	  result = 1;
	}
    }

  for (i = 0; i < array_length (double_tests2); ++i)
    {
      double dummy;
      int ret;
      char c = 0;

      if ((ret = SSCANF (double_tests2[i].str, double_tests2[i].fmt,
			 &dummy, &c)) != double_tests2[i].retval)
	{
	  printf ("double_tests2[%d] returned %d != %d\n",
		  i, ret, double_tests2[i].retval);
	  result = 1;
	}
      else if (ret == 2 && c != double_tests2[i].residual)
	{
	  printf ("double_tests2[%d] stopped at '%c' != '%c'\n",
		  i, c, double_tests2[i].residual);
	  result = 1;
	}
    }

  /* BZ #16618
     The test will segfault during SSCANF if the buffer overflow
     is not fixed.  The size of `s` is such that it forces the use
     of malloc internally and this triggers the incorrect computation.
     Thus the value for SIZE is arbitrariy high enough that malloc
     is used.  */
  {
#define SIZE 131072
    CHAR *s = malloc ((SIZE + 1) * sizeof (*s));
    if (s == NULL)
      abort ();
    for (size_t i = 0; i < SIZE; i++)
      s[i] = L('0');
    s[SIZE] = L('\0');
    int i = 42;
    /* Scan multi-digit zero into `i`.  */
    if (SSCANF (s, L("%d"), &i) != 1)
      {
	printf ("FAIL: bug16618: SSCANF did not read one input item.\n");
	result = 1;
      }
    if (i != 0)
      {
	printf ("FAIL: bug16618: Value of `i` was not zero as expected.\n");
	result = 1;
      }
    free (s);
    if (result != 1)
      printf ("PASS: bug16618: Did not crash.\n");
#undef SIZE
  }


  return result;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
