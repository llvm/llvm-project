/* Basic tests for strtod.
   Copyright (C) 1991-2021 Free Software Foundation, Inc.
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

#include <ctype.h>
#include <locale.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>
#include <math.h>

struct ltest
  {
    const char *str;		/* Convert this.  */
    double expect;		/* To get this.  */
    char left;			/* With this left over.  */
    int err;			/* And this in errno.  */
  };
static const struct ltest tests[] =
  {
    { "12.345", 12.345, '\0', 0 },
    { "12.345e19", 12.345e19, '\0', 0 },
    { "-.1e+9", -.1e+9, '\0', 0 },
    { ".125", .125, '\0', 0 },
    { "1e20", 1e20, '\0', 0 },
    { "0e-19", 0, '\0', 0 },
    { "4\00012", 4.0, '\0', 0 },
    { "5.9e-76", 5.9e-76, '\0', 0 },
    { "0x1.4p+3", 10.0, '\0', 0 },
    { "0xAp0", 10.0, '\0', 0 },
    { "0x0Ap0", 10.0, '\0', 0 },
    { "0x0A", 10.0, '\0', 0 },
    { "0xA0", 160.0, '\0', 0 },
    { "0x0.A0p8", 160.0, '\0', 0 },
    { "0x0.50p9", 160.0, '\0', 0 },
    { "0x0.28p10", 160.0, '\0', 0 },
    { "0x0.14p11", 160.0, '\0', 0 },
    { "0x0.0A0p12", 160.0, '\0', 0 },
    { "0x0.050p13", 160.0, '\0', 0 },
    { "0x0.028p14", 160.0, '\0', 0 },
    { "0x0.014p15", 160.0, '\0', 0 },
    { "0x00.00A0p16", 160.0, '\0', 0 },
    { "0x00.0050p17", 160.0, '\0', 0 },
    { "0x00.0028p18", 160.0, '\0', 0 },
    { "0x00.0014p19", 160.0, '\0', 0 },
    { "0x1p-1023",
      1.11253692925360069154511635866620203210960799023116591527666e-308,
      '\0', 0 },
    { "0x0.8p-1022",
      1.11253692925360069154511635866620203210960799023116591527666e-308,
      '\0', 0 },
    { "Inf", HUGE_VAL, '\0', 0 },
    { "-Inf", -HUGE_VAL, '\0', 0 },
    { "+InFiNiTy", HUGE_VAL, '\0', 0 },
    { "0x80000Ap-23", 0x80000Ap-23, '\0', 0 },
    { "1e-324", 0, '\0', ERANGE },
    { "0x100000000000008p0", 0x1p56, '\0', 0 },
    { "0x100000000000008.p0", 0x1p56, '\0', 0 },
    { "0x100000000000008.00p0", 0x1p56, '\0', 0 },
    { "0x10000000000000800p0", 0x1p64, '\0', 0 },
    { "0x10000000000000801p0", 0x1.0000000000001p64, '\0', 0 },
    { NULL, 0, '\0', 0 }
  };

static void expand (char *dst, int c);
static int long_dbl (void);

static int
do_test (void)
{
  char buf[100];
  const struct ltest *lt;
  char *ep;
  int status = 0;
  int save_errno;

  for (lt = tests; lt->str != NULL; ++lt)
    {
      double d;

      errno = 0;
      d = strtod(lt->str, &ep);
      save_errno = errno;
      printf ("strtod (\"%s\") test %u",
	     lt->str, (unsigned int) (lt - tests));
      if (d == lt->expect && *ep == lt->left && save_errno == lt->err)
	puts ("\tOK");
      else
	{
	  puts ("\tBAD");
	  if (d != lt->expect)
	    printf ("  returns %.60g, expected %.60g\n", d, lt->expect);
	  if (lt->left != *ep)
	    {
	      char exp1[5], exp2[5];
	      expand (exp1, *ep);
	      expand (exp2, lt->left);
	      printf ("  leaves '%s', expected '%s'\n", exp1, exp2);
	    }
	  if (save_errno != lt->err)
	    printf ("  errno %d (%s)  instead of %d (%s)\n",
		    save_errno, strerror (save_errno),
		    lt->err, strerror (lt->err));
	  status = 1;
	}
    }

  sprintf (buf, "%f", strtod ("-0.0", NULL));
  if (strcmp (buf, "-0.000000") != 0)
    {
      printf ("  strtod (\"-0.0\", NULL) returns \"%s\"\n", buf);
      status = 1;
    }

  const char input[] = "3752432815e-39";

  float f1 = strtold (input, NULL);
  float f2;
  float f3 = strtof (input, NULL);
  sscanf (input, "%g", &f2);

  if (f1 != f2)
    {
      printf ("f1 = %a != f2 = %a\n", f1, f2);
      status = 1;
    }
  if (f1 != f3)
    {
      printf ("f1 = %a != f3 = %a\n", f1, f3);
      status = 1;
    }
  if (f2 != f3)
    {
      printf ("f2 = %a != f3 = %a\n", f2, f3);
      status = 1;
    }

  const char input2[] = "+1.000000000116415321826934814453125";
  if (strtold (input2, NULL) != +1.000000000116415321826934814453125L)
    {
      printf ("input2: %La != %La\n", strtold (input2, NULL),
	      +1.000000000116415321826934814453125L);
      status = 1;
    }

  static struct { const char *str; long double l; } ltests[] =
    {
      { "42.0000000000000000001", 42.0000000000000000001L },
      { "42.00000000000000000001", 42.00000000000000000001L },
      { "42.000000000000000000001", 42.000000000000000000001L }
    };
  int n;
  for (n = 0; n < sizeof (ltests) / sizeof (ltests[0]); ++n)
    if (strtold (ltests[n].str, NULL) != ltests[n].l)
      {
	printf ("ltests[%d]: %La != %La\n", n,
		strtold (ltests[n].str, NULL), ltests[n].l);
	status = 1;
      }

  status |= long_dbl ();

  return status ? EXIT_FAILURE : EXIT_SUCCESS;
}

static void
expand (char *dst, int c)
{
  if (isprint (c))
    {
      dst[0] = c;
      dst[1] = '\0';
    }
  else
    (void) sprintf (dst, "%#.3o", (unsigned int) c);
}

static int
long_dbl (void)
{
  /* Regenerate this string using

     echo '(2^53-1)*2^(1024-53)' | bc | sed 's/\([^\]*\)\\*$/    "\1"/'

  */
  static const char longestdbl[] =
    "17976931348623157081452742373170435679807056752584499659891747680315"
    "72607800285387605895586327668781715404589535143824642343213268894641"
    "82768467546703537516986049910576551282076245490090389328944075868508"
    "45513394230458323690322294816580855933212334827479782620414472316873"
    "8177180919299881250404026184124858368";
  double d = strtod (longestdbl, NULL);

  printf ("strtod (\"%s\", NULL) = %g\n", longestdbl, d);

  if (d != 179769313486231570814527423731704356798070567525844996598917476803157260780028538760589558632766878171540458953514382464234321326889464182768467546703537516986049910576551282076245490090389328944075868508455133942304583236903222948165808559332123348274797826204144723168738177180919299881250404026184124858368.000000)
    return 1;

  return 0;
}

#include <support/test-driver.c>
