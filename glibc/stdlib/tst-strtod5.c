/* Tests of strtod in a locale using decimal comma.
   Copyright (C) 2007-2021 Free Software Foundation, Inc.
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
#include <string.h>
#include <math.h>

#define NBSP "\xc2\xa0"

static const struct
{
  const char *in;
  double expected;
} tests[] =
  {
    { "0", 0.0 },
    { "000", 0.0 },
    { "-0", -0.0 },
    { "-000", -0.0 },
    { "0,", 0.0 },
    { "-0,", -0.0 },
    { "0,0", 0.0 },
    { "-0,0", -0.0 },
    { "0e-10", 0.0 },
    { "-0e-10", -0.0 },
    { "0,e-10", 0.0 },
    { "-0,e-10", -0.0 },
    { "0,0e-10", 0.0 },
    { "-0,0e-10", -0.0 },
    { "0e-1000000", 0.0 },
    { "-0e-1000000", -0.0 },
    { "0,0e-1000000", 0.0 },
    { "-0,0e-1000000", -0.0 },
  };
#define NTESTS (sizeof (tests) / sizeof (tests[0]))


static int
do_test (void)
{
  if (setlocale (LC_ALL, "cs_CZ.UTF-8") == NULL)
    {
      puts ("could not set locale");
      return 1;
    }

  int status = 0;

  for (int i = 0; i < NTESTS; ++i)
    {
      char *ep;
      double r = strtod (tests[i].in, &ep);

      if (*ep != '\0')
	{
	  printf ("%d: got rest string \"%s\", expected \"\"\n", i, ep);
	  status = 1;
	}

      if (r != tests[i].expected
	  || copysign (10.0, r) != copysign (10.0, tests[i].expected))
	{
	  printf ("%d: got wrong results %g, expected %g\n",
		  i, r, tests[i].expected);
	  status = 1;
	}
    }

  return status;
}

#include <support/test-driver.c>
