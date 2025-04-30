/* Basic tests for __strtod_internal.
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

/* Perform a few tests in a locale with thousands separators.  */
static int
do_test (void)
{
  static const struct
  {
    const char *loc;
    const char *str;
    double exp;
    ptrdiff_t nread;
  } tests[] =
    {
      { "de_DE.UTF-8", "1,5", 1.5, 3 },
      { "de_DE.UTF-8", "1.5", 1.0, 1 },
      { "de_DE.UTF-8", "1.500", 1500.0, 5 },
      { "de_DE.UTF-8", "36.893.488.147.419.103.232", 0x1.0p65, 26 }
    };
#define ntests (sizeof (tests) / sizeof (tests[0]))
  size_t n;
  int result = 0;

  puts ("\nLocale tests");

  for (n = 0; n < ntests; ++n)
    {
      double d;
      char *endp;

      if (setlocale (LC_ALL, tests[n].loc) == NULL)
	{
	  printf ("cannot set locale %s\n", tests[n].loc);
	  result = 1;
	  continue;
	}

      d = __strtod_internal (tests[n].str, &endp, 1);
      if (d != tests[n].exp)
	{
	  printf ("strtod(\"%s\") returns %g and not %g\n",
		  tests[n].str, d, tests[n].exp);
	  result = 1;
	}
      else if (endp - tests[n].str != tests[n].nread)
	{
	  printf ("strtod(\"%s\") read %td bytes and not %td\n",
		  tests[n].str, endp - tests[n].str, tests[n].nread);
	  result = 1;
	}
    }

  if (result == 0)
    puts ("all OK");

  return result ? EXIT_FAILURE : EXIT_SUCCESS;
}

#include <support/test-driver.c>
