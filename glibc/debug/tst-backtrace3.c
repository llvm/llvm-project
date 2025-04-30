/* Test backtrace and backtrace_symbols for recursive calls.
   Copyright (C) 2010-2021 Free Software Foundation, Inc.
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

#include <execinfo.h>
#include <search.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tst-backtrace.h"

/* The backtrace should include at least 3 * fn, and do_test.  */
#define NUM_FUNCTIONS 4

NO_INLINE int
fn (int c)
{
  void *addresses[NUM_FUNCTIONS];
  char **symbols;
  int n;
  int i;

  if (c > 0)
    {
      fn (c - 1);
      return x;
    }
  /* Get the backtrace addresses.  */
  n = backtrace (addresses, sizeof (addresses) / sizeof (addresses[0]));
  printf ("Obtained backtrace with %d functions\n", n);
  /*  Check that there are at least four functions.  */
  if (n < NUM_FUNCTIONS)
    {
      FAIL ();
      return 1;
    }
  /* Convert them to symbols.  */
  symbols = backtrace_symbols (addresses, n);
  /* Check that symbols were obtained.  */
  if (symbols == NULL)
    {
      FAIL ();
      return 1;
    }
  for (i = 0; i < n; ++i)
    printf ("Function %d: %s\n", i, symbols[i]);
  /* Check that the function names obtained are accurate.  */
  for (i = 0; i < n - 1; ++i)
    if (!match (symbols[i], "fn"))
      {
	FAIL ();
	return 1;
      }
  /* Symbol names are not available for static functions, so we do not
     check do_test.  */
  return x;
}

NO_INLINE int
do_test (void)
{
  fn (2);
  return ret;
}

#include <support/test-driver.c>
