/* Test string function in a transactionally executing RTM region.
   Copyright (C) 2021 Free Software Foundation, Inc.
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

#include <string.h>
#include <x86intrin.h>
#include <sys/platform/x86.h>
#include <support/check.h>
#include <support/test-driver.h>

static int
do_test_1 (const char *name, unsigned int loop, int (*prepare) (void),
	   int (*function) (void))
{
  if (!CPU_FEATURE_ACTIVE (RTM))
    return EXIT_UNSUPPORTED;

  int status = prepare ();
  if (status != EXIT_SUCCESS)
    return status;

  unsigned int i;
  unsigned int naborts = 0;
  unsigned int failed = 0;
  for (i = 0; i < loop; i++)
    {
      failed |= function ();
      if (_xbegin() == _XBEGIN_STARTED)
	{
	  failed |= function ();
	  _xend();
	}
      else
	{
	  failed |= function ();
	  ++naborts;
	}
    }

  if (failed)
    FAIL_EXIT1 ("%s() failed", name);

  if (naborts)
    {
      /* NB: Low single digit (<= 5%) noise-level aborts are normal for
	 TSX.  */
      double rate = 100 * ((double) naborts) / ((double) loop);
      if (rate > 5)
	FAIL_EXIT1 ("TSX abort rate: %.2f%% (%d out of %d)",
		    rate, naborts, loop);
    }

  return EXIT_SUCCESS;
}

static int do_test (void);

#include <support/test-driver.c>
