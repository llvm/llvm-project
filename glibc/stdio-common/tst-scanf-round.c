/* Test for correct rounding of negative floating-point numbers by scanf
   (bug 23280).
   Copyright (C) 2018-2021 Free Software Foundation, Inc.
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

#include <fenv.h>
#include <stdio.h>
#include <stdlib.h>
#include <support/check.h>

static int
do_test (void)
{
#ifdef FE_DOWNWARD
  if (fesetround (FE_DOWNWARD) == 0)
    {
      double a = strtod ("-0.1", NULL);
      double b = 0;
      int r = sscanf ("-0.1", "%lf", &b);
      TEST_VERIFY (r == 1);
      TEST_VERIFY (a == b);
    }
#endif
#ifdef FE_UPWARD
  if (fesetround (FE_UPWARD) == 0)
    {
      double a = strtod ("-0.1", NULL);
      double b = 0;
      int r = sscanf ("-0.1", "%lf", &b);
      TEST_VERIFY (r == 1);
      TEST_VERIFY (a == b);
    }
#endif
  return 0;
}

#include <support/test-driver.c>
