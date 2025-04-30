/* Test for difftime
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

#include <time.h>
#include <support/check.h>

static void
test_difftime_helper (time_t t1, time_t t0, double exp_val)
{
  double sub = difftime (t1, t0);
  if (sub != exp_val)
    FAIL_EXIT1 ("*** Difftime returned %f (expected %f)\n", sub, exp_val);
}

static int
do_test (void)
{
  time_t t = 1383791700; /* Provide reproductible start value.  */

  /* Check if difftime works with current time.  */
  test_difftime_helper (t + 1800, t - 1800, 3600.0);
  test_difftime_helper (t - 1800, t + 1800, -3600.0);

  t = 0x7FFFFFFF;
  /* Check if we run on port with 32 bit time_t size */
  time_t tov;
  if (__builtin_add_overflow (t, 1, &tov))
    return 0;

  /* Check if the time is converted after 32 bit time_t overflow.  */
  test_difftime_helper (t + 1800, t - 1800, 3600.0);
  test_difftime_helper (t - 1800, t + 1800, -3600.0);

  t = tov;
  test_difftime_helper (t + 1800, t - 1800, 3600.0);
  test_difftime_helper (t - 1800, t + 1800, -3600.0);

  return 0;
}

#include <support/test-driver.c>
