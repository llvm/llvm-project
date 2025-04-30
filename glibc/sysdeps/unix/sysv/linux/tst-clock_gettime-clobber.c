/* Check that clock_gettime does not clobber errno on success.
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

#include <errno.h>
#include <time.h>
#include <support/check.h>
#include <stdio.h>

static void
test_clock (clockid_t clk)
{
  printf ("info: testing clock: %d\n", (int) clk);

  for (int original_errno = 0; original_errno < 2; ++original_errno)
    {
      errno = original_errno;
      struct timespec ts;
      if (clock_gettime (clk, &ts) == 0)
        TEST_COMPARE (errno, original_errno);
    }
}

static int
do_test (void)
{
  test_clock (CLOCK_BOOTTIME);
  test_clock (CLOCK_BOOTTIME_ALARM);
  test_clock (CLOCK_MONOTONIC);
  test_clock (CLOCK_MONOTONIC_COARSE);
  test_clock (CLOCK_MONOTONIC_RAW);
  test_clock (CLOCK_PROCESS_CPUTIME_ID);
  test_clock (CLOCK_REALTIME);
  test_clock (CLOCK_REALTIME_ALARM);
  test_clock (CLOCK_REALTIME_COARSE);
#ifdef CLOCK_TAI
  test_clock (CLOCK_TAI);
#endif
  test_clock (CLOCK_THREAD_CPUTIME_ID);
  return 0;
}

#include <support/test-driver.c>
