/* Test for clock_adjtime
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
#include <stdlib.h>
#include <sys/time.h>
#include <support/check.h>
#include <support/timespec.h>

#ifndef ADJTIME_CALL
# define ADJTIME_CALL(__clock, __timex) clock_adjtime (__clock, __timex)
#endif

static int
do_test (void)
{
  struct timespec tv_then, tv_now;
  struct timex delta;

  /* Check if altering target time is allowed.  */
  if (getenv (SETTIME_ENV_NAME) == NULL)
    FAIL_UNSUPPORTED ("clock_adjtime is executed only when "\
                      SETTIME_ENV_NAME" is set\n");

  tv_then = xclock_now (CLOCK_REALTIME);

  /* Setup time value to adjust - 1 sec. */
  delta.time.tv_sec = 1;
  delta.time.tv_usec = 0;
  delta.modes = ADJ_SETOFFSET;

  int ret = ADJTIME_CALL (CLOCK_REALTIME, &delta);
  if (ret == -1)
    FAIL_EXIT1 ("clock_adjtime failed: %m\n");

  tv_now = xclock_now (CLOCK_REALTIME);

  /* Check if clock_adjtime adjusted the system time.  */
  struct timespec r = timespec_sub (tv_now, tv_then);
  TEST_COMPARE (support_timespec_check_in_range
                ((struct timespec) { 1, 0 }, r, 0.9, 1.1), 1);

  return 0;
}

#include <support/test-driver.c>
