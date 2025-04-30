/* Test for sched_rr_get_interval
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
#include <sched.h>
#include <support/check.h>

static int
do_test (void)
{
  struct timespec ts[2] = { { -1, -1 }, { -1, -1 } };
  const struct sched_param param = {
    .sched_priority = sched_get_priority_max (SCHED_RR) - 10,
  };
  int result = sched_setscheduler (0, SCHED_RR, &param);

  if (result != 0)
    FAIL_UNSUPPORTED ("sched_setscheduler error: %m\n");

  TEST_COMPARE (sched_rr_get_interval (0, ts), 0);

  /* Check if returned time values are correct.  */
  TEST_VERIFY (ts[0].tv_sec >= 0);
  TEST_VERIFY (ts[0].tv_nsec >= 0 && ts[0].tv_nsec < 1000000000);
  TEST_VERIFY (ts[1].tv_sec == -1 && ts[1].tv_nsec == -1);

  return 0;
}

#include <support/test-driver.c>
