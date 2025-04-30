/* Copyright (C) 2003-2021 Free Software Foundation, Inc.
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
#include <stdio.h>
#include <unistd.h>
#include <sys/time.h>
#include <time.h>
#include <intprops.h>
#include <support/support.h>
#include <support/check.h>

/* Test that clock_nanosleep() does sleep.  */
static void
clock_nanosleep_test (void)
{
  /* Current time.  */
  struct timeval tv1;
  gettimeofday (&tv1, NULL);

  struct timespec ts = { 1, 0 };
  TEMP_FAILURE_RETRY (clock_nanosleep (CLOCK_REALTIME, 0, &ts, &ts));

  /* At least one second must have passed.  */
  struct timeval tv2;
  gettimeofday (&tv2, NULL);

  tv2.tv_sec -= tv1.tv_sec;
  tv2.tv_usec -= tv1.tv_usec;
  if (tv2.tv_usec < 0)
    --tv2.tv_sec;

  TEST_VERIFY (tv2.tv_sec >= 1);
}

static void
clock_nanosleep_large_timeout (void)
{
  support_create_timer (0, 100000000, false, NULL);
  struct timespec ts = { TYPE_MAXIMUM (time_t), 0 };
  int r = clock_nanosleep (CLOCK_REALTIME, 0, &ts, NULL);
  TEST_VERIFY (r == EINTR || r == EOVERFLOW);
}

static int
do_test (void)
{
  clock_nanosleep_test ();
  clock_nanosleep_large_timeout ();
  return 0;
}

#include <support/test-driver.c>
