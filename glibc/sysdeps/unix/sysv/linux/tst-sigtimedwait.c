/* Test for sigtimedwait timeout.
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
#include <intprops.h>
#include <errno.h>
#include <signal.h>
#include <support/check.h>
#include <support/xtime.h>
#include <support/timespec.h>
#include <support/support.h>
#include <stdbool.h>

static int
test_sigtimedwait_timeout (bool zero_tmo)
{
  /* We wait for half a second.  */
  struct timespec ts;
  xclock_gettime (CLOCK_REALTIME, &ts);
  struct timespec timeout = make_timespec (0, zero_tmo ? 0 : TIMESPEC_HZ/2);
  ts = timespec_add (ts, timeout);

  /* Set sigset to just wait for timeout.  */
  sigset_t ss_usr1;
  sigemptyset (&ss_usr1);
  sigaddset (&ss_usr1, SIGUSR1);

  int ret = sigtimedwait (&ss_usr1, NULL, &timeout);
  if (ret != -1)
    FAIL_EXIT1 ("sigtimedwait failed: %m\n");

  TEST_TIMESPEC_NOW_OR_AFTER (CLOCK_REALTIME, ts);

  return 0;
}

static void
test_sigtimedwait_large_timeout (void)
{
  support_create_timer (0, 100000000, false, NULL);
  struct timespec ts = { TYPE_MAXIMUM (time_t), 0 };

  sigset_t ss_usr1;
  sigemptyset (&ss_usr1);
  sigaddset (&ss_usr1, SIGUSR1);

  TEST_COMPARE (sigtimedwait (&ss_usr1, NULL, &ts), -1);
  TEST_VERIFY (errno == EINTR || errno == EOVERFLOW);
}

static int
do_test (void)
{
  /* Check if sigtimedwait exits immediately.  */
  test_sigtimedwait_timeout (true);

  /* Check if sigtimedwait exits after specified timeout.  */
  test_sigtimedwait_timeout (false);

  test_sigtimedwait_large_timeout ();

  return 0;
}

#include <support/test-driver.c>
