/* Copyright (C) 2003-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@redhat.com>, 2003.

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
#include <pthread.h>
#include <stdio.h>
#include <time.h>
#include <unistd.h>
#include <support/check.h>
#include <support/test-driver.h>
#include <support/timespec.h>
#include <support/xthread.h>
#include <support/xtime.h>

/* A bogus clock value that tells run_test to use pthread_cond_timedwait
   rather than pthread_condclockwait.  */
#define CLOCK_USE_ATTR_CLOCK (-1)

#if defined _POSIX_CLOCK_SELECTION && _POSIX_CLOCK_SELECTION >= 0
static int
run_test (clockid_t attr_clock, clockid_t wait_clock)
{
  pthread_condattr_t condattr;
  pthread_cond_t cond;
  pthread_mutexattr_t mutattr;
  pthread_mutex_t mut;

  verbose_printf ("attr_clock = %d\n", (int) attr_clock);

  TEST_COMPARE (pthread_condattr_init (&condattr), 0);
  TEST_COMPARE (pthread_condattr_setclock (&condattr, attr_clock), 0);

  clockid_t attr_clock_read;
  TEST_COMPARE (pthread_condattr_getclock (&condattr, &attr_clock_read), 0);
  TEST_COMPARE (attr_clock, attr_clock_read);

  TEST_COMPARE (pthread_cond_init (&cond, &condattr), 0);
  TEST_COMPARE (pthread_condattr_destroy (&condattr), 0);

  xpthread_mutexattr_init (&mutattr);
  xpthread_mutexattr_settype (&mutattr, PTHREAD_MUTEX_ERRORCHECK);
  xpthread_mutex_init (&mut, &mutattr);
  xpthread_mutexattr_destroy (&mutattr);

  xpthread_mutex_lock (&mut);
  TEST_COMPARE (pthread_mutex_lock (&mut), EDEADLK);

  struct timespec ts_timeout;
  xclock_gettime (wait_clock == CLOCK_USE_ATTR_CLOCK ? attr_clock : wait_clock,
                  &ts_timeout);

  /* Wait one second.  */
  ++ts_timeout.tv_sec;

  if (wait_clock == CLOCK_USE_ATTR_CLOCK) {
    TEST_COMPARE (pthread_cond_timedwait (&cond, &mut, &ts_timeout), ETIMEDOUT);
    TEST_TIMESPEC_BEFORE_NOW (ts_timeout, attr_clock);
  } else {
    TEST_COMPARE (pthread_cond_clockwait (&cond, &mut, wait_clock, &ts_timeout),
                  ETIMEDOUT);
    TEST_TIMESPEC_BEFORE_NOW (ts_timeout, wait_clock);
  }

  xpthread_mutex_unlock (&mut);
  xpthread_mutex_destroy (&mut);
  TEST_COMPARE (pthread_cond_destroy (&cond), 0);

  return 0;
}
#endif


static int
do_test (void)
{
#if !defined _POSIX_CLOCK_SELECTION || _POSIX_CLOCK_SELECTION == -1

  FAIL_UNSUPPORTED ("_POSIX_CLOCK_SELECTION not supported, test skipped");

#else

  run_test (CLOCK_REALTIME, CLOCK_USE_ATTR_CLOCK);

# if defined _POSIX_MONOTONIC_CLOCK && _POSIX_MONOTONIC_CLOCK >= 0
#  if _POSIX_MONOTONIC_CLOCK == 0
  int e = sysconf (_SC_MONOTONIC_CLOCK);
  if (e < 0)
    puts ("CLOCK_MONOTONIC not supported");
  else if (e == 0)
      FAIL_RET ("sysconf (_SC_MONOTONIC_CLOCK) must not return 0");
  else
#  endif
    {
      run_test (CLOCK_MONOTONIC, CLOCK_USE_ATTR_CLOCK);
      run_test (CLOCK_REALTIME, CLOCK_MONOTONIC);
      run_test (CLOCK_MONOTONIC, CLOCK_MONOTONIC);
      run_test (CLOCK_MONOTONIC, CLOCK_REALTIME);
    }
# else
  puts ("_POSIX_MONOTONIC_CLOCK not defined");
# endif

  return 0;
#endif
}

#include <support/test-driver.c>
