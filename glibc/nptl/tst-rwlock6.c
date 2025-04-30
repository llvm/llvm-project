/* Copyright (C) 2002-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@redhat.com>, 2002.

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
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>
#include <support/check.h>
#include <support/test-driver.h>
#include <support/timespec.h>
#include <support/xthread.h>
#include <support/xtime.h>


/* A bogus clock value that tells run_test to use pthread_rwlock_timedrdlock
   and pthread_rwlock_timedwrlock rather than pthread_rwlock_clockrdlock and
   pthread_rwlock_clockwrlock.  */
#define CLOCK_USE_TIMEDLOCK (-1)

static int kind[] =
  {
    PTHREAD_RWLOCK_PREFER_READER_NP,
    PTHREAD_RWLOCK_PREFER_WRITER_NONRECURSIVE_NP,
    PTHREAD_RWLOCK_PREFER_WRITER_NP,
  };

struct thread_args
{
  pthread_rwlock_t *rwlock;
  clockid_t clockid;
  const char *fnname;
};

static void *
tf (void *arg)
{
  struct thread_args *args = arg;
  pthread_rwlock_t *r = args->rwlock;
  const clockid_t clockid = args->clockid;
  const clockid_t clockid_for_get =
    (clockid == CLOCK_USE_TIMEDLOCK) ? CLOCK_REALTIME : clockid;
  const char *fnname = args->fnname;

  /* Timeout: 0.3 secs.  */
  struct timespec ts_start;
  xclock_gettime (clockid_for_get, &ts_start);

  struct timespec ts_timeout = timespec_add (ts_start,
                                             make_timespec (0, 300000000));

  verbose_printf ("child calling %srdlock\n", fnname);

  if (clockid == CLOCK_USE_TIMEDLOCK)
    TEST_COMPARE (pthread_rwlock_timedrdlock (r, &ts_timeout), ETIMEDOUT);
  else
    TEST_COMPARE (pthread_rwlock_clockrdlock (r, clockid, &ts_timeout),
                  ETIMEDOUT);

  verbose_printf ("1st child %srdlock done\n", fnname);

  TEST_TIMESPEC_NOW_OR_AFTER (CLOCK_REALTIME, ts_timeout);

  xclock_gettime (clockid_for_get, &ts_timeout);
  ts_timeout.tv_sec += 10;
  /* Note that the following operation makes ts invalid.  */
  ts_timeout.tv_nsec += 1000000000;

  if (clockid == CLOCK_USE_TIMEDLOCK)
    TEST_COMPARE (pthread_rwlock_timedrdlock (r, &ts_timeout), EINVAL);
  else
    TEST_COMPARE (pthread_rwlock_clockrdlock (r, clockid, &ts_timeout), EINVAL);

  verbose_printf ("2nd child %srdlock done\n", fnname);

  return NULL;
}


static int
do_test_clock (clockid_t clockid, const char *fnname)
{
  const clockid_t clockid_for_get =
    (clockid == CLOCK_USE_TIMEDLOCK) ? CLOCK_REALTIME : clockid;
  size_t cnt;
  for (cnt = 0; cnt < sizeof (kind) / sizeof (kind[0]); ++cnt)
    {
      pthread_rwlock_t r;
      pthread_rwlockattr_t a;

      if (pthread_rwlockattr_init (&a) != 0)
        FAIL_EXIT1 ("round %Zu: rwlockattr_t failed\n", cnt);

      if (pthread_rwlockattr_setkind_np (&a, kind[cnt]) != 0)
        FAIL_EXIT1 ("round %Zu: rwlockattr_setkind failed\n", cnt);

      if (pthread_rwlock_init (&r, &a) != 0)
        FAIL_EXIT1 ("round %Zu: rwlock_init failed\n", cnt);

      if (pthread_rwlockattr_destroy (&a) != 0)
        FAIL_EXIT1 ("round %Zu: rwlockattr_destroy failed\n", cnt);

      struct timespec ts;
      xclock_gettime (clockid_for_get, &ts);
      ++ts.tv_sec;

      /* Get a write lock.  */
      int e = (clockid == CLOCK_USE_TIMEDLOCK)
	? pthread_rwlock_timedwrlock (&r, &ts)
	: pthread_rwlock_clockwrlock (&r, clockid, &ts);
      if (e != 0)
        FAIL_EXIT1 ("round %Zu: %swrlock failed (%d)\n",
                    cnt, fnname, e);

      verbose_printf ("1st %swrlock done\n", fnname);

      xclock_gettime (clockid_for_get, &ts);
      ++ts.tv_sec;
      if (clockid == CLOCK_USE_TIMEDLOCK)
        TEST_COMPARE (pthread_rwlock_timedrdlock (&r, &ts), EDEADLK);
      else
        TEST_COMPARE (pthread_rwlock_clockrdlock (&r, clockid, &ts), EDEADLK);

      verbose_printf ("1st %srdlock done\n", fnname);

      xclock_gettime (clockid_for_get, &ts);
      ++ts.tv_sec;
      if (clockid == CLOCK_USE_TIMEDLOCK)
        TEST_COMPARE (pthread_rwlock_timedwrlock (&r, &ts), EDEADLK);
      else
        TEST_COMPARE (pthread_rwlock_clockwrlock (&r, clockid, &ts), EDEADLK);

      verbose_printf ("2nd %swrlock done\n", fnname);

      struct thread_args args;
      args.rwlock = &r;
      args.clockid = clockid;
      args.fnname = fnname;
      pthread_t th = xpthread_create (NULL, tf, &args);

      puts ("started thread");

      (void) xpthread_join (th);

      puts ("joined thread");

      if (pthread_rwlock_destroy (&r) != 0)
        FAIL_EXIT1 ("round %Zu: rwlock_destroy failed\n", cnt);
    }

  return 0;
}

static int do_test (void)
{
  do_test_clock (CLOCK_USE_TIMEDLOCK, "timed");
  do_test_clock (CLOCK_REALTIME, "clock(realtime)");
  do_test_clock (CLOCK_MONOTONIC, "clock(monotonic)");
  return 0;
}

#include <support/test-driver.c>
