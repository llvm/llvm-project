/* Copyright (C) 2004-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@redhat.com>, 2004.

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
#include <stdlib.h>
#include <time.h>
#include <support/check.h>
#include <support/xthread.h>
#include <support/xtime.h>


static pthread_barrier_t b;
static pthread_rwlock_t r = PTHREAD_RWLOCK_INITIALIZER;


static void *
tf (void *arg)
{
  /* Lock the read-write lock.  */
  TEST_COMPARE (pthread_rwlock_wrlock (&r), 0);

  pthread_t mt = *(pthread_t *) arg;

  xpthread_barrier_wait (&b);

  /* This call will never return.  */
  xpthread_join (mt);

  return NULL;
}


static int
do_test (void)
{
  struct timespec ts;

  xclock_gettime (CLOCK_REALTIME, &ts);
  xpthread_barrier_init (&b, NULL, 2);

  pthread_t me = pthread_self ();
  xpthread_create (NULL, tf, &me);

  /* Wait until the rwlock is locked.  */
  xpthread_barrier_wait (&b);

  ts.tv_nsec = -1;

  TEST_COMPARE (pthread_rwlock_timedrdlock (&r, &ts), EINVAL);
  TEST_COMPARE (pthread_rwlock_clockrdlock (&r, CLOCK_REALTIME, &ts), EINVAL);
  TEST_COMPARE (pthread_rwlock_clockrdlock (&r, CLOCK_MONOTONIC, &ts), EINVAL);
  TEST_COMPARE (pthread_rwlock_timedwrlock (&r, &ts), EINVAL);
  TEST_COMPARE (pthread_rwlock_clockwrlock (&r, CLOCK_REALTIME, &ts), EINVAL);
  TEST_COMPARE (pthread_rwlock_clockwrlock (&r, CLOCK_MONOTONIC, &ts), EINVAL);

  ts.tv_nsec = 1000000000;

  TEST_COMPARE (pthread_rwlock_timedrdlock (&r, &ts), EINVAL);
  TEST_COMPARE (pthread_rwlock_clockrdlock (&r, CLOCK_REALTIME, &ts), EINVAL);
  TEST_COMPARE (pthread_rwlock_clockrdlock (&r, CLOCK_MONOTONIC, &ts), EINVAL);
  TEST_COMPARE (pthread_rwlock_timedwrlock (&r, &ts), EINVAL);
  TEST_COMPARE (pthread_rwlock_clockwrlock (&r, CLOCK_REALTIME, &ts), EINVAL);
  TEST_COMPARE (pthread_rwlock_clockwrlock (&r, CLOCK_MONOTONIC, &ts), EINVAL);

  ts.tv_nsec = (__typeof (ts.tv_nsec)) 0x100001000LL;
  if ((__typeof (ts.tv_nsec)) 0x100001000LL != 0x100001000LL)
    ts.tv_nsec = 2000000000;

  TEST_COMPARE (pthread_rwlock_timedrdlock (&r, &ts), EINVAL);
  TEST_COMPARE (pthread_rwlock_clockrdlock (&r, CLOCK_REALTIME, &ts), EINVAL);
  TEST_COMPARE (pthread_rwlock_clockrdlock (&r, CLOCK_MONOTONIC, &ts), EINVAL);
  TEST_COMPARE (pthread_rwlock_timedwrlock (&r, &ts), EINVAL);
  TEST_COMPARE (pthread_rwlock_clockwrlock (&r, CLOCK_REALTIME, &ts), EINVAL);
  TEST_COMPARE (pthread_rwlock_clockwrlock (&r, CLOCK_MONOTONIC, &ts), EINVAL);

  return 0;
}

#include <support/test-driver.c>
