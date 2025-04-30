/* Copyright (C) 2010-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Andreas Schwab <schwab@redhat.com>, 2010.

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
#include <semaphore.h>
#include <stdio.h>
#include <support/check.h>
#include <support/xthread.h>

static pthread_cond_t c = PTHREAD_COND_INITIALIZER;
static pthread_mutex_t m1 = PTHREAD_MUTEX_INITIALIZER;
static pthread_mutex_t m2 = PTHREAD_MUTEX_INITIALIZER;
static pthread_rwlock_t rw1 = PTHREAD_RWLOCK_INITIALIZER;
static pthread_rwlock_t rw2 = PTHREAD_RWLOCK_INITIALIZER;
static sem_t sem;

static void *
th (void *arg)
{
  struct timespec t = { -2, 0 };

  TEST_COMPARE (pthread_mutex_timedlock (&m1, &t), ETIMEDOUT);
  TEST_COMPARE (pthread_mutex_clocklock (&m1, CLOCK_REALTIME, &t), ETIMEDOUT);
  TEST_COMPARE (pthread_mutex_clocklock (&m1, CLOCK_MONOTONIC, &t), ETIMEDOUT);
  TEST_COMPARE (pthread_rwlock_timedrdlock (&rw1, &t), ETIMEDOUT);
  TEST_COMPARE (pthread_rwlock_timedwrlock (&rw2, &t), ETIMEDOUT);
  TEST_COMPARE (pthread_rwlock_clockrdlock (&rw1, CLOCK_REALTIME, &t),
                ETIMEDOUT);
  TEST_COMPARE (pthread_rwlock_clockwrlock (&rw2, CLOCK_REALTIME, &t),
                ETIMEDOUT);
  TEST_COMPARE (pthread_rwlock_clockrdlock (&rw1, CLOCK_MONOTONIC, &t),
                ETIMEDOUT);
  TEST_COMPARE (pthread_rwlock_clockwrlock (&rw2, CLOCK_MONOTONIC, &t),
                ETIMEDOUT);
  return NULL;
}

static int
do_test (void)
{
  struct timespec t = { -2, 0 };

  sem_init (&sem, 0, 0);
  TEST_COMPARE (sem_timedwait (&sem, &t), -1);
  TEST_COMPARE (errno, ETIMEDOUT);

  xpthread_mutex_lock (&m1);
  xpthread_rwlock_wrlock (&rw1);
  xpthread_rwlock_rdlock (&rw2);
  xpthread_mutex_lock (&m2);
  pthread_t pth = xpthread_create (0, th, 0);
  TEST_COMPARE (pthread_cond_timedwait (&c, &m2, &t), ETIMEDOUT);
  xpthread_join (pth);
  return 0;
}

#include <support/test-driver.c>
