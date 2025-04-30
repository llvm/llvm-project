/* Test sem_timedwait cancellation for uncontended case.
   Copyright (C) 2003-2021 Free Software Foundation, Inc.
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
#include <semaphore.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/time.h>


static pthread_barrier_t bar;
static sem_t sem;


static void
cleanup (void *arg)
{
  static int ncall;

  if (++ncall != 1)
    {
      puts ("second call to cleanup");
      exit (1);
    }
}


static void *
tf (void *arg)
{
  pthread_cleanup_push (cleanup, NULL);

  int e = pthread_barrier_wait (&bar);
  if (e != 0 && e != PTHREAD_BARRIER_SERIAL_THREAD)
    {
      puts ("error: tf: 1st barrier_wait failed");
      exit (1);
    }

  struct timeval tv;
  (void) gettimeofday (&tv, NULL);

  struct timespec ts;
  TIMEVAL_TO_TIMESPEC (&tv, &ts);

  /* Timeout in 5 seconds.  */
  ts.tv_sec += 5;

  /* This call should block and be cancelable.  */
  sem_timedwait (&sem, &ts);

  pthread_cleanup_pop (0);

  return NULL;
}


static int
do_test (void)
{
  pthread_t th;

  if (pthread_barrier_init (&bar, NULL, 2) != 0)
    {
      puts ("error: barrier_init failed");
      exit (1);
    }

  if (sem_init (&sem, 0, 1) != 0)
    {
      puts ("error: sem_init failed");
      exit (1);
    }

  if (pthread_create (&th, NULL, tf, NULL) != 0)
    {
      puts ("error: create failed");
      exit (1);
    }

  /* Check whether cancellation is honored even before sem_timedwait does
     anything.  */
  if (pthread_cancel (th) != 0)
    {
      puts ("error: 1st cancel failed");
      exit (1);
    }

  int e = pthread_barrier_wait (&bar);
  if (e != 0 && e != PTHREAD_BARRIER_SERIAL_THREAD)
    {
      puts ("1st barrier_wait failed");
      exit (1);
    }

  void *r;
  if (pthread_join (th, &r) != 0)
    {
      puts ("join failed");
      exit (1);
    }

  if (r != PTHREAD_CANCELED)
    {
      puts ("thread not canceled");
      exit (1);
    }

  return 0;
}


#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
