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
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>


static pthread_cond_t c = PTHREAD_COND_INITIALIZER;
static pthread_mutex_t m = PTHREAD_MUTEX_INITIALIZER;
static pthread_barrier_t b;

static void *
tf (void *a)
{
  /* Block SIGUSR1.  */
  sigset_t ss;

  sigemptyset (&ss);
  sigaddset (&ss, SIGUSR1);
  if (pthread_sigmask (SIG_BLOCK, &ss, NULL) != 0)
    {
      puts ("child: sigmask failed");
      exit (1);
    }

  if (pthread_mutex_lock (&m) != 0)
    {
      puts ("child: mutex_lock failed");
      exit (1);
    }

  int e = pthread_barrier_wait (&b);
  if (e != 0 && e != PTHREAD_BARRIER_SERIAL_THREAD)
    {
      puts ("child: barrier_wait failed");
      exit (1);
    }

  /* Compute timeout.  */
  struct timeval tv;
  (void) gettimeofday (&tv, NULL);
  struct timespec ts;
  TIMEVAL_TO_TIMESPEC (&tv, &ts);
  /* Timeout: 1sec.  */
  ts.tv_sec += 1;

  /* This call should never return.  */
  if (pthread_cond_timedwait (&c, &m, &ts) != ETIMEDOUT)
    {
      puts ("cond_timedwait didn't time out");
      exit (1);
    }

  return NULL;
}


int
do_test (void)
{
  pthread_t th;

  if (pthread_barrier_init (&b, NULL, 2) != 0)
    {
      puts ("barrier_init failed");
      exit (1);
    }

  if (pthread_create (&th, NULL, tf, NULL) != 0)
    {
      puts ("create failed");
      exit (1);
    }

  int e = pthread_barrier_wait (&b);
  if (e != 0 && e != PTHREAD_BARRIER_SERIAL_THREAD)
    {
      puts ("barrier_wait failed");
      exit (1);
    }

  if (pthread_mutex_lock (&m) != 0)
    {
      puts ("mutex_lock failed");
      exit (1);
    }

  /* Send the thread a signal which it has blocked.  */
  if (pthread_kill (th, SIGUSR1) != 0)
    {
      puts ("kill failed");
      exit (1);
    }

  if (pthread_mutex_unlock (&m) != 0)
    {
      puts ("mutex_unlock failed");
      exit (1);
    }

  void *r;
  if (pthread_join (th, &r) != 0)
    {
      puts ("join failed");
      exit (1);
    }
  if (r != NULL)
    {
      puts ("return value wrong");
      exit (1);
    }

  return 0;
}


#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
