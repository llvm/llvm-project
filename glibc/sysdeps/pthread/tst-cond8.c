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
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>


static pthread_cond_t cond = PTHREAD_COND_INITIALIZER;
static pthread_mutex_t mut = PTHREAD_ERRORCHECK_MUTEX_INITIALIZER_NP;

static pthread_barrier_t bar;


static void
ch (void *arg)
{
  int e = pthread_mutex_lock (&mut);
  if (e == 0)
    {
      puts ("mutex not locked at all by cond_wait");
      exit (1);
    }

  if (e != EDEADLK)
    {
      puts ("no deadlock error signaled");
      exit (1);
    }

  if (pthread_mutex_unlock (&mut) != 0)
    {
      puts ("ch: cannot unlock mutex");
      exit (1);
    }

  puts ("ch done");
}


static void *
tf1 (void *p)
{
  int err;

  if (pthread_setcancelstate (PTHREAD_CANCEL_ENABLE, NULL) != 0
      || pthread_setcanceltype (PTHREAD_CANCEL_DEFERRED, NULL) != 0)
    {
      puts ("cannot set cancellation options");
      exit (1);
    }

  err = pthread_mutex_lock (&mut);
  if (err != 0)
    {
      puts ("child: cannot get mutex");
      exit (1);
    }

  err = pthread_barrier_wait (&bar);
  if (err != 0 && err != PTHREAD_BARRIER_SERIAL_THREAD)
    {
      printf ("barrier_wait returned %d\n", err);
      exit (1);
    }

  puts ("child: got mutex; waiting");

  pthread_cleanup_push (ch, NULL);

  pthread_cond_wait (&cond, &mut);

  pthread_cleanup_pop (0);

  puts ("child: cond_wait should not have returned");

  return NULL;
}


static void *
tf2 (void *p)
{
  int err;

  if (pthread_setcancelstate (PTHREAD_CANCEL_ENABLE, NULL) != 0
      || pthread_setcanceltype (PTHREAD_CANCEL_DEFERRED, NULL) != 0)
    {
      puts ("cannot set cancellation options");
      exit (1);
    }

  err = pthread_mutex_lock (&mut);
  if (err != 0)
    {
      puts ("child: cannot get mutex");
      exit (1);
    }

  err = pthread_barrier_wait (&bar);
  if (err != 0 && err != PTHREAD_BARRIER_SERIAL_THREAD)
    {
      printf ("barrier_wait returned %d\n", err);
      exit (1);
    }

  puts ("child: got mutex; waiting");

  pthread_cleanup_push (ch, NULL);

  /* Current time.  */
  struct timeval tv;
  (void) gettimeofday (&tv, NULL);
  /* +1000 seconds in correct format.  */
  struct timespec ts;
  TIMEVAL_TO_TIMESPEC (&tv, &ts);
  ts.tv_sec += 1000;

  pthread_cond_timedwait (&cond, &mut, &ts);

  pthread_cleanup_pop (0);

  puts ("child: cond_wait should not have returned");

  return NULL;
}


static int
do_test (void)
{
  pthread_t th;
  int err;

  printf ("&cond = %p\n&mut = %p\n", &cond, &mut);

  puts ("parent: get mutex");

  err = pthread_barrier_init (&bar, NULL, 2);
  if (err != 0)
    {
      puts ("parent: cannot init barrier");
      exit (1);
    }

  puts ("parent: create child");

  err = pthread_create (&th, NULL, tf1, NULL);
  if (err != 0)
    {
      puts ("parent: cannot create thread");
      exit (1);
    }

  puts ("parent: wait for child to lock mutex");

  err = pthread_barrier_wait (&bar);
  if (err != 0 && err != PTHREAD_BARRIER_SERIAL_THREAD)
    {
      puts ("parent: cannot wait for barrier");
      exit (1);
    }

  err = pthread_mutex_lock (&mut);
  if (err != 0)
    {
      puts ("parent: mutex_lock failed");
      exit (1);
    }

  err = pthread_mutex_unlock (&mut);
  if (err != 0)
    {
      puts ("parent: mutex_unlock failed");
      exit (1);
    }

  if (pthread_cancel (th) != 0)
    {
      puts ("cannot cancel thread");
      exit (1);
    }

  void *r;
  err = pthread_join (th, &r);
  if (err != 0)
    {
      puts ("parent: failed to join");
      exit (1);
    }

  if (r != PTHREAD_CANCELED)
    {
      puts ("child hasn't been canceled");
      exit (1);
    }



  puts ("parent: create 2nd child");

  err = pthread_create (&th, NULL, tf2, NULL);
  if (err != 0)
    {
      puts ("parent: cannot create thread");
      exit (1);
    }

  puts ("parent: wait for child to lock mutex");

  err = pthread_barrier_wait (&bar);
  if (err != 0 && err != PTHREAD_BARRIER_SERIAL_THREAD)
    {
      puts ("parent: cannot wait for barrier");
      exit (1);
    }

  err = pthread_mutex_lock (&mut);
  if (err != 0)
    {
      puts ("parent: mutex_lock failed");
      exit (1);
    }

  err = pthread_mutex_unlock (&mut);
  if (err != 0)
    {
      puts ("parent: mutex_unlock failed");
      exit (1);
    }

  if (pthread_cancel (th) != 0)
    {
      puts ("cannot cancel thread");
      exit (1);
    }

  err = pthread_join (th, &r);
  if (err != 0)
    {
      puts ("parent: failed to join");
      exit (1);
    }

  if (r != PTHREAD_CANCELED)
    {
      puts ("child hasn't been canceled");
      exit (1);
    }

  puts ("done");

  return 0;
}


#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
