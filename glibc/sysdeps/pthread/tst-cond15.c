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
#include <string.h>
#include <unistd.h>
#include <sys/time.h>


static pthread_cond_t cond = PTHREAD_COND_INITIALIZER;
static pthread_mutex_t mut = PTHREAD_RECURSIVE_MUTEX_INITIALIZER_NP;
static pthread_mutex_t mut2 = PTHREAD_MUTEX_INITIALIZER;

static void *
tf (void *p)
{
  if (pthread_mutex_lock (&mut) != 0)
    {
      printf ("%s: 1st mutex_lock failed\n", __func__);
      exit (1);
    }
  if (pthread_mutex_lock (&mut) != 0)
    {
      printf ("%s: 2nd mutex_lock failed\n", __func__);
      exit (1);
    }
  if (pthread_mutex_lock (&mut) != 0)
    {
      printf ("%s: 3rd mutex_lock failed\n", __func__);
      exit (1);
    }

  if (pthread_mutex_unlock (&mut2) != 0)
    {
      printf ("%s: mutex_unlock failed\n", __func__);
      exit (1);
    }

  struct timeval tv;
  gettimeofday (&tv, NULL);
  struct timespec ts;
  TIMEVAL_TO_TIMESPEC (&tv, &ts);
  ts.tv_sec += p == NULL ? 100 : 1;

  int err = pthread_cond_timedwait (&cond, &mut, &ts);
  if ((err != 0 && p == NULL) || (err != ETIMEDOUT && p != NULL))
    {
      printf ("%s: cond_wait failed\n", __func__);
      exit (1);
    }

  if (pthread_mutex_unlock (&mut) != 0)
    {
      printf ("%s: 1st mutex_unlock failed\n", __func__);
      exit (1);
    }
  if (pthread_mutex_unlock (&mut) != 0)
    {
      printf ("%s: 2nd mutex_unlock failed\n", __func__);
      exit (1);
    }
  if (pthread_mutex_unlock (&mut) != 0)
    {
      printf ("%s: 3rd mutex_unlock failed\n", __func__);
      exit (1);
    }

  puts ("child: done");

  return NULL;
}


static int
do_test (void)
{
  if (pthread_mutex_lock (&mut2) != 0)
    {
      puts ("1st mutex_lock failed");
      return 1;
    }

  puts ("parent: create 1st child");

  pthread_t th;
  int err = pthread_create (&th, NULL, tf, NULL);
  if (err != 0)
    {
      printf ("parent: cannot 1st create thread: %s\n", strerror (err));
      return 1;
    }

  /* We have to synchronize with the child.  */
  if (pthread_mutex_lock (&mut2) != 0)
    {
      puts ("2nd mutex_lock failed");
      return 1;
    }

  /* Give the child to reach to pthread_cond_wait.  */
  sleep (1);

  if (pthread_cond_signal (&cond) != 0)
    {
      puts ("cond_signal failed");
      return 1;
    }

  err = pthread_join (th, NULL);
  if (err != 0)
    {
      printf ("parent: failed to join: %s\n", strerror (err));
      return 1;
    }


  puts ("parent: create 2nd child");

  err = pthread_create (&th, NULL, tf, (void *) 1l);
  if (err != 0)
    {
      printf ("parent: cannot 2nd create thread: %s\n", strerror (err));
      return 1;
    }

  err = pthread_join (th, NULL);
  if (err != 0)
    {
      printf ("parent: failed to join: %s\n", strerror (err));
      return 1;
    }

  puts ("done");

  return 0;
}


#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
