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

#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


static pthread_mutex_t m1 = PTHREAD_MUTEX_INITIALIZER;
static pthread_mutex_t m2 = PTHREAD_MUTEX_INITIALIZER;

static int cntr;


static void
cleanup (void *arg)
{
  if (arg != (void *) 42l)
    cntr = 42;
  else
    cntr = 1;
}


static void *
tf (void *arg)
{
  /* Ignore all signals.  This must not have any effect on delivering
     the cancellation signal.  */
  sigset_t ss;

  sigfillset (&ss);

  if (pthread_sigmask (SIG_BLOCK, &ss, NULL) != 0)
    {
      puts ("pthread_sigmask failed");
      exit (1);
    }

  pthread_cleanup_push (cleanup, (void *) 42l);

  int err = pthread_setcanceltype (PTHREAD_CANCEL_ASYNCHRONOUS, NULL);
  if (err != 0)
    {
      printf ("setcanceltype failed: %s\n", strerror (err));
      exit (1);
    }
  /* The following code is not standard compliant: the mutex functions
     must not be called with asynchronous cancellation enabled.  */

  err = pthread_mutex_unlock (&m2);
  if (err != 0)
    {
      printf ("child: mutex_unlock failed: %s\n", strerror (err));
      exit (1);
    }

  err = pthread_mutex_lock (&m1);
  if (err != 0)
    {
      printf ("child: 1st mutex_lock failed: %s\n", strerror (err));
      exit (1);
    }

  /* We should never come here.  */

  pthread_cleanup_pop (0);

  return NULL;
}


static int
do_test (void)
{
  int err;
  pthread_t th;
  int result = 0;
  void *retval;

  /* Get the mutexes.  */
  err = pthread_mutex_lock (&m1);
  if (err != 0)
    {
      printf ("parent: 1st mutex_lock failed: %s\n", strerror (err));
      return 1;
    }
  err = pthread_mutex_lock (&m2);
  if (err != 0)
    {
      printf ("parent: 2nd mutex_lock failed: %s\n", strerror (err));
      return 1;
    }

  err = pthread_create (&th, NULL, tf, NULL);
  if (err != 0)
    {
      printf ("create failed: %s\n", strerror (err));
      return 1;
    }

  err = pthread_mutex_lock (&m2);
  if (err != 0)
    {
      printf ("parent: 3rd mutex_lock failed: %s\n", strerror (err));
      return 1;
    }

  err = pthread_cancel (th);
  if (err != 0)
    {
      printf ("cancel failed: %s\n", strerror (err));
      return 1;
    }

  err = pthread_join (th, &retval);
  if (err != 0)
    {
      printf ("join failed: %s\n", strerror (err));
      return 1;
    }

  if (retval != PTHREAD_CANCELED)
    {
      printf ("wrong return value: %p\n", retval);
      result = 1;
    }

  if (cntr == 42)
    {
      puts ("cleanup handler called with wrong argument");
      result = 1;
    }
  else if (cntr != 1)
    {
      puts ("cleanup handling not called");
      result = 1;
    }

  return result;
}


#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
