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
#include <stdio.h>
#include <stdlib.h>


static pthread_mutex_t m;
static pthread_barrier_t b;


static void *
tf (void *arg)
{
  int e = pthread_mutex_unlock (&m);
  if (e == 0)
    {
      puts ("1st mutex_unlock in child succeeded");
      exit (1);
    }
  if (e != EPERM)
    {
      puts ("1st mutex_unlock in child didn't return EPERM");
      exit (1);
    }

  e = pthread_mutex_trylock (&m);
  if (e == 0)
    {
      puts ("mutex_trylock in second thread succeeded");
      exit (1);
    }
  if (e != EBUSY)
    {
      puts ("mutex_trylock returned wrong value");
      exit (1);
    }

  e = pthread_barrier_wait (&b);
  if (e != 0 && e != PTHREAD_BARRIER_SERIAL_THREAD)
    {
      puts ("barrier_wait failed");
      exit (1);
    }

  e = pthread_barrier_wait (&b);
  if (e != 0 && e != PTHREAD_BARRIER_SERIAL_THREAD)
    {
      puts ("barrier_wait failed");
      exit (1);
    }

  e = pthread_mutex_unlock (&m);
  if (e == 0)
    {
      puts ("2nd mutex_unlock in child succeeded");
      exit (1);
    }
  if (e != EPERM)
    {
      puts ("2nd mutex_unlock in child didn't return EPERM");
      exit (1);
    }

  if (pthread_mutex_trylock (&m) != 0)
    {
      puts ("2nd mutex_trylock in second thread failed");
      exit (1);
    }

  if (pthread_mutex_unlock (&m) != 0)
    {
      puts ("3rd mutex_unlock in second thread failed");
      exit (1);
    }

  return NULL;
}


static int
do_test (void)
{
  pthread_mutexattr_t a;

  if (pthread_mutexattr_init (&a) != 0)
    {
      puts ("mutexattr_init failed");
      return 1;
    }

  if (pthread_mutexattr_settype (&a, PTHREAD_MUTEX_RECURSIVE) != 0)
    {
      puts ("mutexattr_settype failed");
      return 1;
    }

#ifdef ENABLE_PI
  if (pthread_mutexattr_setprotocol (&a, PTHREAD_PRIO_INHERIT) != 0)
    {
      puts ("pthread_mutexattr_setprotocol failed");
      return 1;
    }
#endif

  int e;
  e = pthread_mutex_init (&m, &a);
  if (e != 0)
    {
#ifdef ENABLE_PI
      if (e == ENOTSUP)
	{
	  puts ("PI mutexes unsupported");
	  return 0;
	}
#endif
      puts ("mutex_init failed");
      return 1;
    }

  if (pthread_barrier_init (&b, NULL, 2) != 0)
    {
      puts ("barrier_init failed");
      return 1;
    }

  if (pthread_mutex_lock (&m) != 0)
    {
      puts ("mutex_lock failed");
      return 1;
    }

  if (pthread_mutex_lock (&m) != 0)
    {
      puts ("2nd mutex_lock failed");
      return 1;
    }

  if (pthread_mutex_trylock (&m) != 0)
    {
      puts ("1st trylock failed");
      return 1;
    }

  if (pthread_mutex_unlock (&m) != 0)
    {
      puts ("mutex_unlock failed");
      return 1;
    }

  if (pthread_mutex_unlock (&m) != 0)
    {
      puts ("2nd mutex_unlock failed");
      return 1;
    }

  pthread_t th;
  if (pthread_create (&th, NULL, tf, NULL) != 0)
    {
      puts ("create failed");
      return 1;
    }

  e = pthread_barrier_wait (&b);
  if (e != 0 && e != PTHREAD_BARRIER_SERIAL_THREAD)
    {
      puts ("barrier_wait failed");
      return 1;
    }

  if (pthread_mutex_unlock (&m) != 0)
    {
      puts ("3rd mutex_unlock failed");
      return 1;
    }

  e = pthread_mutex_unlock (&m);
  if (e == 0)
    {
      puts ("4th mutex_unlock succeeded");
      return 1;
    }
  if (e != EPERM)
    {
      puts ("4th mutex_unlock didn't return EPERM");
      return 1;
    }

  e = pthread_barrier_wait (&b);
  if (e != 0 && e != PTHREAD_BARRIER_SERIAL_THREAD)
    {
      puts ("barrier_wait failed");
      return 1;
    }

  if (pthread_join (th, NULL) != 0)
    {
      puts ("join failed");
      return 1;
    }

  if (pthread_barrier_destroy (&b) != 0)
    {
      puts ("barrier_destroy failed");
      return 1;
    }

  if (pthread_mutex_destroy (&m) != 0)
    {
      puts ("mutex_destroy failed");
      return 1;
    }

  if (pthread_mutexattr_destroy (&a) != 0)
    {
      puts ("mutexattr_destroy failed");
      return 1;
    }

  return 0;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
