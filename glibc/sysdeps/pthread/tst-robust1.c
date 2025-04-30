/* Copyright (C) 2005-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@redhat.com>, 2005.

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


static pthread_mutex_t m1;
static pthread_mutex_t m2;
static pthread_barrier_t b;


#ifndef LOCK
# define LOCK(m) pthread_mutex_lock (m)
#endif


static void *
tf (void *arg)
{
  long int round = (long int) arg;

  if (pthread_setcancelstate (PTHREAD_CANCEL_ENABLE, NULL) != 0)
    {
      printf ("%ld: setcancelstate failed\n", round);
      exit (1);
    }

  int e = LOCK (&m1);
  if (e != 0)
    {
      printf ("%ld: child: mutex_lock m1 failed with error %d\n", round, e);
      exit (1);
    }

  e = LOCK (&m2);
  if (e != 0)
    {
      printf ("%ld: child: mutex_lock m2 failed with error %d\n", round, e);
      exit (1);
    }

  e = pthread_barrier_wait (&b);
  if (e != 0 && e != PTHREAD_BARRIER_SERIAL_THREAD)
    {
      printf ("%ld: child: 1st barrier_wait failed\n", round);
      exit (1);
    }

  e = pthread_barrier_wait (&b);
  if (e != 0 && e != PTHREAD_BARRIER_SERIAL_THREAD)
    {
      printf ("%ld: child: 2nd barrier_wait failed\n", round);
      exit (1);
    }

  pthread_testcancel ();

  printf ("%ld: testcancel returned\n", round);
  exit (1);
}


static int
do_test (void)
{
#ifdef PREPARE_TMO
  PREPARE_TMO;
#endif

  pthread_mutexattr_t a;
  if (pthread_mutexattr_init (&a) != 0)
    {
      puts ("mutexattr_init failed");
      return 1;
    }
  if (pthread_mutexattr_setrobust (&a, PTHREAD_MUTEX_ROBUST_NP) != 0)
    {
      puts ("mutexattr_setrobust failed");
      return 1;
    }

#ifdef ENABLE_PI
  if (pthread_mutexattr_setprotocol (&a, PTHREAD_PRIO_INHERIT) != 0)
    {
      puts ("pthread_mutexattr_setprotocol failed");
      return 1;
    }
  else
    {
      int e = pthread_mutex_init (&m1, &a);
      if (e == ENOTSUP)
	{
	  puts ("PI robust mutexes not supported");
	  return 0;
	}
      else if (e != 0)
	{
	  puts ("mutex_init m1 failed");
	  return 1;
	}
      pthread_mutex_destroy (&m1);
    }
#endif

#ifndef NOT_CONSISTENT
  if (pthread_mutex_init (&m1, &a) != 0)
    {
      puts ("mutex_init m1 failed");
      return 1;
    }

  if (pthread_mutex_init (&m2, &a) != 0)
    {
      puts ("mutex_init m2 failed");
      return 1;
    }
#endif

  if (pthread_barrier_init (&b, NULL, 2) != 0)
    {
      puts ("barrier_init failed");
      return 1;
    }

  for (long int round = 1; round < 5; ++round)
    {
#ifdef NOT_CONSISTENT
      if (pthread_mutex_init (&m1 , &a) != 0)
	{
	  puts ("mutex_init m1 failed");
	  return 1;
	}
      if (pthread_mutex_init (&m2 , &a) != 0)
	{
	  puts ("mutex_init m2 failed");
	  return 1;
	}
#endif

      pthread_t th;
      if (pthread_create (&th, NULL, tf, (void *) round) != 0)
	{
	  printf ("%ld: create failed\n", round);
	  return 1;
	}

      int e = pthread_barrier_wait (&b);
      if (e != 0 && e != PTHREAD_BARRIER_SERIAL_THREAD)
	{
	  printf ("%ld: parent: 1st barrier_wait failed\n", round);
	  return 1;
	}

      if (pthread_cancel (th) != 0)
	{
	  printf ("%ld: cancel failed\n", round);
	  return 1;
	}

      e = pthread_barrier_wait (&b);
      if (e != 0 && e != PTHREAD_BARRIER_SERIAL_THREAD)
	{
	  printf ("%ld: parent: 2nd barrier_wait failed\n", round);
	  return 1;
	}

#ifndef AFTER_JOIN
      if (round & 1)
#endif
	{
	  void *res;
	  if (pthread_join (th, &res) != 0)
	    {
	      printf ("%ld: join failed\n", round);
	      return 1;
	    }
	  if (res != PTHREAD_CANCELED)
	    {
	      printf ("%ld: thread not canceled\n", round);
	      return 1;
	    }
	}

      e = LOCK (&m1);
      if (e == 0)
	{
	  printf ("%ld: parent: mutex_lock m1 succeeded\n", round);
	  return 1;
	}
      if (e != EOWNERDEAD)
	{
	  printf ("%ld: parent: mutex_lock m1 returned wrong code\n", round);
	  return 1;
	}

      e = LOCK (&m2);
      if (e == 0)
	{
	  printf ("%ld: parent: mutex_lock m2 succeeded\n", round);
	  return 1;
	}
      if (e != EOWNERDEAD)
	{
	  printf ("%ld: parent: mutex_lock m2 returned wrong code\n", round);
	  return 1;
	}

#ifndef AFTER_JOIN
      if ((round & 1) == 0)
	{
	  void *res;
	  if (pthread_join (th, &res) != 0)
	    {
	      printf ("%ld: join failed\n", round);
	      return 1;
	    }
	  if (res != PTHREAD_CANCELED)
	    {
	      printf ("%ld: thread not canceled\n", round);
	      return 1;
	    }
	}
#endif

#ifndef NOT_CONSISTENT
      e = pthread_mutex_consistent (&m1);
      if (e != 0)
	{
	  printf ("%ld: mutex_consistent m1 failed with error %d\n", round, e);
	  return 1;
	}

      e = pthread_mutex_consistent (&m2);
      if (e != 0)
	{
	  printf ("%ld: mutex_consistent m2 failed with error %d\n", round, e);
	  return 1;
	}
#endif

      e = pthread_mutex_unlock (&m1);
      if (e != 0)
	{
	  printf ("%ld: mutex_unlock m1 failed with %d\n", round, e);
	  return 1;
	}

      e = pthread_mutex_unlock (&m2);
      if (e != 0)
	{
	  printf ("%ld: mutex_unlock m2 failed with %d\n", round, e);
	  return 1;
	}

#ifdef NOT_CONSISTENT
      e = LOCK (&m1);
      if (e == 0)
	{
	  printf ("%ld: locking inconsistent mutex m1 succeeded\n", round);
	  return 1;
	}
      if (e != ENOTRECOVERABLE)
	{
	  printf ("%ld: locking inconsistent mutex m1 failed with error %d\n",
		  round, e);
	  return 1;
	}

      if (pthread_mutex_destroy (&m1) != 0)
	{
	  puts ("mutex_destroy m1 failed");
	  return 1;
	}

      e = LOCK (&m2);
      if (e == 0)
	{
	  printf ("%ld: locking inconsistent mutex m2 succeeded\n", round);
	  return 1;
	}
      if (e != ENOTRECOVERABLE)
	{
	  printf ("%ld: locking inconsistent mutex m2 failed with error %d\n",
		  round, e);
	  return 1;
	}

      if (pthread_mutex_destroy (&m2) != 0)
	{
	  puts ("mutex_destroy m2 failed");
	  return 1;
	}
#endif
    }

#ifndef NOT_CONSISTENT
  if (pthread_mutex_destroy (&m1) != 0)
    {
      puts ("mutex_destroy m1 failed");
      return 1;
    }

  if (pthread_mutex_destroy (&m2) != 0)
    {
      puts ("mutex_destroy m2 failed");
      return 1;
    }
#endif

  if (pthread_mutexattr_destroy (&a) != 0)
    {
      puts ("mutexattr_destroy failed");
      return 1;
    }

  return 0;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
