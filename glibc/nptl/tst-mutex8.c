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

/* This test checks behavior not required by POSIX.  */
#include <errno.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <elf/dl-tunables.h>

static pthread_mutex_t *m;
static pthread_barrier_t b;
static pthread_cond_t c;
static bool done;


static void
cl (void *arg)
{
  if (pthread_mutex_unlock (m) != 0)
    {
      puts ("cl: mutex_unlocked failed");
      exit (1);
    }
}


static void *
tf (void *arg)
{
  if (pthread_mutex_lock (m) != 0)
    {
      puts ("tf: mutex_lock failed");
      return (void *) 1l;
    }

  int e = pthread_barrier_wait (&b);
  if (e != 0 && e != PTHREAD_BARRIER_SERIAL_THREAD)
    {
      puts ("barrier_wait failed");
      return (void *) 1l;
    }

  if (arg == NULL)
    do
      if (pthread_cond_wait (&c, m) != 0)
	{
	  puts ("tf: cond_wait failed");
	  return (void *) 1l;
	}
    while (! done);
  else
    do
      {
	pthread_cleanup_push (cl, NULL);

	if (pthread_cond_wait (&c, m) != 0)
	  {
	    puts ("tf: cond_wait failed");
	    return (void *) 1l;
	  }

	pthread_cleanup_pop (0);
      }
    while (! done);

  if (pthread_mutex_unlock (m) != 0)
    {
      puts ("tf: mutex_unlock failed");
      return (void *) 1l;
    }

  return NULL;
}


static int
check_type (const char *mas, pthread_mutexattr_t *ma)
{
  int e;

  /* Check if a mutex will be elided.  Lock elision can only be activated via
     the tunables framework.  By default, lock elision is disabled.  */
  bool assume_elided_mutex = false;
#if HAVE_TUNABLES
  int ma_type = PTHREAD_MUTEX_TIMED_NP;
  if (ma != NULL)
    {
      e = pthread_mutexattr_gettype (ma, &ma_type);
      if (e != 0)
	{
	  printf ("pthread_mutexattr_gettype failed with %d (%m)\n", e);
	  return 1;
	}
    }
  if (ma_type == PTHREAD_MUTEX_TIMED_NP)
    {
      /* This type of mutex can be elided if elision is enabled via the tunables
	 framework.  Some tests below are failing if the mutex is elided.
	 Thus we only run those if we assume that the mutex won't be elided.  */
      if (TUNABLE_GET_FULL (glibc, elision, enable, int32_t, NULL) == 1)
	assume_elided_mutex = true;
    }
#endif

  e = pthread_mutex_init (m, ma);
  if (e != 0)
    {
#ifdef ENABLE_PI
      if (e == ENOTSUP)
	{
	  puts ("PI mutexes unsupported");
	  return 0;
	}
#endif
      printf ("1st mutex_init failed for %s\n", mas);
      return 1;
    }

  if (pthread_mutex_destroy (m) != 0)
    {
      printf ("immediate mutex_destroy failed for %s\n", mas);
      return 1;
    }

  if (pthread_mutex_init (m, ma) != 0)
    {
      printf ("2nd mutex_init failed for %s\n", mas);
      return 1;
    }

  if (pthread_mutex_lock (m) != 0)
    {
      printf ("1st mutex_lock failed for %s\n", mas);
      return 1;
    }

  /* Elided mutexes don't fail destroy, thus only test this if we don't assume
     elision.  */
  if (assume_elided_mutex == false)
    {
      e = pthread_mutex_destroy (m);
      if (e == 0)
	{
	  printf ("mutex_destroy of self-locked mutex succeeded for %s\n", mas);
	  return 1;
	}
      if (e != EBUSY)
	{
	  printf ("\
mutex_destroy of self-locked mutex did not return EBUSY %s\n",
		  mas);
	  return 1;
	}
    }

  if (pthread_mutex_unlock (m) != 0)
    {
      printf ("1st mutex_unlock failed for %s\n", mas);
      return 1;
    }

  if (pthread_mutex_trylock (m) != 0)
    {
      printf ("mutex_trylock failed for %s\n", mas);
      return 1;
    }

  /* Elided mutexes don't fail destroy.  */
  if (assume_elided_mutex == false)
    {
      e = pthread_mutex_destroy (m);
      if (e == 0)
	{
	  printf ("mutex_destroy of self-trylocked mutex succeeded for %s\n",
		  mas);
	  return 1;
	}
      if (e != EBUSY)
	{
	  printf ("\
mutex_destroy of self-trylocked mutex did not return EBUSY %s\n",
		  mas);
	  return 1;
	}
    }

  if (pthread_mutex_unlock (m) != 0)
    {
      printf ("2nd mutex_unlock failed for %s\n", mas);
      return 1;
    }

  pthread_t th;
  if (pthread_create (&th, NULL, tf, NULL) != 0)
    {
      puts ("1st create failed");
      return 1;
    }
  done = false;

  e = pthread_barrier_wait (&b);
  if (e != 0 && e != PTHREAD_BARRIER_SERIAL_THREAD)
    {
      puts ("1st barrier_wait failed");
      return 1;
    }

  if (pthread_mutex_lock (m) != 0)
    {
      printf ("2nd mutex_lock failed for %s\n", mas);
      return 1;
    }

  if (pthread_mutex_unlock (m) != 0)
    {
      printf ("3rd mutex_unlock failed for %s\n", mas);
      return 1;
    }

  /* Elided mutexes don't fail destroy.  */
  if (assume_elided_mutex == false)
    {
      e = pthread_mutex_destroy (m);
      if (e == 0)
	{
	  printf ("mutex_destroy of condvar-used mutex succeeded for %s\n",
		  mas);
	  return 1;
	}
      if (e != EBUSY)
	{
	  printf ("\
mutex_destroy of condvar-used mutex did not return EBUSY for %s\n", mas);
	  return 1;
	}
    }

  done = true;
  if (pthread_cond_signal (&c) != 0)
    {
      puts ("cond_signal failed");
      return 1;
    }

  void *r;
  if (pthread_join (th, &r) != 0)
    {
      puts ("join failed");
      return 1;
    }
  if (r != NULL)
    {
      puts ("thread didn't return NULL");
      return 1;
    }

  if (pthread_mutex_destroy (m) != 0)
    {
      printf ("mutex_destroy after condvar-use failed for %s\n", mas);
      return 1;
    }

  if (pthread_mutex_init (m, ma) != 0)
    {
      printf ("3rd mutex_init failed for %s\n", mas);
      return 1;
    }

  if (pthread_create (&th, NULL, tf, (void *) 1) != 0)
    {
      puts ("2nd create failed");
      return 1;
    }
  done = false;

  e = pthread_barrier_wait (&b);
  if (e != 0 && e != PTHREAD_BARRIER_SERIAL_THREAD)
    {
      puts ("2nd barrier_wait failed");
      return 1;
    }

  if (pthread_mutex_lock (m) != 0)
    {
      printf ("3rd mutex_lock failed for %s\n", mas);
      return 1;
    }

  if (pthread_mutex_unlock (m) != 0)
    {
      printf ("4th mutex_unlock failed for %s\n", mas);
      return 1;
    }

  /* Elided mutexes don't fail destroy.  */
  if (assume_elided_mutex == false)
    {
      e = pthread_mutex_destroy (m);
      if (e == 0)
	{
	  printf ("2nd mutex_destroy of condvar-used mutex succeeded for %s\n",
		  mas);
	  return 1;
	}
      if (e != EBUSY)
	{
	  printf ("\
2nd mutex_destroy of condvar-used mutex did not return EBUSY for %s\n",
		  mas);
	  return 1;
	}
    }

  if (pthread_cancel (th) != 0)
    {
      puts ("cond_cancel failed");
      return 1;
    }

  if (pthread_join (th, &r) != 0)
    {
      puts ("join failed");
      return 1;
    }
  if (r != PTHREAD_CANCELED)
    {
      puts ("thread not canceled");
      return 1;
    }

  if (pthread_mutex_destroy (m) != 0)
    {
      printf ("mutex_destroy after condvar-canceled failed for %s\n", mas);
      return 1;
    }

  return 0;
}


static int
do_test (void)
{
  pthread_mutex_t mm;
  m = &mm;

  if (pthread_barrier_init (&b, NULL, 2) != 0)
    {
      puts ("barrier_init failed");
      return 1;
    }

  if (pthread_cond_init (&c, NULL) != 0)
    {
      puts ("cond_init failed");
      return 1;
    }

  puts ("check normal mutex");
  int res = check_type ("normal", NULL);

  pthread_mutexattr_t ma;
  if (pthread_mutexattr_init (&ma) != 0)
    {
      puts ("1st mutexattr_init failed");
      return 1;
    }
  if (pthread_mutexattr_settype (&ma, PTHREAD_MUTEX_RECURSIVE) != 0)
    {
      puts ("1st mutexattr_settype failed");
      return 1;
    }
#ifdef ENABLE_PI
  if (pthread_mutexattr_setprotocol (&ma, PTHREAD_PRIO_INHERIT))
    {
      puts ("1st pthread_mutexattr_setprotocol failed");
      return 1;
    }
#endif
  puts ("check recursive mutex");
  res |= check_type ("recursive", &ma);
  if (pthread_mutexattr_destroy (&ma) != 0)
    {
      puts ("1st mutexattr_destroy failed");
      return 1;
    }

  if (pthread_mutexattr_init (&ma) != 0)
    {
      puts ("2nd mutexattr_init failed");
      return 1;
    }
  if (pthread_mutexattr_settype (&ma, PTHREAD_MUTEX_ERRORCHECK) != 0)
    {
      puts ("2nd mutexattr_settype failed");
      return 1;
    }
#ifdef ENABLE_PI
  if (pthread_mutexattr_setprotocol (&ma, PTHREAD_PRIO_INHERIT))
    {
      puts ("2nd pthread_mutexattr_setprotocol failed");
      return 1;
    }
#endif
  puts ("check error-checking mutex");
  res |= check_type ("error-checking", &ma);
  if (pthread_mutexattr_destroy (&ma) != 0)
    {
      puts ("2nd mutexattr_destroy failed");
      return 1;
    }

  return res;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
