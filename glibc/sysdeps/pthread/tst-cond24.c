/* Verify that condition variables synchronized by PI mutexes don't hang.
   Copyright (C) 2012-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <sys/types.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <sys/time.h>
#include <time.h>

#define THREADS_NUM 5
#define MAXITER 50000

static pthread_mutex_t mutex;
static pthread_mutexattr_t mutex_attr;
static pthread_cond_t cond;
static pthread_t threads[THREADS_NUM];
static int pending = 0;

typedef void * (*threadfunc) (void *);

void *
thread_fun_timed (void *arg)
{
  int *ret = arg;
  int rv, i;

  printf ("Started thread_fun_timed[%d]\n", *ret);

  for (i = 0; i < MAXITER / THREADS_NUM; i++)
    {
      rv = pthread_mutex_lock (&mutex);
      if (rv)
        {
	  printf ("pthread_mutex_lock: %s(%d)\n", strerror (rv), rv);
	  *ret = 1;
	  goto out;
	}

      while (!pending)
	{
	  struct timespec ts;
	  clock_gettime(CLOCK_REALTIME, &ts);
	  ts.tv_sec += 20;
	  rv = pthread_cond_timedwait (&cond, &mutex, &ts);

	  /* There should be no timeout either.  */
	  if (rv)
            {
	      printf ("pthread_cond_wait: %s(%d)\n", strerror (rv), rv);
	      *ret = 1;
	      goto out;
	    }
	}

      pending--;

      rv = pthread_mutex_unlock (&mutex);
      if (rv)
        {
	  printf ("pthread_mutex_unlock: %s(%d)\n", strerror (rv), rv);
	  *ret = 1;
	  goto out;
	}
    }

  *ret = 0;

out:
  return ret;
}

void *
thread_fun (void *arg)
{
  int *ret = arg;
  int rv, i;

  printf ("Started thread_fun[%d]\n", *ret);

  for (i = 0; i < MAXITER / THREADS_NUM; i++)
    {
      rv = pthread_mutex_lock (&mutex);
      if (rv)
        {
	  printf ("pthread_mutex_lock: %s(%d)\n", strerror (rv), rv);
	  *ret = 1;
	  goto out;
	}

      while (!pending)
	{
	  rv = pthread_cond_wait (&cond, &mutex);

	  if (rv)
            {
	      printf ("pthread_cond_wait: %s(%d)\n", strerror (rv), rv);
	      *ret = 1;
	      goto out;
	    }
	}

      pending--;

      rv = pthread_mutex_unlock (&mutex);
      if (rv)
        {
	  printf ("pthread_mutex_unlock: %s(%d)\n", strerror (rv), rv);
	  *ret = 1;
	  goto out;
	}
    }

  *ret = 0;

out:
  return ret;
}

static int
do_test_wait (threadfunc f)
{
  int i;
  int rv;
  int counter = 0;
  int retval[THREADS_NUM];

  puts ("Starting test");

  rv = pthread_mutexattr_init (&mutex_attr);
  if (rv)
    {
      printf ("pthread_mutexattr_init: %s(%d)\n", strerror (rv), rv);
      return 1;
    }

  rv = pthread_mutexattr_setprotocol (&mutex_attr, PTHREAD_PRIO_INHERIT);
  if (rv)
    {
      printf ("pthread_mutexattr_setprotocol: %s(%d)\n", strerror (rv), rv);
      return 1;
    }

  rv = pthread_mutex_init (&mutex, &mutex_attr);
  if (rv)
    {
      printf ("pthread_mutex_init: %s(%d)\n", strerror (rv), rv);
      return 1;
    }

  rv = pthread_cond_init (&cond, NULL);
  if (rv)
    {
      printf ("pthread_cond_init: %s(%d)\n", strerror (rv), rv);
      return 1;
    }

  for (i = 0; i < THREADS_NUM; i++)
    {
      retval[i] = i;
      rv = pthread_create (&threads[i], NULL, f, &retval[i]);
      if (rv)
        {
          printf ("pthread_create: %s(%d)\n", strerror (rv), rv);
          return 1;
        }
    }

  for (; counter < MAXITER; counter++)
    {
      rv = pthread_mutex_lock (&mutex);
      if (rv)
        {
          printf ("pthread_mutex_lock: %s(%d)\n", strerror (rv), rv);
          return 1;
        }

      if (!(counter % 100))
	printf ("counter: %d\n", counter);
      pending += 1;

      rv = pthread_cond_signal (&cond);
      if (rv)
        {
          printf ("pthread_cond_signal: %s(%d)\n", strerror (rv), rv);
          return 1;
        }

      rv = pthread_mutex_unlock (&mutex);
      if (rv)
        {
          printf ("pthread_mutex_unlock: %s(%d)\n", strerror (rv), rv);
          return 1;
        }
    }

  for (i = 0; i < THREADS_NUM; i++)
    {
      void *ret;
      rv = pthread_join (threads[i], &ret);
      if (rv)
        {
          printf ("pthread_join: %s(%d)\n", strerror (rv), rv);
          return 1;
        }
      if (ret && *(int *)ret)
        {
	  printf ("Thread %d returned with an error\n", i);
	  return 1;
	}
    }

  return 0;
}

static int
do_test (void)
{
  puts ("Testing pthread_cond_wait");
  int ret = do_test_wait (thread_fun);
  if (ret)
    return ret;

  puts ("Testing pthread_cond_timedwait");
  return do_test_wait (thread_fun_timed);
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
