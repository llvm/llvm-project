/* Testing race while enabling lock elision.
   Copyright (C) 2018-2021 Free Software Foundation, Inc.
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
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <pthread.h>
#include <unistd.h>
#include <getopt.h>
#include <support/support.h>
#include <support/xthread.h>

static pthread_barrier_t barrier;
static pthread_mutex_t mutex;
static long long int iteration_count = 1000000;
static unsigned int thread_count = 3;

static void *
thr_func (void *arg)
{
  long long int i;
  for (i = 0; i < iteration_count; i++)
    {
      if ((uintptr_t) arg == 0)
	{
	  xpthread_mutex_destroy (&mutex);
	  xpthread_mutex_init (&mutex, NULL);
	}

      xpthread_barrier_wait (&barrier);

      /* Test if enabling lock elision works if it is enabled concurrently.
	 There was a race in FORCE_ELISION macro which leads to either
	 pthread_mutex_destroy returning EBUSY as the owner was recorded
	 by pthread_mutex_lock - in "normal mutex" code path - but was not
	 resetted in pthread_mutex_unlock - in "elision" code path.
	 Or it leads to the assertion in nptl/pthread_mutex_lock.c:
	 assert (mutex->__data.__owner == 0);
	 Please ensure that the test is run with lock elision:
	 export GLIBC_TUNABLES=glibc.elision.enable=1  */
      xpthread_mutex_lock (&mutex);
      xpthread_mutex_unlock (&mutex);

      xpthread_barrier_wait (&barrier);
    }
  return NULL;
}

static int
do_test (void)
{
  unsigned int i;
  printf ("Starting %d threads to run %lld iterations.\n",
	  thread_count, iteration_count);

  pthread_t *threads = xmalloc (thread_count * sizeof (pthread_t));
  xpthread_barrier_init (&barrier, NULL, thread_count);
  xpthread_mutex_init (&mutex, NULL);

  for (i = 0; i < thread_count; i++)
    threads[i] = xpthread_create (NULL, thr_func, (void *) (uintptr_t) i);

  for (i = 0; i < thread_count; i++)
    xpthread_join (threads[i]);

  xpthread_barrier_destroy (&barrier);
  free (threads);

  return EXIT_SUCCESS;
}

#define OPT_ITERATIONS	10000
#define OPT_THREADS	10001
#define CMDLINE_OPTIONS						\
  { "iterations", required_argument, NULL, OPT_ITERATIONS },	\
  { "threads", required_argument, NULL, OPT_THREADS },
static void
cmdline_process (int c)
{
  long long int arg = strtoll (optarg, NULL, 0);
  switch (c)
    {
    case OPT_ITERATIONS:
      if (arg > 0)
	iteration_count = arg;
      break;
    case OPT_THREADS:
      if (arg > 0 && arg < 100)
	thread_count = arg;
      break;
    }
}
#define CMDLINE_PROCESS cmdline_process
#define TIMEOUT 50
#include <support/test-driver.c>
