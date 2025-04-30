/* Measure various lock acquisition times for empty critical sections.
   Copyright (C) 2020-2021 Free Software Foundation, Inc.
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

#define TEST_MAIN
#define TEST_NAME "pthread-locks"

#include <stdio.h>
#include <string.h>
#include <limits.h>
#include <stdlib.h>
#include <pthread.h>
#include <semaphore.h>
#include <stdatomic.h>
#include <sys/time.h>
#include <math.h>
#include "bench-timing.h"
#include "json-lib.h"

/* The point of this benchmark is to measure the overhead of an empty
   critical section or a small critical section.  This is never going
   to be indicative of real application performance.  Instead we are
   trying to benchmark the effects of the compiler and the runtime
   coupled with a particular set of hardware atomic operations.
   The numbers from this benchmark should be taken with a massive gain
   of salt and viewed through the eyes of expert reviewers.  */

static pthread_mutex_t m;
static pthread_rwlock_t rw;
static pthread_cond_t cv;
static pthread_cond_t consumer_c, producer_c;
static int cv_done;
static pthread_spinlock_t sp;
static sem_t sem;

typedef timing_t (*test_t)(long, int);

#define START_ITERS 1000

#define FILLER_GOES_HERE \
  if (filler) \
    do_filler ();

/* Everyone loves a good fibonacci series.  This isn't quite one of
   them because we need larger values in fewer steps, in a way that
   won't be optimized away.  We're looking to approximately double the
   total time each test iteration takes, so as to not swamp the useful
   timings.  */

#pragma GCC push_options
#pragma GCC optimize(1)

static int __attribute__((noinline))
fibonacci (int i)
{
  asm("");
  if (i > 2)
    return fibonacci (i-1) + fibonacci (i-2);
  return 10+i;
}

static void
do_filler (void)
{
  static char buf1[512], buf2[512];
  int f = fibonacci (5);
  memcpy (buf1, buf2, f);
}

#pragma GCC pop_options

static timing_t
test_mutex (long iters, int filler)
{
  timing_t start, stop, cur;

  pthread_mutex_init (&m, NULL);

  TIMING_NOW (start);
  for (long j = iters; j >= 0; --j)
    {
      pthread_mutex_lock (&m);
      FILLER_GOES_HERE;
      pthread_mutex_unlock (&m);
    }
  TIMING_NOW (stop);
  TIMING_DIFF (cur, start, stop);

  return cur;
}

static timing_t
test_mutex_trylock (long iters, int filler)
{
  timing_t start, stop, cur;

  pthread_mutex_init (&m, NULL);
  pthread_mutex_lock (&m);

  TIMING_NOW (start);
  for (long j = iters; j >= 0; --j)
    {
      pthread_mutex_trylock (&m);
      FILLER_GOES_HERE;
    }
  TIMING_NOW (stop);
  TIMING_DIFF (cur, start, stop);

  pthread_mutex_unlock (&m);
  return cur;
}

static timing_t
test_rwlock_read (long iters, int filler)
{
  timing_t start, stop, cur;

  pthread_rwlock_init (&rw, NULL);

  TIMING_NOW (start);
  for (long j = iters; j >= 0; --j)
    {
      pthread_rwlock_rdlock (&rw);
      FILLER_GOES_HERE;
      pthread_rwlock_unlock (&rw);
    }
  TIMING_NOW (stop);
  TIMING_DIFF (cur, start, stop);

  return cur;
}

static timing_t
test_rwlock_tryread (long iters, int filler)
{
  timing_t start, stop, cur;

  pthread_rwlock_init (&rw, NULL);
  pthread_rwlock_wrlock (&rw);

  TIMING_NOW (start);
  for (long j = iters; j >= 0; --j)
    {
      pthread_rwlock_tryrdlock (&rw);
      FILLER_GOES_HERE;
    }
  TIMING_NOW (stop);
  TIMING_DIFF (cur, start, stop);

  pthread_rwlock_unlock (&rw);
  return cur;
}

static timing_t
test_rwlock_write (long iters, int filler)
{
  timing_t start, stop, cur;

  pthread_rwlock_init (&rw, NULL);

  TIMING_NOW (start);
  for (long j = iters; j >= 0; --j)
    {
      pthread_rwlock_wrlock (&rw);
      FILLER_GOES_HERE;
      pthread_rwlock_unlock (&rw);
    }
  TIMING_NOW (stop);
  TIMING_DIFF (cur, start, stop);

  return cur;
}

static timing_t
test_rwlock_trywrite (long iters, int filler)
{
  timing_t start, stop, cur;

  pthread_rwlock_init (&rw, NULL);
  pthread_rwlock_rdlock (&rw);

  TIMING_NOW (start);
  for (long j = iters; j >= 0; --j)
    {
      pthread_rwlock_trywrlock (&rw);
      FILLER_GOES_HERE;
    }
  TIMING_NOW (stop);
  TIMING_DIFF (cur, start, stop);

  pthread_rwlock_unlock (&rw);
  return cur;
}

static timing_t
test_spin_lock (long iters, int filler)
{
  timing_t start, stop, cur;

  pthread_spin_init (&sp, PTHREAD_PROCESS_PRIVATE);

  TIMING_NOW (start);
  for (long j = iters; j >= 0; --j)
    {
      pthread_spin_lock (&sp);
      FILLER_GOES_HERE;
      pthread_spin_unlock (&sp);
    }
  TIMING_NOW (stop);
  TIMING_DIFF (cur, start, stop);

  return cur;
}

static timing_t
test_spin_trylock (long iters, int filler)
{
  timing_t start, stop, cur;

  pthread_spin_init (&sp, PTHREAD_PROCESS_PRIVATE);
  pthread_spin_lock (&sp);

  TIMING_NOW (start);
  for (long j = iters; j >= 0; --j)
    {
      pthread_spin_trylock (&sp);
      FILLER_GOES_HERE;
    }
  TIMING_NOW (stop);
  TIMING_DIFF (cur, start, stop);

  pthread_spin_unlock (&sp);
  return cur;
}

static timing_t
test_sem_wait (long iters, int filler)
{
  timing_t start, stop, cur;

  sem_init (&sem, 0, 1);

  TIMING_NOW (start);
  for (long j = iters; j >= 0; --j)
    {
      sem_post (&sem);
      FILLER_GOES_HERE;
      sem_wait (&sem);
    }
  TIMING_NOW (stop);
  TIMING_DIFF (cur, start, stop);

  return cur;
}

static timing_t
test_sem_trywait (long iters, int filler)
{
  timing_t start, stop, cur;

  sem_init (&sem, 0, 0);

  TIMING_NOW (start);
  for (long j = iters; j >= 0; --j)
    {
      sem_trywait (&sem);
      FILLER_GOES_HERE;
    }
  TIMING_NOW (stop);
  TIMING_DIFF (cur, start, stop);

  return cur;
}

static void *
test_condvar_helper (void *v)
{
  /* This is wasteful, but the alternative is to add the overhead of a
     mutex lock/unlock to the overall iteration (both threads) and we
     don't want that.  Ideally, this thread would run on an
     independent processing core anyway.  The ONLY goal here is to
     minimize the time the other thread spends waiting for us.  */
  while (__atomic_load_n (&cv_done, __ATOMIC_RELAXED) == 0)
    pthread_cond_signal (&cv);

  return NULL;
}

static timing_t
test_condvar (long iters, int filler)
{
  timing_t start, stop, cur;
  pthread_t helper_id;

  pthread_mutex_init (&m, NULL);
  pthread_cond_init (&cv, NULL);
  pthread_mutex_lock (&m);

  __atomic_store_n (&cv_done, 0, __ATOMIC_RELAXED);
  pthread_create (&helper_id, NULL, test_condvar_helper, &iters);

  TIMING_NOW (start);
  for (long j = iters; j >= 0; --j)
    {
      pthread_cond_wait (&cv, &m);
      FILLER_GOES_HERE;
    }
  TIMING_NOW (stop);
  TIMING_DIFF (cur, start, stop);

  pthread_mutex_unlock (&m);
  __atomic_store_n (&cv_done, 1, __ATOMIC_RELAXED);

  pthread_join (helper_id, NULL);
  return cur;
}

/* How many items are "queued" in our pretend queue.  */
static int queued = 0;

typedef struct Producer_Params {
  long iters;
  int filler;
} Producer_Params;

/* We only benchmark the consumer thread, but both threads are doing
   essentially the same thing, and never run in parallel due to the
   locks.  Thus, even if they run on separate processing cores, we
   count the time for both threads.  */
static void *
test_producer_thread (void *v)
{
  Producer_Params *p = (Producer_Params *) v;
  long iters = p->iters;
  int filler = p->filler;
  long j;

  for (j = iters; j >= 0; --j)
    {
      /* Aquire lock on the queue.  */
      pthread_mutex_lock (&m);
      /* if something's already there, wait.  */
      while (queued > 0)
	pthread_cond_wait (&consumer_c, &m);

      /* Put something on the queue */
      FILLER_GOES_HERE;
      ++ queued;
      pthread_cond_signal (&producer_c);

      /* Give the other thread a chance to run.  */
      pthread_mutex_unlock (&m);
    }

  return NULL;
}

static timing_t
test_consumer_producer (long iters, int filler)
{
  timing_t start, stop, cur;
  pthread_t helper_id;
  Producer_Params p;

  p.iters = iters;
  p.filler = filler;

  pthread_mutex_init (&m, NULL);
  pthread_cond_init (&cv, NULL);

  pthread_create (&helper_id, NULL, test_producer_thread, &p);

  TIMING_NOW (start);

  for (long j = iters; j >= 0; --j)
    {
      /* Aquire lock on the queue.  */
      pthread_mutex_lock (&m);
      /* Wait for something to be on the queue.  */
      while (queued == 0)
	pthread_cond_wait (&producer_c, &m);

      /* Take if off. */
      FILLER_GOES_HERE;
      -- queued;
      pthread_cond_signal (&consumer_c);

      /* Give the other thread a chance to run.  */
      pthread_mutex_unlock (&m);
    }

  TIMING_NOW (stop);
  TIMING_DIFF (cur, start, stop);


  pthread_join (helper_id, NULL);
  return cur;
}

/* Number of runs we use for computing mean and standard deviation.
   We actually do two additional runs and discard the outliers.  */
#define RUN_COUNT 10

static int
do_bench_2 (const char *name, test_t func, int filler, json_ctx_t *js)
{
  timing_t cur;
  struct timeval ts, te;
  double tsd, ted, td;
  long iters, iters_limit;
  timing_t curs[RUN_COUNT + 2];
  int i, j;
  double mean, stdev;

  iters = START_ITERS;
  iters_limit = LONG_MAX / 100;

  while (1) {
    gettimeofday (&ts, NULL);
    cur = func(iters, filler);
    gettimeofday (&te, NULL);

    /* We want a test to take at least 0.01 seconds, and try
       increasingly larger iteration counts until it does.  This
       allows for approximately constant-time tests regardless of
       hardware speed, without the overhead of checking the time
       inside the test loop itself.  We stop at a million iterations
       as that should be precise enough.  Once we determine a suitable
       iteration count, we run the test multiple times to calculate
       mean and standard deviation.  */

    /* Note that this also primes the CPU cache and triggers faster
       MHz, we hope.  */
    tsd = ts.tv_sec + ts.tv_usec / 1000000.0;
    ted = te.tv_sec + te.tv_usec / 1000000.0;
    td = ted - tsd;
    if (td >= 0.01
	|| iters >= iters_limit
	|| iters >= 1000000)
      break;

    iters *= 10;
  }

  curs[0] = cur;
  for (i = 1; i < RUN_COUNT + 2; i ++)
    curs[i] = func(iters, filler);

  /* We sort the results so we can discard the fastest and slowest
     times as outliers.  In theory we should keep the fastest time,
     but IMHO this is more fair.  A simple bubble sort suffices.  */

  for (i = 0; i < RUN_COUNT + 1; i ++)
    for (j = i + 1; j < RUN_COUNT + 2; j ++)
      if (curs[i] > curs[j])
	{
	  timing_t temp = curs[i];
	  curs[i] = curs[j];
	  curs[j] = temp;
	}

  /* Now calculate mean and standard deviation, skipping the outliers.  */
  mean = 0.0;
  for (i = 1; i<RUN_COUNT + 1; i ++)
    mean += (double) curs[i] / (double) iters;
  mean /= RUN_COUNT;

  stdev = 0.0;
  for (i = 1; i < RUN_COUNT + 1; i ++)
    {
      double s = (double) curs[i] / (double) iters - mean;
      stdev += s * s;
    }
  stdev = sqrt (stdev / (RUN_COUNT - 1));

  char buf[128];
  snprintf (buf, sizeof buf, "%s-%s", name, filler ? "filler" : "empty");

  json_attr_object_begin (js, buf);

  json_attr_double (js, "duration", (double) cur);
  json_attr_double (js, "iterations", (double) iters);
  json_attr_double (js, "wall-sec", (double) td);
  json_attr_double (js, "mean", mean);
  json_attr_double (js, "stdev", stdev);
  json_attr_double (js, "min-outlier", (double) curs[0] / (double) iters);
  json_attr_double (js, "min", (double) curs[1] / (double) iters);
  json_attr_double (js, "max", (double) curs[RUN_COUNT] / (double) iters);
  json_attr_double (js, "max-outlier", (double) curs[RUN_COUNT + 1] / (double) iters);

  json_attr_object_end (js);

  return 0;
}

static int
do_bench_1 (const char *name, test_t func, json_ctx_t *js)
{
  int rv = 0;

  rv += do_bench_2 (name, func, 0, js);
  rv += do_bench_2 (name, func, 1, js);

  return rv;
}

int
do_bench (void)
{
  int rv = 0;
  json_ctx_t json_ctx;

  json_init (&json_ctx, 2, stdout);
  json_attr_object_begin (&json_ctx, "pthread_locks");

#define BENCH(n) rv += do_bench_1 (#n, test_##n, &json_ctx)

  BENCH (mutex);
  BENCH (mutex_trylock);
  BENCH (rwlock_read);
  BENCH (rwlock_tryread);
  BENCH (rwlock_write);
  BENCH (rwlock_trywrite);
  BENCH (spin_lock);
  BENCH (spin_trylock);
  BENCH (sem_wait);
  BENCH (sem_trywait);
  BENCH (condvar);
  BENCH (consumer_producer);

  json_attr_object_end (&json_ctx);

  return rv;
}


#define TEST_FUNCTION do_bench ()

#include "../test-skeleton.c"
