/* Generic test for CPU affinity functions, multi-threaded variant.
   Copyright (C) 2015-2021 Free Software Foundation, Inc.
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

/* Before including this file, a test has to declare the helper
   getaffinity and setaffinity functions described in
   tst-skeleton-affinity.c, which is included below.  */

#include <errno.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdlib.h>
#include <support/xthread.h>
#include <sys/time.h>

struct conf;
static bool early_test (struct conf *);

/* Arbitrary run time for each pass.  */
#define PASS_TIMEOUT 2

/* There are two passes (one with sched_yield, one without), and we
   double the timeout to be on the safe side.  */
#define TIMEOUT (2 * PASS_TIMEOUT * 2)

#include "tst-skeleton-affinity.c"

/* 0 if still running, 1 of stopping requested.  */
static int still_running;

/* 0 if no scheduling failures, 1 if failures are encountered.  */
static int failed;

static void *
thread_burn_one_cpu (void *closure)
{
  int cpu = (uintptr_t) closure;
  while (__atomic_load_n (&still_running, __ATOMIC_RELAXED) == 0)
    {
      int current = sched_getcpu ();
      if (sched_getcpu () != cpu)
	{
	  printf ("error: Pinned thread %d ran on impossible cpu %d\n",
		  cpu, current);
	  __atomic_store_n (&failed, 1, __ATOMIC_RELAXED);
	  /* Terminate early.  */
	  __atomic_store_n (&still_running, 1, __ATOMIC_RELAXED);
	}
    }
  return NULL;
}

struct burn_thread
{
  pthread_t self;
  struct conf *conf;
  cpu_set_t *initial_set;
  cpu_set_t *seen_set;
  int thread;
};

static void *
thread_burn_any_cpu (void *closure)
{
  struct burn_thread *param = closure;

  /* Schedule this thread around a bit to see if it lands on another
     CPU.  Run this for 2 seconds, once with sched_yield, once
     without.  */
  for (int pass = 1; pass <= 2; ++pass)
    {
      time_t start = time (NULL);
      while (time (NULL) - start <= PASS_TIMEOUT)
	{
	  int cpu = sched_getcpu ();
	  if (cpu > param->conf->last_cpu
	      || !CPU_ISSET_S (cpu, CPU_ALLOC_SIZE (param->conf->set_size),
			       param->initial_set))
	    {
	      printf ("error: Unpinned thread %d ran on impossible CPU %d\n",
		      param->thread, cpu);
	      __atomic_store_n (&failed, 1, __ATOMIC_RELAXED);
	      return NULL;
	    }
	  CPU_SET_S (cpu, CPU_ALLOC_SIZE (param->conf->set_size),
		     param->seen_set);
	  if (pass == 1)
	    sched_yield ();
	}
    }
  return NULL;
}

static void
stop_and_join_threads (struct conf *conf, cpu_set_t *set,
		       pthread_t *pinned_first, pthread_t *pinned_last,
		       struct burn_thread *other_first,
		       struct burn_thread *other_last)
{
  __atomic_store_n (&still_running, 1, __ATOMIC_RELAXED);
  for (pthread_t *p = pinned_first; p < pinned_last; ++p)
    {
      int cpu = p - pinned_first;
      if (!CPU_ISSET_S (cpu, CPU_ALLOC_SIZE (conf->set_size), set))
	continue;

      int ret = pthread_join (*p, NULL);
      if (ret != 0)
	{
	  printf ("error: Failed to join thread %d: %s\n", cpu, strerror (ret));
	  fflush (stdout);
	  /* Cannot shut down cleanly with threads still running.  */
	  abort ();
	}
    }

  for (struct burn_thread *p = other_first; p < other_last; ++p)
    {
      int cpu = p - other_first;
      if (!CPU_ISSET_S (cpu, CPU_ALLOC_SIZE (conf->set_size), set))
	continue;

      int ret = pthread_join (p->self, NULL);
      if (ret != 0)
	{
	  printf ("error: Failed to join thread %d: %s\n", cpu, strerror (ret));
	  fflush (stdout);
	  /* Cannot shut down cleanly with threads still running.  */
	  abort ();
	}
    }
}

/* Tries to check that the initial set of CPUs is complete and that
   the main thread will not run on any other threads.  */
static bool
early_test (struct conf *conf)
{
  pthread_t *pinned_threads
    = calloc (conf->last_cpu + 1, sizeof (*pinned_threads));
  struct burn_thread *other_threads
    = calloc (conf->last_cpu + 1, sizeof (*other_threads));
  cpu_set_t *initial_set = CPU_ALLOC (conf->set_size);
  cpu_set_t *scratch_set = CPU_ALLOC (conf->set_size);

  if (pinned_threads == NULL || other_threads == NULL
      || initial_set == NULL || scratch_set == NULL)
    {
      puts ("error: Memory allocation failure");
      return false;
    }
  if (getaffinity (CPU_ALLOC_SIZE (conf->set_size), initial_set) < 0)
    {
      printf ("error: pthread_getaffinity_np failed: %m\n");
      return false;
    }
  for (int cpu = 0; cpu <= conf->last_cpu; ++cpu)
    {
      if (!CPU_ISSET_S (cpu, CPU_ALLOC_SIZE (conf->set_size), initial_set))
	continue;
      other_threads[cpu].conf = conf;
      other_threads[cpu].initial_set = initial_set;
      other_threads[cpu].thread = cpu;
      other_threads[cpu].seen_set = CPU_ALLOC (conf->set_size);
      if (other_threads[cpu].seen_set == NULL)
	{
	  puts ("error: Memory allocation failure");
	  return false;
	}
      CPU_ZERO_S (CPU_ALLOC_SIZE (conf->set_size),
		  other_threads[cpu].seen_set);
    }

  pthread_attr_t attr;
  int ret = pthread_attr_init (&attr);
  if (ret != 0)
    {
      printf ("error: pthread_attr_init failed: %s\n", strerror (ret));
      return false;
    }
  support_set_small_thread_stack_size (&attr);

  /* Spawn a thread pinned to each available CPU.  */
  for (int cpu = 0; cpu <= conf->last_cpu; ++cpu)
    {
      if (!CPU_ISSET_S (cpu, CPU_ALLOC_SIZE (conf->set_size), initial_set))
	continue;
      CPU_ZERO_S (CPU_ALLOC_SIZE (conf->set_size), scratch_set);
      CPU_SET_S (cpu, CPU_ALLOC_SIZE (conf->set_size), scratch_set);
      ret = pthread_attr_setaffinity_np
	(&attr, CPU_ALLOC_SIZE (conf->set_size), scratch_set);
      if (ret != 0)
	{
	  printf ("error: pthread_attr_setaffinity_np for CPU %d failed: %s\n",
		  cpu, strerror (ret));
	  stop_and_join_threads (conf, initial_set,
				 pinned_threads, pinned_threads + cpu,
				 NULL, NULL);
	  return false;
	}
      ret = pthread_create (pinned_threads + cpu, &attr,
			    thread_burn_one_cpu, (void *) (uintptr_t) cpu);
      if (ret != 0)
	{
	  printf ("error: pthread_create for CPU %d failed: %s\n",
		  cpu, strerror (ret));
	  stop_and_join_threads (conf, initial_set,
				 pinned_threads, pinned_threads + cpu,
				 NULL, NULL);
	  return false;
	}
    }

  /* Spawn another set of threads running on all CPUs.  */
  for (int cpu = 0; cpu <= conf->last_cpu; ++cpu)
    {
      if (!CPU_ISSET_S (cpu, CPU_ALLOC_SIZE (conf->set_size), initial_set))
	continue;
      ret = pthread_create (&other_threads[cpu].self,
			    support_small_stack_thread_attribute (),
			    thread_burn_any_cpu, other_threads + cpu);
      if (ret != 0)
	{
	  printf ("error: pthread_create for thread %d failed: %s\n",
		  cpu, strerror (ret));
	  stop_and_join_threads (conf, initial_set,
				 pinned_threads,
				 pinned_threads + conf->last_cpu + 1,
				 other_threads, other_threads + cpu);
	  return false;
	}
    }

  /* Main thread.  */
  struct burn_thread main_thread;
  main_thread.conf = conf;
  main_thread.initial_set = initial_set;
  main_thread.seen_set = scratch_set;
  main_thread.thread = -1;
  CPU_ZERO_S (CPU_ALLOC_SIZE (conf->set_size), main_thread.seen_set);
  thread_burn_any_cpu (&main_thread);
  stop_and_join_threads (conf, initial_set,
			 pinned_threads,
			 pinned_threads + conf->last_cpu + 1,
			 other_threads, other_threads + conf->last_cpu + 1);

  printf ("info: Main thread ran on %d CPU(s) of %d available CPU(s)\n",
	  CPU_COUNT_S (CPU_ALLOC_SIZE (conf->set_size), scratch_set),
	  CPU_COUNT_S (CPU_ALLOC_SIZE (conf->set_size), initial_set));
  CPU_ZERO_S (CPU_ALLOC_SIZE (conf->set_size), scratch_set);
  for (int cpu = 0; cpu <= conf->last_cpu; ++cpu)
    {
      if (!CPU_ISSET_S (cpu, CPU_ALLOC_SIZE (conf->set_size), initial_set))
	continue;
      CPU_OR_S (CPU_ALLOC_SIZE (conf->set_size),
		scratch_set, scratch_set, other_threads[cpu].seen_set);
      CPU_FREE (other_threads[cpu].seen_set);
    }
  printf ("info: Other threads ran on %d CPU(s)\n",
	  CPU_COUNT_S (CPU_ALLOC_SIZE (conf->set_size), scratch_set));;


  pthread_attr_destroy (&attr);
  CPU_FREE (scratch_set);
  CPU_FREE (initial_set);
  free (pinned_threads);
  free (other_threads);
  return failed == 0;
}
