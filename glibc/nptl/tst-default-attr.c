/* Verify that pthread_[gs]etattr_default_np work correctly.

   Copyright (C) 2013-2021 Free Software Foundation, Inc.
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
#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <stdbool.h>

#define RETURN_IF_FAIL(f, ...) \
  ({									      \
    int ret = f (__VA_ARGS__);						      \
    if (ret != 0)							      \
      {									      \
	printf ("%s:%d: %s returned %d (errno = %d)\n", __FILE__, __LINE__,   \
		#f, ret, errno);					      \
	return ret;							      \
      }									      \
  })

static int (*verify_result) (pthread_attr_t *);
static size_t stacksize = 1024 * 1024;
static size_t guardsize;
static bool do_join = true;
static int running = 0;
static int detach_failed = 0;
static pthread_mutex_t m = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t c = PTHREAD_COND_INITIALIZER;

static void *
thr (void *unused __attribute__ ((unused)))
{
  pthread_attr_t attr;
  int ret;

  memset (&attr, 0xab, sizeof attr);
  /* To verify that the pthread_setattr_default_np worked.  */
  if ((ret = pthread_getattr_default_np (&attr)) != 0)
    {
      printf ("pthread_getattr_default_np failed: %s\n", strerror (ret));
      goto out;
    }

  if ((ret = (*verify_result) (&attr)) != 0)
    goto out;

  memset (&attr, 0xab, sizeof attr);
  /* To verify that the attributes actually got applied.  */
  if ((ret = pthread_getattr_np (pthread_self (), &attr)) != 0)
    {
      printf ("pthread_getattr_default_np failed: %s\n", strerror (ret));
      goto out;
    }

  ret = (*verify_result) (&attr);

out:
  if (!do_join)
    {
      pthread_mutex_lock (&m);
      running--;
      pthread_cond_signal (&c);
      pthread_mutex_unlock (&m);

      detach_failed |= ret;
    }

  return (void *) (uintptr_t) ret;
}

static int
run_threads (const pthread_attr_t *attr)
{
  pthread_t t;
  void *tret = NULL;

  RETURN_IF_FAIL (pthread_setattr_default_np, attr);

  /* Run twice to ensure that the attributes do not get overwritten in the
     first run somehow.  */
  for (int i = 0; i < 2; i++)
    {
      RETURN_IF_FAIL (pthread_create, &t, NULL, thr, NULL);
      if (do_join)
	RETURN_IF_FAIL (pthread_join, t, &tret);
      else
	{
	  pthread_mutex_lock (&m);
	  running++;
	  pthread_mutex_unlock (&m);
	}

      if (tret != NULL)
	{
	  puts ("Thread failed");
	  return 1;
	}
    }

  /* Stay in sync for detached threads and get their status.  */
  while (!do_join)
    {
      pthread_mutex_lock (&m);
      if (running == 0)
	{
	  pthread_mutex_unlock (&m);
	  break;
	}
      pthread_cond_wait (&c, &m);
      pthread_mutex_unlock (&m);
    }

  return 0;
}

static int
verify_detach_result (pthread_attr_t *attr)
{
  int state;

  RETURN_IF_FAIL (pthread_attr_getdetachstate, attr, &state);

  if (state != PTHREAD_CREATE_DETACHED)
    {
      puts ("failed to set detach state");
      return 1;
    }

  return 0;
}

static int
do_detach_test (void)
{
  pthread_attr_t attr;

  do_join = false;
  RETURN_IF_FAIL (pthread_attr_init, &attr);
  RETURN_IF_FAIL (pthread_attr_setdetachstate, &attr, PTHREAD_CREATE_DETACHED);

  RETURN_IF_FAIL (run_threads, &attr);
  return detach_failed;
}

static int
verify_affinity_result (pthread_attr_t *attr)
{
  cpu_set_t cpuset;

  RETURN_IF_FAIL (pthread_attr_getaffinity_np, attr, sizeof (cpuset), &cpuset);
  if (!CPU_ISSET (0, &cpuset))
    {
      puts ("failed to set cpu affinity");
      return 1;
    }

  return 0;
}

static int
do_affinity_test (void)
{
  pthread_attr_t attr;

  RETURN_IF_FAIL (pthread_attr_init, &attr);

  /* Processor affinity.  Like scheduling policy, this could fail if the user
     does not have the necessary privileges.  So we only spew a warning if
     pthread_create fails with EPERM.  A computer has at least one CPU.  */
  cpu_set_t cpuset;
  CPU_ZERO (&cpuset);
  CPU_SET (0, &cpuset);
  RETURN_IF_FAIL (pthread_attr_setaffinity_np, &attr, sizeof (cpuset), &cpuset);

  int ret = run_threads (&attr);

  if (ret == EPERM)
    {
      printf ("Skipping CPU Affinity test: %s\n", strerror (ret));
      return 0;
    }
  else if (ret != 0)
    return ret;

  return 0;
}

static int
verify_sched_result (pthread_attr_t *attr)
{
  int inherited, policy;
  struct sched_param param;

  RETURN_IF_FAIL (pthread_attr_getinheritsched, attr, &inherited);
  if (inherited != PTHREAD_EXPLICIT_SCHED)
    {
      puts ("failed to set EXPLICIT_SCHED (%d != %d)");
      return 1;
    }

  RETURN_IF_FAIL (pthread_attr_getschedpolicy, attr, &policy);
  if (policy != SCHED_RR)
    {
      printf ("failed to set SCHED_RR (%d != %d)\n", policy, SCHED_RR);
      return 1;
    }

  RETURN_IF_FAIL (pthread_attr_getschedparam, attr, &param);
  if (param.sched_priority != 42)
    {
      printf ("failed to set sched_priority (%d != %d)\n",
	      param.sched_priority, 42);
      return 1;
    }

  return 0;
}

static int
do_sched_test (void)
{
  pthread_attr_t attr;

  RETURN_IF_FAIL (pthread_attr_init, &attr);

  /* Scheduling policy.  Note that we don't always test these since it's
     possible that the user the tests run as don't have the appropriate
     privileges.  */
  RETURN_IF_FAIL (pthread_attr_setinheritsched, &attr, PTHREAD_EXPLICIT_SCHED);
  RETURN_IF_FAIL (pthread_attr_setschedpolicy, &attr, SCHED_RR);

  struct sched_param param;
  param.sched_priority = 42;
  RETURN_IF_FAIL (pthread_attr_setschedparam, &attr, &param);

  int ret = run_threads (&attr);

  if (ret == EPERM)
    {
      printf ("Skipping Scheduler Attributes test: %s\n", strerror (ret));
      return 0;
    }
  else if (ret != 0)
    return ret;

  return 0;
}

static int
verify_guardsize_result (pthread_attr_t *attr)
{
  size_t guard;

  RETURN_IF_FAIL (pthread_attr_getguardsize, attr, &guard);

  if (guardsize != guard)
    {
      printf ("failed to set guardsize (%zu, %zu)\n", guardsize, guard);
      return 1;
    }

  return 0;
}

static int
do_guardsize_test (void)
{
  long int pagesize = sysconf (_SC_PAGESIZE);
  pthread_attr_t attr;

  if (pagesize < 0)
    {
      printf ("sysconf failed: %s\n", strerror (errno));
      return 1;
    }

  RETURN_IF_FAIL (pthread_getattr_default_np, &attr);

  /* Increase default guardsize by a page.  */
  RETURN_IF_FAIL (pthread_attr_getguardsize, &attr, &guardsize);
  guardsize += pagesize;
  RETURN_IF_FAIL (pthread_attr_setguardsize, &attr, guardsize);
  RETURN_IF_FAIL (run_threads, &attr);

  return 0;
}

static int
verify_stacksize_result (pthread_attr_t *attr)
{
  size_t stack;

  RETURN_IF_FAIL (pthread_attr_getstacksize, attr, &stack);

  if (stacksize != stack)
    {
      printf ("failed to set default stacksize (%zu, %zu)\n", stacksize, stack);
      return 1;
    }

  return 0;
}

static int
do_stacksize_test (void)
{
  long int pagesize = sysconf (_SC_PAGESIZE);
  pthread_attr_t attr;

  if (pagesize < 0)
    {
      printf ("sysconf failed: %s\n", strerror (errno));
      return 1;
    }

  /* Perturb the size by a page so that we're not aligned on the 64K boundary.
     pthread_create does this perturbation on x86 to avoid causing the 64k
     aliasing conflict.  We want to prevent pthread_create from doing that
     since it is not consistent for all architectures.  */
  stacksize += pagesize;

  RETURN_IF_FAIL (pthread_attr_init, &attr);

  /* Run twice to ensure that we don't give a false positive.  */
  RETURN_IF_FAIL (pthread_attr_setstacksize, &attr, stacksize);
  RETURN_IF_FAIL (run_threads, &attr);
  stacksize *= 2;
  RETURN_IF_FAIL (pthread_attr_setstacksize, &attr, stacksize);
  RETURN_IF_FAIL (run_threads, &attr);
  return 0;
}

/* We test each attribute separately because sched and affinity tests may need
   additional user privileges that may not be available during the test run.
   Each attribute test is a set of two functions, viz. a function to set the
   default attribute (do_foo_test) and another to verify its result
   (verify_foo_result).  Each test spawns a thread and checks (1) if the
   attribute values were applied correctly and (2) if the change in the default
   value reflected.  */
static int
do_test (void)
{
  puts ("stacksize test");
  verify_result = verify_stacksize_result;
  RETURN_IF_FAIL (do_stacksize_test);

  puts ("guardsize test");
  verify_result = verify_guardsize_result;
  RETURN_IF_FAIL (do_guardsize_test);

  puts ("sched test");
  verify_result = verify_sched_result;
  RETURN_IF_FAIL (do_sched_test);

  puts ("affinity test");
  verify_result = verify_affinity_result;
  RETURN_IF_FAIL (do_affinity_test);

  puts ("detach test");
  verify_result = verify_detach_result;
  RETURN_IF_FAIL (do_detach_test);

  return 0;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
