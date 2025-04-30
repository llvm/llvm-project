/* Bug 20116: Test rapid creation of detached threads.
   Copyright (C) 2017-2021 Free Software Foundation, Inc.
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
   License along with the GNU C Library; see the file COPYING.LIB.  If
   not, see <https://www.gnu.org/licenses/>.  */

/* The goal of the test is to trigger a failure if the parent touches
   any part of the thread descriptor after the detached thread has
   exited.  We test this by creating many detached threads with large
   stacks.  The stacks quickly fill the the stack cache and subsequent
   threads will start to cause the thread stacks to be immediately
   unmapped to satisfy the stack cache max.  With the stacks being
   unmapped the parent's read of any part of the thread descriptor will
   trigger a segfault.  That segfault is what we are trying to cause,
   since any segfault is a defect in the implementation.  */

#include <pthread.h>
#include <stdio.h>
#include <errno.h>
#include <unistd.h>
#include <stdbool.h>
#include <sys/resource.h>
#include <support/xthread.h>

/* Number of threads to create.  */
enum { threads_to_create = 100000 };

/* Number of threads which should spawn other threads.  */
enum { creator_threads  = 2 };

/* Counter of threads created so far.  This is incremented by all the
   running creator threads.  */
static unsigned threads_created;

/* Thread callback which does nothing, so that the thread exits
   immediatedly.  */
static void *
do_nothing (void *arg)
{
  return NULL;
}

/* Attribute indicating that the thread should be created in a detached
   fashion.  */
static pthread_attr_t detached;

/* Barrier to synchronize initialization.  */
static pthread_barrier_t barrier;

static void *
creator_thread (void *arg)
{
  int ret;
  xpthread_barrier_wait (&barrier);

  while (true)
    {
      pthread_t thr;
      /* Thread creation will fail if the kernel does not free old
	 threads quickly enough, so we do not report errors.  */
      ret = pthread_create (&thr, &detached, do_nothing, NULL);
      if (ret == 0 && __atomic_add_fetch (&threads_created, 1, __ATOMIC_SEQ_CST)
          >= threads_to_create)
        break;
    }

  return NULL;
}

static int
do_test (void)
{
  /* Limit the size of the process, so that memory allocation will
     fail without impacting the entire system.  */
  {
    struct rlimit limit;
    if (getrlimit (RLIMIT_AS, &limit) != 0)
      {
        printf ("FAIL: getrlimit (RLIMIT_AS) failed: %m\n");
        return 1;
      }
    /* This limit, 800MB, is just a heuristic. Any value can be
       picked.  */
    long target = 800 * 1024 * 1024;
    if (limit.rlim_cur == RLIM_INFINITY || limit.rlim_cur > target)
      {
        limit.rlim_cur = target;
        if (setrlimit (RLIMIT_AS, &limit) != 0)
          {
            printf ("FAIL: setrlimit (RLIMIT_AS) failed: %m\n");
            return 1;
          }
      }
  }

  xpthread_attr_init (&detached);

  xpthread_attr_setdetachstate (&detached, PTHREAD_CREATE_DETACHED);

  /* A large thread stack seems beneficial for reproducing a race
     condition in detached thread creation.  The goal is to reach the
     limit of the runtime thread stack cache such that the detached
     thread's stack is unmapped after exit and causes a segfault when
     the parent reads the thread descriptor data stored on the the
     unmapped stack.  */
  xpthread_attr_setstacksize (&detached, 16 * 1024 * 1024);

  xpthread_barrier_init (&barrier, NULL, creator_threads);

  pthread_t threads[creator_threads];

  for (int i = 0; i < creator_threads; ++i)
    threads[i] = xpthread_create (NULL, creator_thread, NULL);

  for (int i = 0; i < creator_threads; ++i)
    xpthread_join (threads[i]);

  xpthread_attr_destroy (&detached);

  xpthread_barrier_destroy (&barrier);

  return 0;
}

#define TIMEOUT 100
#include <support/test-driver.c>
