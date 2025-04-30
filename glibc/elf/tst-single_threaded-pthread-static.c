/* Test support for single-thread optimizations.  With threads, static version.
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

/* This test is a stripped-down version of
   tst-single_threaded-pthread.c, without any loading of dynamic
   objects.  */

#include <stdio.h>
#include <support/check.h>
#include <support/xthread.h>
#include <sys/single_threaded.h>

/* First barrier synchronizes main thread, thread 1, thread 2.  */
static pthread_barrier_t barrier1;

/* Second barrier synchronizes main thread, thread 2.  */
static pthread_barrier_t barrier2;

static void *
threadfunc (void *closure)
{
  TEST_VERIFY (!__libc_single_threaded);

  /* Wait for the main thread and the other thread.  */
  xpthread_barrier_wait (&barrier1);
  TEST_VERIFY (!__libc_single_threaded);

  /* Second thread waits on second barrier, too.  */
  if (closure != NULL)
    xpthread_barrier_wait (&barrier2);
  TEST_VERIFY (!__libc_single_threaded);

  return NULL;
}

static int
do_test (void)
{
  TEST_VERIFY (__libc_single_threaded);

  /* Two threads plus main thread.  */
  xpthread_barrier_init (&barrier1, NULL, 3);

  /* Main thread and second thread.  */
  xpthread_barrier_init (&barrier2, NULL, 2);

  pthread_t thr1 = xpthread_create (NULL, threadfunc, NULL);
  TEST_VERIFY (!__libc_single_threaded);

  pthread_t thr2 = xpthread_create (NULL, threadfunc, &thr2);
  TEST_VERIFY (!__libc_single_threaded);

  xpthread_barrier_wait (&barrier1);
  TEST_VERIFY (!__libc_single_threaded);

  /* Join first thread.  This should not bring us back into
     single-threaded mode.  */
  xpthread_join (thr1);
  TEST_VERIFY (!__libc_single_threaded);

  /* We may be back in single-threaded mode after joining both
     threads, but this is not guaranteed.  */
  xpthread_barrier_wait (&barrier2);
  xpthread_join (thr2);
  printf ("info: __libc_single_threaded after joining all threads: %d\n",
          __libc_single_threaded);

  return 0;
}

#include <support/test-driver.c>
