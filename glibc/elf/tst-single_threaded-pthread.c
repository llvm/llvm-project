/* Test support for single-thread optimizations.  With threads.
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

#include <stddef.h>
#include <stdio.h>
#include <support/check.h>
#include <support/namespace.h>
#include <support/xdlfcn.h>
#include <support/xthread.h>
#include <sys/single_threaded.h>

/* First barrier synchronizes main thread, thread 1, thread 2.  */
static pthread_barrier_t barrier1;

/* Second barrier synchronizes main thread, thread 2.  */
static pthread_barrier_t barrier2;

/* Defined in tst-single-threaded-mod1.so.  */
_Bool single_threaded_1 (void);

/* Initialized via dlsym.  */
static _Bool (*single_threaded_2) (void);
static _Bool (*single_threaded_3) (void);
static _Bool (*single_threaded_4) (void);

static void *
threadfunc (void *closure)
{
  TEST_VERIFY (!__libc_single_threaded);
  TEST_VERIFY (!single_threaded_1 ());
  TEST_VERIFY (!single_threaded_2 ());

  /* Wait until the main thread loads more functions.  */
  xpthread_barrier_wait (&barrier1);

  TEST_VERIFY (!__libc_single_threaded);
  TEST_VERIFY (!single_threaded_1 ());
  TEST_VERIFY (!single_threaded_2 ());
  TEST_VERIFY (!single_threaded_3 ());
  TEST_VERIFY (!single_threaded_4 ());

  /* Second thread waits on second barrier, too.  */
  if (closure != NULL)
    xpthread_barrier_wait (&barrier2);
  TEST_VERIFY (!__libc_single_threaded);
  TEST_VERIFY (!single_threaded_1 ());
  TEST_VERIFY (!single_threaded_2 ());
  TEST_VERIFY (!single_threaded_3 ());
  TEST_VERIFY (!single_threaded_4 ());

  return NULL;
}

/* Used for closure arguments to the subprocess function.  */
static char expected_false = 0;
static char expected_true = 1;

/* A subprocess inherits currently inherits the single-threaded state
   of the parent process.  */
static void
subprocess (void *closure)
{
  const char *expected = closure;
  TEST_COMPARE (__libc_single_threaded, *expected);
  TEST_COMPARE (single_threaded_1 (), *expected);
  if (single_threaded_2 != NULL)
    TEST_COMPARE (single_threaded_2 (), *expected);
  if (single_threaded_3 != NULL)
    TEST_COMPARE (single_threaded_3 (), *expected);
  if (single_threaded_4 != NULL)
    TEST_VERIFY (!single_threaded_4 ());
}

static int
do_test (void)
{
  printf ("info: main __libc_single_threaded address: %p\n",
          &__libc_single_threaded);
  TEST_VERIFY (__libc_single_threaded);
  TEST_VERIFY (single_threaded_1 ());
  support_isolate_in_subprocess (subprocess, &expected_true);

  void *handle_mod2 = xdlopen ("tst-single_threaded-mod2.so", RTLD_LAZY);
  single_threaded_2 = xdlsym (handle_mod2, "single_threaded_2");
  TEST_VERIFY (single_threaded_2 ());

  /* Two threads plus main thread.  */
  xpthread_barrier_init (&barrier1, NULL, 3);

  /* Main thread and second thread.  */
  xpthread_barrier_init (&barrier2, NULL, 2);

  pthread_t thr1 = xpthread_create (NULL, threadfunc, NULL);
  TEST_VERIFY (!__libc_single_threaded);
  TEST_VERIFY (!single_threaded_1 ());
  TEST_VERIFY (!single_threaded_2 ());
  support_isolate_in_subprocess (subprocess, &expected_false);

  pthread_t thr2 = xpthread_create (NULL, threadfunc, &thr2);
  TEST_VERIFY (!__libc_single_threaded);
  TEST_VERIFY (!single_threaded_1 ());
  TEST_VERIFY (!single_threaded_2 ());
  support_isolate_in_subprocess (subprocess, &expected_false);

  /* Delayed library load, while already multi-threaded.  */
  void *handle_mod3 = xdlopen ("tst-single_threaded-mod3.so", RTLD_LAZY);
  single_threaded_3 = xdlsym (handle_mod3, "single_threaded_3");
  TEST_VERIFY (!__libc_single_threaded);
  TEST_VERIFY (!single_threaded_1 ());
  TEST_VERIFY (!single_threaded_2 ());
  TEST_VERIFY (!single_threaded_3 ());
  support_isolate_in_subprocess (subprocess, &expected_false);

  /* Same with dlmopen.  */
  void *handle_mod4 = dlmopen (LM_ID_NEWLM, "tst-single_threaded-mod4.so",
                               RTLD_LAZY);
  single_threaded_4 = xdlsym (handle_mod4, "single_threaded_4");
  TEST_VERIFY (!__libc_single_threaded);
  TEST_VERIFY (!single_threaded_1 ());
  TEST_VERIFY (!single_threaded_2 ());
  TEST_VERIFY (!single_threaded_3 ());
  TEST_VERIFY (!single_threaded_4 ());
  support_isolate_in_subprocess (subprocess, &expected_false);

  /* Run the newly loaded functions from the other threads as
     well.  */
  xpthread_barrier_wait (&barrier1);
  TEST_VERIFY (!__libc_single_threaded);
  TEST_VERIFY (!single_threaded_1 ());
  TEST_VERIFY (!single_threaded_2 ());
  TEST_VERIFY (!single_threaded_3 ());
  TEST_VERIFY (!single_threaded_4 ());
  support_isolate_in_subprocess (subprocess, &expected_false);

  /* Join first thread.  This should not bring us back into
     single-threaded mode.  */
  xpthread_join (thr1);
  TEST_VERIFY (!__libc_single_threaded);
  TEST_VERIFY (!single_threaded_1 ());
  TEST_VERIFY (!single_threaded_2 ());
  TEST_VERIFY (!single_threaded_3 ());
  TEST_VERIFY (!single_threaded_4 ());
  support_isolate_in_subprocess (subprocess, &expected_false);

  /* We may be back in single-threaded mode after joining both
     threads, but this is not guaranteed.  */
  xpthread_barrier_wait (&barrier2);
  xpthread_join (thr2);
  printf ("info: __libc_single_threaded after joining all threads: %d\n",
          __libc_single_threaded);

  xdlclose (handle_mod4);
  xdlclose (handle_mod3);
  xdlclose (handle_mod2);

  return 0;
}

#include <support/test-driver.c>
