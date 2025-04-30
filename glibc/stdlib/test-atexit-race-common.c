/* Bug 14333: Support file for atexit/exit, etc. race tests.
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
   License along with the GNU C Library; if not, see
   <https://www.gnu.org/licenses/>.  */

/* This file must be run from within a directory called "stdlib".  */

/* The atexit/exit, at_quick_exit/quick_exit, __cxa_atexit/exit, etc.
   exhibited data race while accessing destructor function list (Bug 14333).

   This test spawns large number of threads, which all race to register
   large number of destructors.

   Before the fix, running this test resulted in a SIGSEGV.
   After the fix, we expect clean process termination.  */

#if !defined(CALL_EXIT) || !defined(CALL_ATEXIT)
#error Must define CALL_EXIT and CALL_ATEXIT before using this file.
#endif

#include <stdio.h>
#include <stdlib.h>
#include <support/xthread.h>
#include <limits.h>

const size_t kNumThreads = 1024;
const size_t kNumHandlers = 1024;

static void *
threadfunc (void *unused)
{
  size_t i;
  for (i = 0; i < kNumHandlers; ++i) {
    CALL_ATEXIT;
  }
  return NULL;
}

static int
do_test (void)
{
  size_t i;
  pthread_attr_t attr;

  xpthread_attr_init (&attr);
  xpthread_attr_setdetachstate (&attr, 1);

  /* With default 8MiB Linux stack size, creating 1024 threads can cause
     VM exhausiton on 32-bit machines.  Reduce stack size of each thread to
     128KiB for a maximum required VM size of 128MiB.  */
  size_t kStacksize =
#ifdef PTHREAD_STACK_MIN
    0x20000 < PTHREAD_STACK_MIN ? PTHREAD_STACK_MIN :
#endif
    0x20000;

  xpthread_attr_setstacksize (&attr, kStacksize);

  for (i = 0; i < kNumThreads; ++i) {
    xpthread_create (&attr, threadfunc, NULL);
  }
  xpthread_attr_destroy (&attr);

  CALL_EXIT;
}

#define TEST_FUNCTION do_test
#include <support/test-driver.c>
