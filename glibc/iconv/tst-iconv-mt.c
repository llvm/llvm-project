/* Test that iconv works in a multi-threaded program.
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

/* This test runs several worker threads that perform the following three
   steps in staggered synchronization with each other:
   1. initialization (iconv_open)
   2. conversion (iconv)
   3. cleanup (iconv_close)
   Having several threads synchronously (using pthread_barrier_*) perform
   these routines exercises iconv code that handles concurrent attempts to
   initialize and/or read global iconv state (namely configuration).  */

#include <iconv.h>
#include <stdio.h>
#include <string.h>

#include <support/support.h>
#include <support/xthread.h>
#include <support/check.h>

#define TCOUNT 16
_Static_assert (TCOUNT %2 == 0,
                "thread count must be even, since we need two groups.");


#define CONV_INPUT "Hello, iconv!"


pthread_barrier_t sync;
pthread_barrier_t sync_half_pool;


/* Execute iconv_open, iconv and iconv_close in a synchronized way in
   conjunction with other sibling worker threads.  If any step fails, print
   an error to stdout and return NULL to the main thread to indicate the
   error.  */
static void *
worker (void * arg)
{
  long int tidx = (long int) arg;

  iconv_t cd;

  char ascii[] = CONV_INPUT;
  char *inbufpos = ascii;
  size_t inbytesleft = sizeof (CONV_INPUT);

  char *utf8 = xcalloc (sizeof (CONV_INPUT), 1);
  char *outbufpos = utf8;
  size_t outbytesleft = sizeof (CONV_INPUT);

  if (tidx < TCOUNT/2)
    /* The first half of the worker thread pool synchronize together here,
       then call iconv_open immediately after.  */
    xpthread_barrier_wait (&sync_half_pool);
  else
    /* The second half wait for the first half to finish iconv_open and
       continue to the next barrier (before the call to iconv below).  */
    xpthread_barrier_wait (&sync);

  /* The above block of code staggers all subsequent pthread_barrier_wait
     calls in a way that ensures a high chance of encountering these
     combinations of concurrent iconv usage:
     1) concurrent calls to iconv_open,
     2) concurrent calls to iconv_open *and* iconv,
     3) concurrent calls to iconv,
     4) concurrent calls to iconv *and* iconv_close,
     5) concurrent calls to iconv_close.  */

  cd = iconv_open ("UTF8", "ASCII");
  TEST_VERIFY_EXIT (cd != (iconv_t) -1);

  xpthread_barrier_wait (&sync);

  TEST_VERIFY_EXIT (iconv (cd, &inbufpos, &inbytesleft, &outbufpos,
                           &outbytesleft)
                    != (size_t) -1);

  *outbufpos = '\0';

  xpthread_barrier_wait (&sync);

  TEST_VERIFY_EXIT (iconv_close (cd) == 0);

  /* The next conditional barrier wait is needed because we staggered the
     threads into two groups in the beginning and at this point, the second
     half of worker threads are waiting for the first half to finish
     iconv_close before they can executing the same:  */
  if (tidx < TCOUNT/2)
    xpthread_barrier_wait (&sync);

  if (strncmp (utf8, CONV_INPUT, sizeof CONV_INPUT))
    {
      printf ("FAIL: thread %lx: invalid conversion output from iconv\n", tidx);
      pthread_exit ((void *) (long int) 1);
    }

  pthread_exit (NULL);
}


static int
do_test (void)
{
  pthread_t thread[TCOUNT];
  void * worker_output;
  int i;

  xpthread_barrier_init (&sync, NULL, TCOUNT);
  xpthread_barrier_init (&sync_half_pool, NULL, TCOUNT/2);

  for (i = 0; i < TCOUNT; i++)
    thread[i] = xpthread_create (NULL, worker, (void *) (long int) i);

  for (i = 0; i < TCOUNT; i++)
    {
      worker_output = xpthread_join (thread[i]);
      TEST_VERIFY_EXIT (worker_output == NULL);
    }

  xpthread_barrier_destroy (&sync);
  xpthread_barrier_destroy (&sync_half_pool);

  return 0;
}

#include <support/test-driver.c>
