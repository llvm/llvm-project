/* Copyright (C) 2015-2021 Free Software Foundation, Inc.
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

#include <errno.h>
#include <pthread.h>
#include <stdio.h>
#include <string.h>


static pthread_once_t once = PTHREAD_ONCE_INIT;

// Exception type thrown from the pthread_once init routine.
struct OnceException { };

// Test iteration counter.
static int niter;

static void
init_routine (void)
{
  if (niter < 2)
    throw OnceException ();
}

// Verify that an exception thrown from the pthread_once init routine
// is propagated to the pthread_once caller and that the function can
// be subsequently invoked to attempt the initialization again.
static int
do_test (void)
{
  int result = 1;

  // Repeat three times, having the init routine throw the first two
  // times and succeed on the final attempt.
  for (niter = 0; niter != 3; ++niter) {

    try {
      int rc = pthread_once (&once, init_routine);
      if (rc)
        fprintf (stderr, "pthread_once failed: %i (%s)\n",
                 rc, strerror (rc));

      if (niter < 2)
        fputs ("pthread_once unexpectedly returned without"
               " throwing an exception", stderr);
    }
    catch (OnceException) {
      if (niter > 1)
        fputs ("pthread_once unexpectedly threw", stderr);
      result = 0;
    }
    catch (...) {
      fputs ("pthread_once threw an unknown exception", stderr);
    }

    // Abort the test on the first failure.
    if (result)
      break;
  }

  return result;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
