/* Global test failure counter.
   Copyright (C) 2016-2021 Free Software Foundation, Inc.
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

#include <support/check.h>
#include <support/support.h>
#include <support/test-driver.h>

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <unistd.h>

/* This structure keeps track of test failures.  The counter is
   incremented on each failure.  The failed member is set to true if a
   failure is detected, so that even if the counter wraps around to
   zero, the failure of a test can be detected.

   The init constructor function below puts *state on a shared
   annonymous mapping, so that failure reports from subprocesses
   propagate to the parent process.  */
struct test_failures
{
  unsigned int counter;
  unsigned int failed;
};
static struct test_failures *state;

static __attribute__ ((constructor)) void
init (void)
{
  void *ptr = mmap (NULL, sizeof (*state), PROT_READ | PROT_WRITE,
                    MAP_ANONYMOUS | MAP_SHARED, -1, 0);
  if (ptr == MAP_FAILED)
    {
      printf ("error: could not map %zu bytes: %m\n", sizeof (*state));
      exit (1);
    }
  /* Zero-initialization of the struct is sufficient.  */
  state = ptr;
}

void
support_record_failure (void)
{
  if (state == NULL)
    {
      write_message
        ("error: support_record_failure called without initialization\n");
      _exit (1);
    }
  /* Relaxed MO is sufficient because we are only interested in the
     values themselves, in isolation.  */
  __atomic_store_n (&state->failed, 1, __ATOMIC_RELEASE);
  __atomic_add_fetch (&state->counter, 1, __ATOMIC_RELEASE);
}

int
support_report_failure (int status)
{
  if (state == NULL)
    {
      write_message
        ("error: support_report_failure called without initialization\n");
      return 1;
    }

  /* Relaxed MO is sufficient because acquire test result reporting
     assumes that exiting from the main thread happens before the
     error reporting via support_record_failure, which requires some
     form of external synchronization.  */
  bool failed = __atomic_load_n (&state->failed, __ATOMIC_RELAXED);
  if (failed)
    printf ("error: %u test failures\n",
            __atomic_load_n (&state->counter, __ATOMIC_RELAXED));

  if ((status == 0 || status == EXIT_UNSUPPORTED) && failed)
    /* If we have a recorded failure, it overrides a non-failure
       report from the test function.  */
    status = 1;
  return status;
}

void
support_record_failure_reset (void)
{
  /* Only used for testing the test framework, with external
     synchronization, but use release MO for consistency.  */
  __atomic_store_n (&state->failed, 0, __ATOMIC_RELAXED);
  __atomic_add_fetch (&state->counter, 0, __ATOMIC_RELAXED);
}

int
support_record_failure_is_failed (void)
{
  /* Relaxed MO is sufficient because we need (blocking) external
     synchronization for reliable test error reporting anyway.  */
  return __atomic_load_n (&state->failed, __ATOMIC_RELAXED);
}
