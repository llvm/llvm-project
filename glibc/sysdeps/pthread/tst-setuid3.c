/* Copyright (C) 2014-2021 Free Software Foundation, Inc.
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

#include <stdio.h>
#include <errno.h>
#include <pthread.h>
#include <stdbool.h>
#include <unistd.h>

/* The test must run under a non-privileged user ID.  */
static const uid_t test_uid = 1;

static pthread_barrier_t barrier1;
static pthread_barrier_t barrier2;

#define FAIL(fmt, ...) \
  do { printf ("FAIL: " fmt "\n", __VA_ARGS__); _exit (1); } while (0)

#define FAIL_ERR(fmt, ...) \
  do { printf ("FAIL: " fmt ": %m\n", __VA_ARGS__); _exit (1); } while (0)

/* True if x is not a successful return code from pthread_barrier_wait.  */
static inline bool
is_invalid_barrier_ret (int x)
{
  return x != 0 && x != PTHREAD_BARRIER_SERIAL_THREAD;
}

static void *
thread_func (void *ctx __attribute__ ((unused)))
{
  int ret = pthread_barrier_wait (&barrier1);
  if (is_invalid_barrier_ret (ret))
    FAIL ("pthread_barrier_wait (barrier1) (on thread): %d", ret);
  ret = pthread_barrier_wait (&barrier2);
  if (is_invalid_barrier_ret (ret))
    FAIL ("pthread_barrier_wait (barrier2) (on thread): %d", ret);
  return NULL;
}

static void
setuid_failure (int phase)
{
  int ret = setuid (0);
  switch (ret)
    {
    case 0:
      FAIL ("setuid succeeded unexpectedly in phase %d", phase);
    case -1:
      if (errno != EPERM)
	FAIL_ERR ("setuid phase %d", phase);
      break;
    default:
      FAIL ("invalid setuid return value in phase %d: %d", phase, ret);
    }
}

static int
do_test (void)
{
  if (getuid () == 0)
    if (setuid (test_uid) != 0)
      FAIL_ERR ("setuid (%u)", (unsigned) test_uid);
  if (setuid (getuid ()))
    FAIL_ERR ("setuid (%s)", "getuid ()");
  setuid_failure (1);

  int ret = pthread_barrier_init (&barrier1, NULL, 2);
  if (ret != 0)
    FAIL ("pthread_barrier_init (barrier1): %d", ret);
  ret = pthread_barrier_init (&barrier2, NULL, 2);
  if (ret != 0)
    FAIL ("pthread_barrier_init (barrier2): %d", ret);

  pthread_t thread;
  ret = pthread_create (&thread, NULL, thread_func, NULL);
  if (ret != 0)
    FAIL ("pthread_create: %d", ret);

  /* Ensure that the thread is running properly.  */
  ret = pthread_barrier_wait (&barrier1);
  if (is_invalid_barrier_ret (ret))
    FAIL ("pthread_barrier_wait (barrier1): %d", ret);

  setuid_failure (2);

  /* Check success case. */
  if (setuid (getuid ()) != 0)
    FAIL_ERR ("setuid (%s)", "getuid ()");

  /* Shutdown.  */
  ret = pthread_barrier_wait (&barrier2);
  if (is_invalid_barrier_ret (ret))
    FAIL ("pthread_barrier_wait (barrier2): %d", ret);

  ret = pthread_join (thread, NULL);
  if (ret != 0)
    FAIL ("pthread_join: %d", ret);

  return 0;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
