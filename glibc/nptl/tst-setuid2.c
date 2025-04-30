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

#include <errno.h>
#include <pthread.h>
#include <signal.h>
#include <stdbool.h>
#include <stdio.h>
#include <sys/syscall.h>
#include <unistd.h>

/* Check that a partial setuid failure aborts the process. */

static pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t cond_send;
static void (*func_sent) (void);
static pthread_cond_t cond_recv;

#define FAIL(fmt, ...) \
  do { printf ("FAIL: " fmt "\n", __VA_ARGS__); _exit (1); } while (0)

static void *
thread_func (void *ctx __attribute__ ((unused)))
{
  int ret = pthread_mutex_lock (&mutex);
  if (ret != 0)
    FAIL ("pthread_mutex_lock (thread): %d", ret);

  while (true)
    {
      if (func_sent != NULL)
	{
	  void (*func) (void) = func_sent;
	  ret = pthread_mutex_unlock (&mutex);
	  if (ret != 0)
	    FAIL ("pthread_mutex_unlock (thread): %d", ret);
	  func ();
	  ret = pthread_mutex_lock (&mutex);
	  if (ret != 0)
	    FAIL ("pthread_mutex_lock (thread): %d", ret);
	  func_sent = NULL;
	  ret = pthread_cond_signal (&cond_recv);
	  if (ret != 0)
	    FAIL ("pthread_cond_signal (recv): %d", ret);
	}
      ret = pthread_cond_wait (&cond_send, &mutex);
      if (ret != 0)
	FAIL ("pthread_cond_wait (send): %d", ret);
    }
  return NULL;
}

static void
run_on_thread (void (*func) (void))
{
  int ret = pthread_mutex_lock (&mutex);
  if (ret != 0)
    FAIL ("pthread_mutex_lock (%s): %d", __func__, ret);
  func_sent = func;
  ret = pthread_mutex_unlock (&mutex);
  if (ret != 0)
    FAIL ("pthread_mutex_unlock (%s): %d", __func__, ret);

  ret = pthread_cond_signal (&cond_send);
  if (ret != 0)
    FAIL ("pthread_mutex_lock (%s): %d", __func__, ret);

  ret = pthread_mutex_lock (&mutex);
  if (ret != 0)
    FAIL ("pthread_mutex_lock (%s): %d", __func__, ret);

  while (func_sent != NULL)
    {
      ret = pthread_cond_wait (&cond_recv, &mutex);
      if (ret != 0)
	FAIL ("pthread_mutex_wait (%s): %d", __func__, ret);
    }
  ret = pthread_mutex_unlock (&mutex);
  if (ret != 0)
    FAIL ("pthread_mutex_unlock (%s): %d", __func__, ret);
}

static void
change_thread_ids (void)
{
  long ret = syscall (__NR_setresuid, 2001, 2002, 2003);
  if (ret != 0)
    FAIL ("setresuid (2001, 2002, 2003): %ld", ret);
}

static uid_t ruid, euid, suid;

static void
get_thread_ids (void)
{
  if (getresuid (&ruid, &euid, &suid) < 0)
    FAIL ("getresuid: %m (%d)", errno);
}

static void
abort_expected (int signal __attribute__ ((unused)))
{
  _exit (0);
}

static int
do_test (void)
{
  pthread_t thread;
  int ret = pthread_create (&thread, NULL, thread_func, NULL);
  if (ret != 0)
    FAIL ("pthread_create: %d", ret);

  run_on_thread (change_thread_ids);

  signal (SIGABRT, &abort_expected);
  /* This should abort the process.  */
  if (setresuid (1001, 1002, 1003) < 0)
    FAIL ("setresuid: %m (%d)", errno);
  signal (SIGABRT, SIG_DFL);

  /* If we get here, check that the kernel did the right thing. */
  run_on_thread (get_thread_ids);
  if (ruid != 1001 || euid != 1002 || suid != 1003)
    FAIL ("unexpected UIDs after setuid: %ld, %ld, %ld",
	  (long) ruid, (long) euid, (long) suid);
  return 0;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
