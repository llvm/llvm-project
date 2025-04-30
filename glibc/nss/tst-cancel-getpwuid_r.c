/* Test cancellation of getpwuid_r.
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

/* Test if cancellation of getpwuid_r incorrectly leaves internal
   function state locked resulting in hang of subsequent calls to
   getpwuid_r.  The main thread creates a second thread which will do
   the calls to getpwuid_r.  A semaphore is used by the second thread to
   signal to the main thread that it is as close as it can be to the
   call site of getpwuid_r.  The goal of the semaphore is to avoid any
   cancellable function calls between the sem_post and the call to
   getpwuid_r.  The main thread then attempts to cancel the second
   thread.  Without the fixes the cancellation happens at any number of
   calls to cancellable functions in getpuid_r, but with the fix the
   cancellation either does not happen or happens only at expected
   points where the internal state is consistent.  We use an explicit
   pthread_testcancel call to terminate the loop in a timely fashion
   if the implementation does not have a cancellation point.  */

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <pwd.h>
#include <nss.h>
#include <sys/types.h>
#include <unistd.h>
#include <semaphore.h>
#include <errno.h>
#include <support/support.h>

sem_t started;
char *wbuf;
long wbufsz;

void
worker_free (void *arg)
{
  free (arg);
}

static void *
worker (void *arg)
{
  int ret;
  unsigned int iter = 0;
  struct passwd pwbuf, *pw;
  uid_t uid;

  uid = geteuid ();

  /* Use a reasonable sized buffer.  Note that _SC_GETPW_R_SIZE_MAX is
     just a hint and not any kind of maximum value.  */
  wbufsz = sysconf (_SC_GETPW_R_SIZE_MAX);
  if (wbufsz == -1)
    wbufsz = 1024;
  wbuf = xmalloc (wbufsz);

  pthread_cleanup_push (worker_free, wbuf);
  sem_post (&started);
  while (1)
    {
      iter++;

      ret = getpwuid_r (uid, &pwbuf, wbuf, wbufsz, &pw);

      /* The call to getpwuid_r may not cancel so we need to test
	 for cancellation after some number of iterations of the
	 function.  Choose an arbitrary 100,000 iterations of running
	 getpwuid_r in a tight cancellation loop before testing for
	 cancellation.  */
      if (iter > 100000)
	pthread_testcancel ();

      if (ret == ERANGE)
	{
	  /* Increase the buffer size.  */
	  free (wbuf);
	  wbufsz = wbufsz * 2;
	  wbuf = xmalloc (wbufsz);
	}

    }
  pthread_cleanup_pop (1);

  return NULL;
}

static int
do_test (void)
{
  int ret;
  char *buf;
  long bufsz;
  void *retval;
  struct passwd pwbuf, *pw;
  pthread_t thread;

  /* Configure the test to only use files. We control the files plugin
     as part of glibc so we assert that it should be deferred
     cancellation safe.  */
  __nss_configure_lookup ("passwd", "files");

  /* Use a reasonable sized buffer.  Note that  _SC_GETPW_R_SIZE_MAX is
     just a hint and not any kind of maximum value.  */
  bufsz = sysconf (_SC_GETPW_R_SIZE_MAX);
  if (bufsz == -1)
    bufsz = 1024;
  buf = xmalloc (bufsz);

  sem_init (&started, 0, 0);

  pthread_create (&thread, NULL, worker, NULL);

  do
  {
    ret = sem_wait (&started);
    if (ret == -1 && errno != EINTR)
      {
        printf ("FAIL: Failed to wait for second thread to start.\n");
	exit (EXIT_FAILURE);
      }
  }
  while (ret != 0);

  printf ("INFO: Cancelling thread\n");
  if ((ret = pthread_cancel (thread)) != 0)
    {
      printf ("FAIL: Failed to cancel thread. Returned %d\n", ret);
      exit (EXIT_FAILURE);
    }

  printf ("INFO: Joining...\n");
  pthread_join (thread, &retval);
  if (retval != PTHREAD_CANCELED)
    {
      printf ("FAIL: Thread was not cancelled.\n");
      exit (EXIT_FAILURE);
    }
  printf ("INFO: Joined, trying getpwuid_r call\n");

  /* Before the fix in 312be3f9f5eab1643d7dcc7728c76d413d4f2640 for this
     issue the cancellation point could happen in any number of internal
     calls, and therefore locks would be left held and the following
     call to getpwuid_r would block and the test would time out.  */
  do
    {
      ret = getpwuid_r (geteuid (), &pwbuf, buf, bufsz, &pw);
      if (ret == ERANGE)
	{
	  /* Increase the buffer size.  */
	  free (buf);
	  bufsz = bufsz * 2;
	  buf = xmalloc (bufsz);
	}
    }
  while (ret == ERANGE);

  free (buf);

  /* Before the fix we would never get here.  */
  printf ("PASS: Canceled getpwuid_r successfully"
	  " and called it again without blocking.\n");

  return 0;
}

#define TIMEOUT 900
#include <support/test-driver.c>
