/* Copyright (C) 2003-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Jakub Jelinek <jakub@redhat.com>, 2003.

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
#include <error.h>
#include <fcntl.h>
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/select.h>
#include <sys/time.h>
#include <unistd.h>

static void *
tf (void *arg)
{
  return NULL;
}

static void
handler (int sig)
{
}

static void __attribute__ ((noinline))
clobber_lots_of_regs (void)
{
#define X1(n) long r##n = 10##n; __asm __volatile ("" : "+r" (r##n));
#define X2(n) X1(n##0) X1(n##1) X1(n##2) X1(n##3) X1(n##4)
#define X3(n) X2(n##0) X2(n##1) X2(n##2) X2(n##3) X2(n##4)
  X3(0) X3(1) X3(2) X3(3) X3(4)
#undef X1
#define X1(n) __asm __volatile ("" : : "r" (r##n));
  X3(0) X3(1) X3(2) X3(3) X3(4)
#undef X1
#undef X2
#undef X3
}

static int
do_test (void)
{
  pthread_t th;
  int old, rc;
  int ret = 0;
  int fd[2];

  rc = pipe (fd);
  if (rc < 0)
    error (EXIT_FAILURE, errno, "couldn't create pipe");

  rc = pthread_create (&th, NULL, tf, NULL);
  if (rc)
    error (EXIT_FAILURE, rc, "couldn't create thread");

  rc = pthread_setcanceltype (PTHREAD_CANCEL_DEFERRED, &old);
  if (rc)
    {
      error (0, rc, "1st pthread_setcanceltype failed");
      ret = 1;
    }
  if (old != PTHREAD_CANCEL_DEFERRED && old != PTHREAD_CANCEL_ASYNCHRONOUS)
    {
      error (0, 0, "1st pthread_setcanceltype returned invalid value %d",
	     old);
      ret = 1;
    }

  clobber_lots_of_regs ();
  close (fd[0]);

  rc = pthread_setcanceltype (PTHREAD_CANCEL_ASYNCHRONOUS, &old);
  if (rc)
    {
      error (0, rc, "pthread_setcanceltype after close failed");
      ret = 1;
    }
  if (old != PTHREAD_CANCEL_DEFERRED)
    {
      error (0, 0, "pthread_setcanceltype after close returned invalid value %d",
	     old);
      ret = 1;
    }

  clobber_lots_of_regs ();
  close (fd[1]);

  rc = pthread_setcanceltype (PTHREAD_CANCEL_DEFERRED, &old);
  if (rc)
    {
      error (0, rc, "pthread_setcanceltype after 2nd close failed");
      ret = 1;
    }
  if (old != PTHREAD_CANCEL_ASYNCHRONOUS)
    {
      error (0, 0, "pthread_setcanceltype after 2nd close returned invalid value %d",
	     old);
      ret = 1;
    }

  struct sigaction sa = { .sa_handler = handler, .sa_flags = 0 };
  sigemptyset (&sa.sa_mask);
  sigaction (SIGALRM, &sa, NULL);

  struct itimerval it;
  it.it_value.tv_sec = 1;
  it.it_value.tv_usec = 0;
  it.it_interval = it.it_value;
  setitimer (ITIMER_REAL, &it, NULL);

  clobber_lots_of_regs ();
  pause ();

  memset (&it, 0, sizeof (it));
  setitimer (ITIMER_REAL, &it, NULL);

  rc = pthread_setcanceltype (PTHREAD_CANCEL_ASYNCHRONOUS, &old);
  if (rc)
    {
      error (0, rc, "pthread_setcanceltype after pause failed");
      ret = 1;
    }
  if (old != PTHREAD_CANCEL_DEFERRED)
    {
      error (0, 0, "pthread_setcanceltype after pause returned invalid value %d",
	     old);
      ret = 1;
    }

  it.it_value.tv_sec = 1;
  it.it_value.tv_usec = 0;
  it.it_interval = it.it_value;
  setitimer (ITIMER_REAL, &it, NULL);

  clobber_lots_of_regs ();
  pause ();

  memset (&it, 0, sizeof (it));
  setitimer (ITIMER_REAL, &it, NULL);

  rc = pthread_setcanceltype (PTHREAD_CANCEL_DEFERRED, &old);
  if (rc)
    {
      error (0, rc, "pthread_setcanceltype after 2nd pause failed");
      ret = 1;
    }
  if (old != PTHREAD_CANCEL_ASYNCHRONOUS)
    {
      error (0, 0, "pthread_setcanceltype after 2nd pause returned invalid value %d",
	     old);
      ret = 1;
    }

  char fname[] = "/tmp/tst-cancel19-dir-XXXXXX\0foo/bar";
  char *enddir = strchr (fname, '\0');
  if (mkdtemp (fname) == NULL)
    {
      error (0, errno, "mkdtemp failed");
      ret = 1;
    }
  *enddir = '/';

  clobber_lots_of_regs ();
  creat (fname, 0400);

  rc = pthread_setcanceltype (PTHREAD_CANCEL_ASYNCHRONOUS, &old);
  if (rc)
    {
      error (0, rc, "pthread_setcanceltype after creat failed");
      ret = 1;
    }
  if (old != PTHREAD_CANCEL_DEFERRED)
    {
      error (0, 0, "pthread_setcanceltype after creat returned invalid value %d",
	     old);
      ret = 1;
    }

  clobber_lots_of_regs ();
  creat (fname, 0400);

  rc = pthread_setcanceltype (PTHREAD_CANCEL_DEFERRED, &old);
  if (rc)
    {
      error (0, rc, "pthread_setcanceltype after 2nd creat failed");
      ret = 1;
    }
  if (old != PTHREAD_CANCEL_ASYNCHRONOUS)
    {
      error (0, 0, "pthread_setcanceltype after 2nd creat returned invalid value %d",
	     old);
      ret = 1;
    }

  clobber_lots_of_regs ();
  open (fname, O_CREAT, 0400);

  rc = pthread_setcanceltype (PTHREAD_CANCEL_ASYNCHRONOUS, &old);
  if (rc)
    {
      error (0, rc, "pthread_setcanceltype after open failed");
      ret = 1;
    }
  if (old != PTHREAD_CANCEL_DEFERRED)
    {
      error (0, 0, "pthread_setcanceltype after open returned invalid value %d",
	     old);
      ret = 1;
    }

  clobber_lots_of_regs ();
  open (fname, O_CREAT, 0400);

  rc = pthread_setcanceltype (PTHREAD_CANCEL_DEFERRED, &old);
  if (rc)
    {
      error (0, rc, "pthread_setcanceltype after 2nd open failed");
      ret = 1;
    }
  if (old != PTHREAD_CANCEL_ASYNCHRONOUS)
    {
      error (0, 0, "pthread_setcanceltype after 2nd open returned invalid value %d",
	     old);
      ret = 1;
    }

  *enddir = '\0';
  rmdir (fname);

  clobber_lots_of_regs ();
  select (-1, NULL, NULL, NULL, NULL);

  rc = pthread_setcanceltype (PTHREAD_CANCEL_ASYNCHRONOUS, &old);
  if (rc)
    {
      error (0, rc, "pthread_setcanceltype after select failed");
      ret = 1;
    }
  if (old != PTHREAD_CANCEL_DEFERRED)
    {
      error (0, 0, "pthread_setcanceltype after select returned invalid value %d",
	     old);
      ret = 1;
    }

  clobber_lots_of_regs ();
  select (-1, NULL, NULL, NULL, NULL);

  rc = pthread_setcanceltype (PTHREAD_CANCEL_DEFERRED, &old);
  if (rc)
    {
      error (0, rc, "pthread_setcanceltype after 2nd select failed");
      ret = 1;
    }
  if (old != PTHREAD_CANCEL_ASYNCHRONOUS)
    {
      error (0, 0, "pthread_setcanceltype after 2nd select returned invalid value %d",
	     old);
      ret = 1;
    }

  pthread_join (th, NULL);

  return ret;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
