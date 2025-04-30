/* Test mq_notify.
   Copyright (C) 2004-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Jakub Jelinek <jakub@redhat.com>, 2004.

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
#include <fcntl.h>
#include <mqueue.h>
#include <limits.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/time.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>
#include "tst-mqueue.h"

#if _POSIX_THREADS
# include <pthread.h>

# define mqsend(q) (mqsend) (q, __LINE__)
static int
(mqsend) (mqd_t q, int line)
{
  char c;
  if (mq_send (q, &c, 1, 1) != 0)
    {
      printf ("mq_send on line %d failed with: %m\n", line);
      return 1;
    }
  return 0;
}

# define mqrecv(q) (mqrecv) (q, __LINE__)
static int
(mqrecv) (mqd_t q, int line)
{
  char c;
  ssize_t rets = TEMP_FAILURE_RETRY (mq_receive (q, &c, 1, NULL));
  if (rets != 1)
    {
      if (rets == -1)
	printf ("mq_receive on line %d failed with: %m\n", line);
      else
	printf ("mq_receive on line %d returned %zd != 1\n",
		line, rets);
      return 1;
    }
  return 0;
}

volatile int fct_cnt, fct_err;
size_t fct_guardsize;

static void
fct (union sigval s)
{
  mqd_t q = *(mqd_t *) s.sival_ptr;

  pthread_attr_t nattr;
  int ret = pthread_getattr_np (pthread_self (), &nattr);
  if (ret)
    {
      errno = ret;
      printf ("pthread_getattr_np failed: %m\n");
      fct_err = 1;
    }
  else
    {
      ret = pthread_attr_getguardsize (&nattr, &fct_guardsize);
      if (ret)
	{
	  errno = ret;
	  printf ("pthread_attr_getguardsize failed: %m\n");
	  fct_err = 1;
	}
      if (pthread_attr_destroy (&nattr) != 0)
	{
	  puts ("pthread_attr_destroy failed");
	  fct_err = 1;
	}
    }

  ++fct_cnt;
  fct_err |= mqsend (q);
}

# define TEST_FUNCTION do_test ()
static int
do_test (void)
{
  int result = 0;

  char name[sizeof "/tst-mqueue6-" + sizeof (pid_t) * 3];
  snprintf (name, sizeof (name), "/tst-mqueue6-%u", getpid ());

  struct mq_attr attr = { .mq_maxmsg = 1, .mq_msgsize = 1 };
  mqd_t q = mq_open (name, O_CREAT | O_EXCL | O_RDWR, 0600, &attr);

  if (q == (mqd_t) -1)
    {
      printf ("mq_open failed with: %m\n");
      return result;
    }
  else
    add_temp_mq (name);

  pthread_attr_t nattr;
  if (pthread_attr_init (&nattr)
      || pthread_attr_setguardsize (&nattr, 0))
    {
      puts ("pthread_attr_t setup failed");
      result = 1;
    }

  fct_guardsize = 1;

  struct sigevent ev;
  memset (&ev, 0xaa, sizeof (ev));
  ev.sigev_notify = SIGEV_THREAD;
  ev.sigev_notify_function = fct;
  ev.sigev_notify_attributes = &nattr;
  ev.sigev_value.sival_ptr = &q;
  if (mq_notify (q, &ev) != 0)
    {
      printf ("mq_notify (q, { SIGEV_THREAD }) failed with: %m\n");
      result = 1;
    }

  size_t ps = sysconf (_SC_PAGESIZE);
  if (pthread_attr_setguardsize (&nattr, 32 * ps))
    {
      puts ("pthread_attr_t setup failed");
      result = 1;
    }

  if (mq_notify (q, &ev) == 0)
    {
      puts ("second mq_notify (q, { SIGEV_NONE }) unexpectedly succeeded");
      result = 1;
    }
  else if (errno != EBUSY)
    {
      printf ("second mq_notify (q, { SIGEV_NONE }) failed with: %m\n");
      result = 1;
    }

  if (fct_cnt != 0)
    {
      printf ("fct called too early (%d on %d)\n", fct_cnt, __LINE__);
      result = 1;
    }

  result |= mqsend (q);

  result |= mqrecv (q);
  result |= mqrecv (q);

  if (fct_cnt != 1)
    {
      printf ("fct not called (%d on %d)\n", fct_cnt, __LINE__);
      result = 1;
    }
  else if (fct_guardsize != 0)
    {
      printf ("fct_guardsize %zd != 0\n", fct_guardsize);
      result = 1;
    }

  if (mq_notify (q, &ev) != 0)
    {
      printf ("third mq_notify (q, { SIGEV_NONE }) failed with: %m\n");
      result = 1;
    }

  if (mq_notify (q, NULL) != 0)
    {
      printf ("mq_notify (q, NULL) failed with: %m\n");
      result = 1;
    }

  memset (&ev, 0x11, sizeof (ev));
  ev.sigev_notify = SIGEV_THREAD;
  ev.sigev_notify_function = fct;
  ev.sigev_notify_attributes = &nattr;
  ev.sigev_value.sival_ptr = &q;
  if (mq_notify (q, &ev) != 0)
    {
      printf ("mq_notify (q, { SIGEV_THREAD }) failed with: %m\n");
      result = 1;
    }

  if (pthread_attr_setguardsize (&nattr, 0))
    {
      puts ("pthread_attr_t setup failed");
      result = 1;
    }

  if (mq_notify (q, &ev) == 0)
    {
      puts ("second mq_notify (q, { SIGEV_NONE }) unexpectedly succeeded");
      result = 1;
    }
  else if (errno != EBUSY)
    {
      printf ("second mq_notify (q, { SIGEV_NONE }) failed with: %m\n");
      result = 1;
    }

  if (fct_cnt != 1)
    {
      printf ("fct called too early (%d on %d)\n", fct_cnt, __LINE__);
      result = 1;
    }

  result |= mqsend (q);

  result |= mqrecv (q);
  result |= mqrecv (q);

  if (fct_cnt != 2)
    {
      printf ("fct not called (%d on %d)\n", fct_cnt, __LINE__);
      result = 1;
    }
  else if (fct_guardsize != 32 * ps)
    {
      printf ("fct_guardsize %zd != %zd\n", fct_guardsize, 32 * ps);
      result = 1;
    }

  if (mq_notify (q, &ev) != 0)
    {
      printf ("third mq_notify (q, { SIGEV_NONE }) failed with: %m\n");
      result = 1;
    }

  if (mq_notify (q, NULL) != 0)
    {
      printf ("mq_notify (q, NULL) failed with: %m\n");
      result = 1;
    }

  if (pthread_attr_destroy (&nattr) != 0)
    {
      puts ("pthread_attr_destroy failed");
      result = 1;
    }

  if (mq_unlink (name) != 0)
    {
      printf ("mq_unlink failed: %m\n");
      result = 1;
    }

  if (mq_close (q) != 0)
    {
      printf ("mq_close failed: %m\n");
      result = 1;
    }

  memset (&ev, 0x55, sizeof (ev));
  ev.sigev_notify = SIGEV_THREAD;
  ev.sigev_notify_function = fct;
  ev.sigev_notify_attributes = NULL;
  ev.sigev_value.sival_int = 0;
  if (mq_notify (q, &ev) == 0)
    {
      puts ("mq_notify on closed mqd_t unexpectedly succeeded");
      result = 1;
    }
  else if (errno != EBADF)
    {
      printf ("mq_notify on closed mqd_t did not fail with EBADF: %m\n");
      result = 1;
    }

  if (fct_err)
    result = 1;
  return result;
}
#else
# define TEST_FUNCTION 0
#endif

#include "../test-skeleton.c"
