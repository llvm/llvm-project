/* Copyright (C) 2004-2021 Free Software Foundation, Inc.
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
#include <mqueue.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#if _POSIX_THREADS
# include <pthread.h>

static pthread_barrier_t b;

/* Cleanup handling test.  */
static int cl_called;

static void
cl (void *arg)
{
  ++cl_called;
}

#define TF_MQ_RECEIVE		0L
#define TF_MQ_TIMEDRECEIVE	1L
#define TF_MQ_SEND		2L
#define TF_MQ_TIMEDSEND		3L

static const char *names[]
  = { "mq_receive", "mq_timedreceive", "mq_send", "mq_timedsend" };

static mqd_t q;
static struct timespec never;

static void *
tf (void *arg)
{
  int r = pthread_barrier_wait (&b);
  if (r != 0 && r != PTHREAD_BARRIER_SERIAL_THREAD)
    {
      puts ("tf: barrier_wait failed");
      exit (1);
    }

  pthread_cleanup_push (cl, NULL);

  char c = ' ';

  switch ((long) arg)
    {
    case TF_MQ_SEND:
      TEMP_FAILURE_RETRY (mq_send (q, &c, 1, 1));
      break;
    case TF_MQ_TIMEDSEND:
      TEMP_FAILURE_RETRY (mq_timedsend (q, &c, 1, 1, &never));
      break;
    case TF_MQ_RECEIVE:
      TEMP_FAILURE_RETRY (mq_receive (q, &c, 1, NULL));
      break;
    case TF_MQ_TIMEDRECEIVE:
      TEMP_FAILURE_RETRY (mq_timedreceive (q, &c, 1, NULL, &never));
      break;
    }

  pthread_cleanup_pop (0);

  printf ("tf: %s returned\n", names[(long) arg]);

  exit (1);
}

#define TEST_FUNCTION do_test ()
static int
do_test (void)
{
  char name[sizeof "/tst-mqueue8-" + sizeof (pid_t) * 3];
  snprintf (name, sizeof (name), "/tst-mqueue8-%u", getpid ());

  struct mq_attr attr = { .mq_maxmsg = 1, .mq_msgsize = 1 };
  q = mq_open (name, O_CREAT | O_EXCL | O_RDWR, 0600, &attr);

  if (q == (mqd_t) -1)
    {
      printf ("mq_open failed with: %m\n");
      return 0;
    }

  if (mq_unlink (name) != 0)
    {
      printf ("mq_unlink failed with: %m\n");
      return 1;
    }

  if (pthread_barrier_init (&b, NULL, 2) != 0)
    {
      puts ("barrier_init failed");
      return 1;
    }

  if (clock_gettime (CLOCK_REALTIME, &never) == 0)
    never.tv_sec += 100;
  else
    {
      never.tv_sec = time (NULL) + 100;
      never.tv_nsec = 0;
    }

  int result = 0;
  for (long l = TF_MQ_RECEIVE; l <= TF_MQ_TIMEDSEND; ++l)
    {
      cl_called = 0;

      pthread_t th;
      if (pthread_create (&th, NULL, tf, (void *) l) != 0)
	{
	  printf ("1st %s create failed\n", names[l]);
	  result = 1;
	  continue;
	}

      int r = pthread_barrier_wait (&b);
      if (r != 0 && r != PTHREAD_BARRIER_SERIAL_THREAD)
	{
	  puts ("barrier_wait failed");
	  result = 1;
	  continue;
	}

      struct timespec ts = { .tv_sec = 0, .tv_nsec = 100000000 };
      while (nanosleep (&ts, &ts) != 0)
	continue;

      printf ("going to cancel %s in-time\n", names[l]);
      if (pthread_cancel (th) != 0)
	{
	  printf ("1st cancel of %s failed\n", names[l]);
	  result = 1;
	  continue;
	}

      void *status;
      if (pthread_join (th, &status) != 0)
	{
	  printf ("1st join of %s failed\n", names[l]);
	  result = 1;
	  continue;
	}
      if (status != PTHREAD_CANCELED)
	{
	  printf ("1st %s thread not canceled\n", names[l]);
	  result = 1;
	  continue;
	}

      if (cl_called == 0)
	{
	  printf ("%s cleanup handler not called\n", names[l]);
	  result = 1;
	  continue;
	}
      if (cl_called > 1)
	{
	  printf ("%s cleanup handler called more than once\n", names[l]);
	  result = 1;
	  continue;
	}

      printf ("in-time %s cancellation succeeded\n", names[l]);

      cl_called = 0;

      if (pthread_create (&th, NULL, tf, (void *) l) != 0)
	{
	  printf ("2nd %s create failed\n", names[l]);
	  result = 1;
	  continue;
	}

      printf ("going to cancel %s early\n", names[l]);
      if (pthread_cancel (th) != 0)
	{
	  printf ("2nd cancel of %s failed\n", names[l]);
	  result = 1;
	  continue;
	}

      r = pthread_barrier_wait (&b);
      if (r != 0 && r != PTHREAD_BARRIER_SERIAL_THREAD)
	{
	  puts ("barrier_wait failed");
	  result = 1;
	  continue;
	}

      if (pthread_join (th, &status) != 0)
	{
	  printf ("2nd join of %s failed\n", names[l]);
	  result = 1;
	  continue;
	}
      if (status != PTHREAD_CANCELED)
	{
	  printf ("2nd %s thread not canceled\n", names[l]);
	  result = 1;
	  continue;
	}

      if (cl_called == 0)
	{
	  printf ("%s cleanup handler not called\n", names[l]);
	  result = 1;
	  continue;
	}
      if (cl_called > 1)
	{
	  printf ("%s cleanup handler called more than once\n", names[l]);
	  result = 1;
	  continue;
	}

      printf ("early %s cancellation succeeded\n", names[l]);

      if (l == TF_MQ_TIMEDRECEIVE)
	{
	  /* mq_receive and mq_timedreceive are tested on empty mq.
	     For mq_send and mq_timedsend we need to make it full.
	     If this fails, there is no point in doing further testing.  */
	  char c = ' ';
	  if (mq_send (q, &c, 1, 1) != 0)
	    {
	      printf ("mq_send failed: %m\n");
	      result = 1;
	      break;
	    }
	}
    }

  if (mq_close (q) != 0)
    {
      printf ("mq_close failed: %m\n");
      result = 1;
    }

  return result;
}
#else
# define TEST_FUNCTION 0
#endif

#include "../test-skeleton.c"
