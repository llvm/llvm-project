/* Test message queue passing.
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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>
#include <stdint.h>
#include "tst-mqueue.h"

static int
intcmp (const void *a, const void *b)
{
  if (*(unsigned char *)a < *(unsigned char *)b)
    return 1;
  if (*(unsigned char *)a > *(unsigned char *)b)
    return -1;
  return 0;
}

static int
check_attrs (struct mq_attr *attr, int nonblock, long cnt)
{
  int result = 0;

  if (attr->mq_maxmsg != 10 || attr->mq_msgsize != 1)
    {
      printf ("attributes don't match those passed to mq_open\n"
	      "mq_maxmsg %jd, mq_msgsize %jd\n",
	      (intmax_t) attr->mq_maxmsg, (intmax_t) attr->mq_msgsize);
      result = 1;
    }

  if ((attr->mq_flags & O_NONBLOCK) != nonblock)
    {
      printf ("mq_flags %jx != %x\n",
	      (intmax_t) (attr->mq_flags & O_NONBLOCK), nonblock);
      result = 1;
    }

  if (attr->mq_curmsgs != cnt)
    {
      printf ("mq_curmsgs %jd != %ld\n", (intmax_t) attr->mq_curmsgs, cnt);
      result = 1;
    }

  return result;
}

static int
do_one_test (mqd_t q, const char *name, int nonblock)
{
  int result = 0;

  unsigned char v []
    = { 0x32, 0x62, 0x22, 0x31, 0x11, 0x73, 0x61, 0x21, 0x72, 0x71, 0x81 };

  struct mq_attr attr;
  memset (&attr, 0xaa, sizeof (attr));
  if (mq_getattr (q, &attr) != 0)
    {
      printf ("mq_getattr failed: %m\n");
      result = 1;
    }
  else
    result |= check_attrs (&attr, nonblock, 0);

  if (mq_receive (q, (char *) &v[0], 1, NULL) != -1)
    {
      puts ("mq_receive on O_WRONLY mqd_t unexpectedly succeeded");
      result = 1;
    }
  else if (errno != EBADF)
    {
      printf ("mq_receive on O_WRONLY mqd_t did not fail with EBADF: %m\n");
      result = 1;
    }

  struct timespec ts;
  if (clock_gettime (CLOCK_REALTIME, &ts) == 0)
    --ts.tv_sec;
  else
    {
      ts.tv_sec = time (NULL) - 1;
      ts.tv_nsec = 0;
    }

  int ret;
  for (int i = 0; i < 10; ++i)
    {
      if (i & 1)
	ret = mq_send (q, (char *) &v[i], 1, v[i] >> 4);
      else
	ret = mq_timedsend (q, (char *) &v[i], 1, v[i] >> 4, &ts);

      if (ret)
	{
	  printf ("mq_%ssend failed: %m\n", (i & 1) ? "" : "timed");
	  result = 1;
	}
    }

  ret = mq_timedsend (q, (char *) &v[10], 1, 8, &ts);
  if (ret != -1)
    {
      puts ("mq_timedsend on full queue did not fail");
      result = 1;
    }
  else if (errno != (nonblock ? EAGAIN : ETIMEDOUT))
    {
      printf ("mq_timedsend on full queue did not fail with %s: %m\n",
	      nonblock ? "EAGAIN" : "ETIMEDOUT");
      result = 1;
    }

  if (nonblock)
    {
      ret = mq_send (q, (char *) &v[10], 1, 8);
      if (ret != -1)
	{
	  puts ("mq_send on full non-blocking queue did not fail");
	  result = 1;
	}
      else if (errno != EAGAIN)
	{
	  printf ("mq_send on full non-blocking queue did not fail"
		  "with EAGAIN: %m\n");
	  result = 1;
	}
    }

  memset (&attr, 0xaa, sizeof (attr));
  if (mq_getattr (q, &attr) != 0)
    {
      printf ("mq_getattr failed: %m\n");
      result = 1;
    }
  else
    result |= check_attrs (&attr, nonblock, 10);

  pid_t pid = fork ();
  if (pid == -1)
    {
      printf ("fork failed: %m\n");
      result = 1;
    }
  else if (pid == 0)
    {
      result = 0;

      if (mq_close (q) != 0)
	{
	  printf ("mq_close in child failed: %m\n");
	  result = 1;
	}

      q = mq_open (name, O_RDONLY | nonblock);
      if (q == (mqd_t) -1)
        {
	  printf ("mq_open in child failed: %m\n");
	  exit (1);
        }

      memset (&attr, 0xaa, sizeof (attr));
      if (mq_getattr (q, &attr) != 0)
	{
	  printf ("mq_getattr failed: %m\n");
	  result = 1;
	}
      else
	result |= check_attrs (&attr, nonblock, 10);

      unsigned char vr[11] = { };
      unsigned int prio;
      ssize_t rets;

      if (mq_send (q, (char *) &v[0], 1, 1) != -1)
	{
	  puts ("mq_send on O_RDONLY mqd_t unexpectedly succeeded");
	  result = 1;
	}
      else if (errno != EBADF)
	{
	  printf ("mq_send on O_WRONLY mqd_t did not fail with EBADF: %m\n");
	  result = 1;
	}

      for (int i = 0; i < 10; ++i)
	{
	  if (i & 1)
	    rets = mq_receive (q, (char *) &vr[i], 1, &prio);
	  else
	    rets = mq_timedreceive (q, (char *) &vr[i], 1, &prio, &ts);

	  if (rets != 1)
	    {
	      if (rets == -1)
		printf ("mq_%sreceive failed: %m\n", (i & 1) ? "" : "timed");
	      else
		printf ("mq_%sreceive returned %zd != 1\n",
			(i & 1) ? "" : "timed", rets);
	      result = 1;
	    }
	  else if (prio != (unsigned int) vr[i] >> 4)
	    {
	      printf ("unexpected priority %x for value %02x\n", prio,
		      vr[i]);
	      result = 1;
	    }
	}

      qsort (v, 10, 1, intcmp);
      if (memcmp (v, vr, 10) != 0)
	{
	  puts ("messages not received in expected order");
	  result = 1;
	}

      rets = mq_timedreceive (q, (char *) &vr[10], 1, &prio, &ts);
      if (rets != -1)
	{
	  puts ("mq_timedreceive on empty queue did not fail");
	  result = 1;
	}
      else if (errno != (nonblock ? EAGAIN : ETIMEDOUT))
	{
	  printf ("mq_timedreceive on empty queue did not fail with %s: %m\n",
		  nonblock ? "EAGAIN" : "ETIMEDOUT");
	  result = 1;
	}

      if (nonblock)
	{
	  ret = mq_receive (q, (char *) &vr[10], 1, &prio);
	  if (ret != -1)
	    {
	      puts ("mq_receive on empty non-blocking queue did not fail");
	      result = 1;
	    }
	  else if (errno != EAGAIN)
	    {
	      printf ("mq_receive on empty non-blocking queue did not fail"
		      "with EAGAIN: %m\n");
	      result = 1;
	    }
	}

      memset (&attr, 0xaa, sizeof (attr));
      if (mq_getattr (q, &attr) != 0)
	{
	  printf ("mq_getattr failed: %m\n");
	  result = 1;
	}
      else
	result |= check_attrs (&attr, nonblock, 0);

      if (mq_close (q) != 0)
	{
	  printf ("mq_close in child failed: %m\n");
	  result = 1;
	}

      exit (result);
    }

  int status;
  if (TEMP_FAILURE_RETRY (waitpid (pid, &status, 0)) != pid)
    {
      printf ("waitpid failed: %m\n");
      kill (pid, SIGKILL);
      result = 1;
    }
  else if (!WIFEXITED (status) || WEXITSTATUS (status))
    {
      printf ("child failed: %d\n", status);
      result = 1;
    }

  memset (&attr, 0xaa, sizeof (attr));
  if (mq_getattr (q, &attr) != 0)
    {
      printf ("mq_getattr failed: %m\n");
      result = 1;
    }
  else
    result |= check_attrs (&attr, nonblock, 0);

  return result;
}

#define TEST_FUNCTION do_test ()
static int
do_test (void)
{
  int result = 0;

  char name[sizeof "/tst-mqueue1-" + sizeof (pid_t) * 3];
  snprintf (name, sizeof (name), "/tst-mqueue1-%u", getpid ());

  struct mq_attr attr = { .mq_maxmsg = 10, .mq_msgsize = 1 };
  mqd_t q = mq_open (name, O_CREAT | O_EXCL | O_WRONLY, 0600, &attr);

  if (q == (mqd_t) -1)
    {
      printf ("mq_open failed with: %m\n");
      return result;
    }
  else
    add_temp_mq (name);

  result |= do_one_test (q, name, 0);

  mqd_t q2 = mq_open (name, O_WRONLY | O_NONBLOCK);
  if (q2 == (mqd_t) -1)
    {
      printf ("mq_open failed with: %m\n");
      q2 = q;
      result = 1;
    }
  else
    {
      if (mq_close (q) != 0)
	{
	  printf ("mq_close in parent failed: %m\n");
	  result = 1;
	}

      q = q2;
      result |= do_one_test (q, name, O_NONBLOCK);

      if (mq_getattr (q, &attr) != 0)
	{
	  printf ("mq_getattr failed: %m\n");
	  result = 1;
	}
      else
	{
	  attr.mq_flags ^= O_NONBLOCK;

	  struct mq_attr attr2;
	  memset (&attr2, 0x55, sizeof (attr2));
	  if (mq_setattr (q, &attr, &attr2) != 0)
	    {
	      printf ("mq_setattr failed: %m\n");
	      result = 1;
	    }
	  else if (attr.mq_flags != (attr2.mq_flags ^ O_NONBLOCK)
		   || attr.mq_maxmsg != attr2.mq_maxmsg
		   || attr.mq_msgsize != attr2.mq_msgsize
		   || attr.mq_curmsgs != 0
		   || attr2.mq_curmsgs != 0)
	    {
	      puts ("mq_setattr returned unexpected values in *omqstat");
	      result = 1;
	    }
	  else
	    {
	      result |= do_one_test (q, name, 0);

	      if (mq_setattr (q, &attr2, NULL) != 0)
		{
		  printf ("mq_setattr failed: %m\n");
		  result = 1;
		}
	      else
		result |= do_one_test (q, name, O_NONBLOCK);
	    }
	}
    }

  if (mq_unlink (name) != 0)
    {
      printf ("mq_unlink failed: %m\n");
      result = 1;
    }

  if (mq_close (q) != 0)
    {
      printf ("mq_close in parent failed: %m\n");
      result = 1;
    }

  if (mq_close (q) != -1)
    {
      puts ("second mq_close did not fail");
      result = 1;
    }
  else if (errno != EBADF)
    {
      printf ("second mq_close did not fail with EBADF: %m\n");
      result = 1;
    }

  return result;
}

#include "../test-skeleton.c"
