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
#include <limits.h>
#include <signal.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>
#include "tst-mqueue.h"

#define TEST_FUNCTION do_test ()
static int
do_test (void)
{
  int result = 0;

  char name[sizeof "/tst-mqueue4-" + sizeof (pid_t) * 3 + NAME_MAX];
  char *p;
  p = name + snprintf (name, sizeof (name), "/tst-mqueue4-%u", getpid ());
  struct mq_attr attr = { .mq_maxmsg = 2, .mq_msgsize = 2 };
  mqd_t q = mq_open (name, O_CREAT | O_EXCL | O_RDWR, 0600, &attr);

  if (q == (mqd_t) -1)
    {
      printf ("mq_open failed with: %m\n");
      return result;
    }
  else
    add_temp_mq (name);

  *p = '.';
  memset (p + 1, 'x', NAME_MAX + 1 - (p - name));
  name[NAME_MAX + 1] = '\0';

  mqd_t q2 = mq_open (name, O_CREAT | O_EXCL | O_RDWR, 0600, &attr);
  if (q2 == (mqd_t) -1)
    {
      printf ("mq_open with NAME_MAX long name compoment failed with: %m\n");
      result = 1;
    }

  if (mq_unlink (name) != 0)
    {
      printf ("mq_unlink failed: %m\n");
      result = 1;
    }

  if (mq_close (q2) != 0)
    {
      printf ("mq_close failed: %m\n");
      result = 1;
    }

  name[NAME_MAX + 1] = 'x';
  name[NAME_MAX + 2] = '\0';
  q2 = mq_open (name, O_CREAT | O_EXCL | O_RDWR, 0600, &attr);
  if (q2 != (mqd_t) -1)
    {
      puts ("mq_open with too long name component unexpectedly succeeded");
      mq_unlink (name);
      mq_close (q2);
      result = 1;
    }
  else if (errno != ENAMETOOLONG)
    {
      printf ("mq_open with too long name component did not fail with "
	      "ENAMETOOLONG: %m\n");
      result = 1;
    }

  if (mq_unlink (name) == 0)
    {
      puts ("mq_unlink with too long name component unexpectedly succeeded");
      result = 1;
    }
  else if (errno != ENAMETOOLONG)
    {
      printf ("mq_unlink with too long name component did not fail with "
	      "ENAMETOOLONG: %m\n");
      result = 1;
    }

  *p = '\0';
  attr.mq_maxmsg = 1;
  attr.mq_msgsize = 3;
  q2 = mq_open (name, O_CREAT | O_RDWR, 0600, &attr);
  if (q2 == (mqd_t) -1)
    {
      printf ("mq_open without O_EXCL failed with %m\n");
      result = 1;
    }

  char buf[3];
  strcpy (buf, "jk");
  if (mq_send (q, buf, 2, 4) != 0)
    {
      printf ("mq_send failed: %m\n");
      result = 1;
    }

  if (mq_send (q, buf + 1, 1, 5) != 0)
    {
      printf ("mq_send failed: %m\n");
      result = 1;
    }

  if (mq_getattr (q2, &attr) != 0)
    {
      printf ("mq_getattr failed: %m\n");
      result = 1;
    }

  if ((attr.mq_flags & O_NONBLOCK)
      || attr.mq_maxmsg != 2
      || attr.mq_msgsize != 2
      || attr.mq_curmsgs != 2)
    {
      printf ("mq_getattr returned unexpected { .mq_flags = %jd,\n"
	      ".mq_maxmsg = %jd, .mq_msgsize = %jd, .mq_curmsgs = %jd }\n",
	      (intmax_t) attr.mq_flags, (intmax_t) attr.mq_maxmsg,
	      (intmax_t) attr.mq_msgsize, (intmax_t) attr.mq_curmsgs);
      result = 1;
    }

  struct timespec ts;
  if (clock_gettime (CLOCK_REALTIME, &ts) == 0)
    ++ts.tv_sec;
  else
    {
      ts.tv_sec = time (NULL) + 1;
      ts.tv_nsec = 0;
    }

  if (mq_timedsend (q2, buf, 1, 1, &ts) == 0)
    {
      puts ("mq_timedsend unexpectedly succeeded");
      result = 1;
    }
  else if (errno != ETIMEDOUT)
    {
      printf ("mq_timedsend did not fail with ETIMEDOUT: %m\n");
      result = 1;
    }

  if (mq_close (q2) != 0)
    {
      printf ("mq_close failed: %m\n");
      result = 1;
    }

  q2 = mq_open (name, O_RDONLY, 0600);
  if (q2 == (mqd_t) -1)
    {
      printf ("mq_open without O_CREAT failed with %m\n");
      result = 1;
    }

  mqd_t q3 = mq_open (name, O_RDONLY, 0600);
  if (q3 == (mqd_t) -1)
    {
      printf ("mq_open without O_CREAT failed with %m\n");
      result = 1;
    }

  memset (buf, ' ', sizeof (buf));

  unsigned int prio;
  ssize_t rets = mq_receive (q2, buf, 2, &prio);
  if (rets != 1)
    {
      if (rets == -1)
	printf ("mq_receive failed with: %m\n");
      else
	printf ("mq_receive returned %zd != 1\n", rets);
      result = 1;
    }
  else if (prio != 5 || memcmp (buf, "k  ", 3) != 0)
    {
      printf ("mq_receive returned prio %u (2) buf \"%c%c%c\" (\"k  \")\n",
	      prio, buf[0], buf[1], buf[2]);
      result = 1;
    }

  if (mq_getattr (q3, &attr) != 0)
    {
      printf ("mq_getattr failed: %m\n");
      result = 1;
    }

  if ((attr.mq_flags & O_NONBLOCK)
      || attr.mq_maxmsg != 2
      || attr.mq_msgsize != 2
      || attr.mq_curmsgs != 1)
    {
      printf ("mq_getattr returned unexpected { .mq_flags = %jd,\n"
	      ".mq_maxmsg = %jd, .mq_msgsize = %jd, .mq_curmsgs = %jd }\n",
	      (intmax_t) attr.mq_flags, (intmax_t) attr.mq_maxmsg,
	      (intmax_t) attr.mq_msgsize, (intmax_t) attr.mq_curmsgs);
      result = 1;
    }

  rets = mq_receive (q3, buf, 2, NULL);
  if (rets != 2)
    {
      if (rets == -1)
	printf ("mq_receive failed with: %m\n");
      else
	printf ("mq_receive returned %zd != 2\n", rets);
      result = 1;
    }
  else if (memcmp (buf, "jk ", 3) != 0)
    {
      printf ("mq_receive returned buf \"%c%c%c\" != \"jk \"\n",
	      buf[0], buf[1], buf[2]);
      result = 1;
    }

  if (clock_gettime (CLOCK_REALTIME, &ts) == 0)
    ++ts.tv_sec;
  else
    {
      ts.tv_sec = time (NULL) + 1;
      ts.tv_nsec = 0;
    }

  if (mq_timedreceive (q2, buf, 2, NULL, &ts) != -1)
    {
      puts ("mq_timedreceive on empty queue unexpectedly succeeded");
      result = 1;
    }
  else if (errno != ETIMEDOUT)
    {
      printf ("mq_timedreceive on empty queue did not fail with "
	      "ETIMEDOUT: %m\n");
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

  if (mq_close (q2) != 0)
    {
      printf ("mq_close failed: %m\n");
      result = 1;
    }

  if (mq_close (q3) != 0)
    {
      printf ("mq_close failed: %m\n");
      result = 1;
    }

  return result;
}

#include "../test-skeleton.c"
