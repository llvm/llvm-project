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
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>
#include "tst-mqueue.h"

static void
alrm_handler (int sig)
{
}

#define TEST_FUNCTION do_test ()
static int
do_test (void)
{
  int result = 0;

  char name[sizeof "/tst-mqueue2-" + sizeof (pid_t) * 3];
  snprintf (name, sizeof (name), "/tst-mqueue2-%u", getpid ());

  struct mq_attr attr = { .mq_maxmsg = 2, .mq_msgsize = 2 };
  mqd_t q = mq_open (name, O_CREAT | O_EXCL | O_RDWR, 0600, &attr);

  if (q == (mqd_t) -1)
    {
      printf ("mq_open failed with: %m\n");
      return result;
    }
  else
    add_temp_mq (name);

  mqd_t q2 = mq_open (name, O_CREAT | O_EXCL | O_RDWR, 0600, &attr);
  if (q2 != (mqd_t) -1)
    {
      puts ("mq_open with O_EXCL unexpectedly succeeded");
      result = 1;
    }
  else if (errno != EEXIST)
    {
      printf ("mq_open did not fail with EEXIST: %m\n");
      result = 1;
    }

  char name2[sizeof "/tst-mqueue2-2-" + sizeof (pid_t) * 3];
  snprintf (name2, sizeof (name2), "/tst-mqueue2-2-%u", getpid ());

  attr.mq_maxmsg = -2;
  q2 = mq_open (name2, O_CREAT | O_EXCL | O_RDWR, 0600, &attr);
  if (q2 != (mqd_t) -1)
    {
      puts ("mq_open with invalid mq_maxmsg unexpectedly succeeded");
      add_temp_mq (name2);
      result = 1;
    }
  else if (errno != EINVAL)
    {
      printf ("mq_open with invalid mq_maxmsg did not fail with "
	      "EINVAL: %m\n");
      result = 1;
    }

  attr.mq_maxmsg = 2;
  attr.mq_msgsize = -56;
  q2 = mq_open (name2, O_CREAT | O_EXCL | O_RDWR, 0600, &attr);
  if (q2 != (mqd_t) -1)
    {
      puts ("mq_open with invalid mq_msgsize unexpectedly succeeded");
      add_temp_mq (name2);
      result = 1;
    }
  else if (errno != EINVAL)
    {
      printf ("mq_open with invalid mq_msgsize did not fail with "
	      "EINVAL: %m\n");
      result = 1;
    }

  char buf[3];
  struct timespec ts;
  if (clock_gettime (CLOCK_REALTIME, &ts) == 0)
    ts.tv_sec += 10;
  else
    {
      ts.tv_sec = time (NULL) + 10;
      ts.tv_nsec = 0;
    }

  if (mq_timedreceive (q, buf, 1, NULL, &ts) == 0)
    {
      puts ("mq_timedreceive with too small msg_len did not fail");
      result = 1;
    }
  else if (errno != EMSGSIZE)
    {
      printf ("mq_timedreceive with too small msg_len did not fail with "
	      "EMSGSIZE: %m\n");
      result = 1;
    }

  ts.tv_nsec = -1;
  if (mq_timedreceive (q, buf, 2, NULL, &ts) == 0)
    {
      puts ("mq_timedreceive with negative tv_nsec did not fail");
      result = 1;
    }
  else if (errno != EINVAL)
    {
      printf ("mq_timedreceive with negative tv_nsec did not fail with "
	      "EINVAL: %m\n");
      result = 1;
    }

  ts.tv_nsec = 1000000000;
  if (mq_timedreceive (q, buf, 2, NULL, &ts) == 0)
    {
      puts ("mq_timedreceive with tv_nsec >= 1000000000 did not fail");
      result = 1;
    }
  else if (errno != EINVAL)
    {
      printf ("mq_timedreceive with tv_nsec >= 1000000000 did not fail with "
	      "EINVAL: %m\n");
      result = 1;
    }

  struct sigaction sa = { .sa_handler = alrm_handler, .sa_flags = 0 };
  sigemptyset (&sa.sa_mask);
  sigaction (SIGALRM, &sa, NULL);

  struct itimerval it = { .it_value = { .tv_sec = 1 } };
  setitimer (ITIMER_REAL, &it, NULL);

  if (mq_receive (q, buf, 2, NULL) == 0)
    {
      puts ("mq_receive on empty queue did not block");
      result = 1;
    }
  else if (errno != EINTR)
    {
      printf ("mq_receive on empty queue did not fail with EINTR: %m\n");
      result = 1;
    }

  setitimer (ITIMER_REAL, &it, NULL);

  ts.tv_nsec = 0;
  if (mq_timedreceive (q, buf, 2, NULL, &ts) == 0)
    {
      puts ("mq_timedreceive on empty queue did not block");
      result = 1;
    }
  else if (errno != EINTR)
    {
      printf ("mq_timedreceive on empty queue did not fail with EINTR: %m\n");
      result = 1;
    }

  buf[0] = '6';
  buf[1] = '7';
  if (mq_send (q, buf, 2, 3) != 0
      || (buf[0] = '8', mq_send (q, buf, 1, 4) != 0))
    {
      printf ("mq_send failed: %m\n");
      result = 1;
    }

  memset (buf, ' ', sizeof (buf));

  unsigned int prio;
  ssize_t rets = mq_receive (q, buf, 3, &prio);
  if (rets != 1)
    {
      if (rets == -1)
	printf ("mq_receive failed: %m\n");
      else
	printf ("mq_receive returned %zd != 1\n", rets);
      result = 1;
    }
  else if (prio != 4 || memcmp (buf, "8  ", 3) != 0)
    {
      printf ("mq_receive prio %u (4) buf \"%c%c%c\" (\"8  \")\n",
	      prio, buf[0], buf[1], buf[2]);
      result = 1;
    }

  rets = mq_receive (q, buf, 2, NULL);
  if (rets != 2)
    {
      if (rets == -1)
	printf ("mq_receive failed: %m\n");
      else
	printf ("mq_receive returned %zd != 2\n", rets);
      result = 1;
    }
  else if (memcmp (buf, "67 ", 3) != 0)
    {
      printf ("mq_receive buf \"%c%c%c\" != \"67 \"\n",
	      buf[0], buf[1], buf[2]);
      result = 1;
    }

  buf[0] = '2';
  buf[1] = '1';
  if (clock_gettime (CLOCK_REALTIME, &ts) != 0)
    ts.tv_sec = time (NULL);
  ts.tv_nsec = -1000000001;
  if ((mq_timedsend (q, buf, 2, 5, &ts) != 0
       && (errno != EINVAL || mq_send (q, buf, 2, 5) != 0))
      || (buf[0] = '3', ts.tv_nsec = -ts.tv_nsec,
	  (mq_timedsend (q, buf, 1, 4, &ts) != 0
	   && (errno != EINVAL || mq_send (q, buf, 1, 4) != 0))))
    {
      printf ("mq_timedsend failed: %m\n");
      result = 1;
    }

  buf[0] = '-';
  ts.tv_nsec = 1000000001;
  if (mq_timedsend (q, buf, 1, 6, &ts) == 0)
    {
      puts ("mq_timedsend with tv_nsec >= 1000000000 did not fail");
      result = 1;
    }
  else if (errno != EINVAL)
    {
      printf ("mq_timedsend with tv_nsec >= 1000000000 did not fail with "
	      "EINVAL: %m\n");
      result = 1;
    }

  ts.tv_nsec = -2;
  if (mq_timedsend (q, buf, 1, 6, &ts) == 0)
    {
      puts ("mq_timedsend with negative tv_nsec did not fail");
      result = 1;
    }
  else if (errno != EINVAL)
    {
      printf ("mq_timedsend with megatove tv_nsec did not fail with "
	      "EINVAL: %m\n");
      result = 1;
    }

  setitimer (ITIMER_REAL, &it, NULL);

  if (mq_send (q, buf, 2, 8) == 0)
    {
      puts ("mq_send on full queue did not block");
      result = 1;
    }
  else if (errno != EINTR)
    {
      printf ("mq_send on full queue did not fail with EINTR: %m\n");
      result = 1;
    }

  setitimer (ITIMER_REAL, &it, NULL);

  ts.tv_sec += 10;
  ts.tv_nsec = 0;
  if (mq_timedsend (q, buf, 2, 7, &ts) == 0)
    {
      puts ("mq_timedsend on full queue did not block");
      result = 1;
    }
  else if (errno != EINTR)
    {
      printf ("mq_timedsend on full queue did not fail with EINTR: %m\n");
      result = 1;
    }

  memset (buf, ' ', sizeof (buf));

  if (clock_gettime (CLOCK_REALTIME, &ts) != 0)
    ts.tv_sec = time (NULL);
  ts.tv_nsec = -1000000001;
  rets = mq_timedreceive (q, buf, 2, &prio, &ts);
  if (rets == -1 && errno == EINVAL)
    rets = mq_receive (q, buf, 2, &prio);
  if (rets != 2)
    {
      if (rets == -1)
	printf ("mq_timedreceive failed: %m\n");
      else
	printf ("mq_timedreceive returned %zd != 2\n", rets);
      result = 1;
    }
  else if (prio != 5 || memcmp (buf, "21 ", 3) != 0)
    {
      printf ("mq_timedreceive prio %u (5) buf \"%c%c%c\" (\"21 \")\n",
	      prio, buf[0], buf[1], buf[2]);
      result = 1;
    }

  if (mq_receive (q, buf, 1, NULL) == 0)
    {
      puts ("mq_receive with too small msg_len did not fail");
      result = 1;
    }
  else if (errno != EMSGSIZE)
    {
      printf ("mq_receive with too small msg_len did not fail with "
	      "EMSGSIZE: %m\n");
      result = 1;
    }

  ts.tv_nsec = -ts.tv_nsec;
  rets = mq_timedreceive (q, buf, 2, NULL, &ts);
  if (rets == -1 && errno == EINVAL)
    rets = mq_receive (q, buf, 2, NULL);
  if (rets != 1)
    {
      if (rets == -1)
	printf ("mq_timedreceive failed: %m\n");
      else
	printf ("mq_timedreceive returned %zd != 1\n", rets);
      result = 1;
    }
  else if (memcmp (buf, "31 ", 3) != 0)
    {
      printf ("mq_timedreceive buf \"%c%c%c\" != \"31 \"\n",
	      buf[0], buf[1], buf[2]);
      result = 1;
    }

  if (mq_send (q, "", 0, 2) != 0)
    {
      printf ("mq_send with msg_len 0 failed: %m\n");
      result = 1;
    }

  rets = mq_receive (q, buf, 2, &prio);
  if (rets)
    {
      if (rets == -1)
	printf ("mq_receive failed: %m\n");
      else
	printf ("mq_receive returned %zd != 0\n", rets);
      result = 1;
    }

  long mq_prio_max = sysconf (_SC_MQ_PRIO_MAX);
  if (mq_prio_max > 0 && (unsigned int) mq_prio_max == mq_prio_max)
    {
      if (mq_send (q, buf, 1, mq_prio_max) == 0)
	{
	  puts ("mq_send with MQ_PRIO_MAX priority unpexpectedly succeeded");
	  result = 1;
	}
      else if (errno != EINVAL)
	{
	  printf ("mq_send with MQ_PRIO_MAX priority did not fail with "
		  "EINVAL: %m\n");
	  result = 1;
	}

      if (mq_send (q, buf, 1, mq_prio_max - 1) != 0)
	{
	  printf ("mq_send with MQ_PRIO_MAX-1 priority failed: %m\n");
	  result = 1;
	}
    }

  if (mq_unlink (name) != 0)
    {
      printf ("mq_unlink failed: %m\n");
      result = 1;
    }

  q2 = mq_open (name, O_RDWR);
  if (q2 != (mqd_t) -1)
    {
      printf ("mq_open of unlinked %s without O_CREAT unexpectedly"
	      "succeeded\n", name);
      result = 1;
    }
  else if (errno != ENOENT)
    {
      printf ("mq_open of unlinked %s without O_CREAT did not fail with "
	      "ENOENT: %m\n", name);
      result = 1;
    }

  if (mq_close (q) != 0)
    {
      printf ("mq_close in parent failed: %m\n");
      result = 1;
    }

  if (mq_receive (q, buf, 2, NULL) == 0)
    {
      puts ("mq_receive on invalid mqd_t did not fail");
      result = 1;
    }
  else if (errno != EBADF)
    {
      printf ("mq_receive on invalid mqd_t did not fail with EBADF: %m\n");
      result = 1;
    }

  if (mq_send (q, buf, 1, 2) == 0)
    {
      puts ("mq_send on invalid mqd_t did not fail");
      result = 1;
    }
  else if (errno != EBADF)
    {
      printf ("mq_send on invalid mqd_t did not fail with EBADF: %m\n");
      result = 1;
    }

  if (mq_getattr (q, &attr) == 0)
    {
      puts ("mq_getattr on invalid mqd_t did not fail");
      result = 1;
    }
  else if (errno != EBADF)
    {
      printf ("mq_getattr on invalid mqd_t did not fail with EBADF: %m\n");
      result = 1;
    }

  memset (&attr, 0, sizeof (attr));
  if (mq_setattr (q, &attr, NULL) == 0)
    {
      puts ("mq_setattr on invalid mqd_t did not fail");
      result = 1;
    }
  else if (errno != EBADF)
    {
      printf ("mq_setattr on invalid mqd_t did not fail with EBADF: %m\n");
      result = 1;
    }

  if (mq_unlink ("/tst-mqueue2-which-should-never-exist") != -1)
    {
      puts ("mq_unlink of non-existant message queue unexpectedly succeeded");
      result = 1;
    }
  else if (errno != ENOENT)
    {
      printf ("mq_unlink of non-existant message queue did not fail with "
	      "ENOENT: %m\n");
      result = 1;
    }
  return result;
}

#include "../test-skeleton.c"
