/* Check for large timeout with mq_timedsend and mq_timedreceive.
   Copyright (C) 2021 Free Software Foundation, Inc.
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
#include <intprops.h>
#include <mqueue.h>
#include <stdio.h>
#include <support/check.h>
#include <support/support.h>
#include <support/temp_file.h>
#include <unistd.h>

static char name[sizeof "/tst-mqueue2-" + INT_BUFSIZE_BOUND (pid_t)];

static void
do_cleanup (void)
{
  mq_unlink (name);
}
#define CLEANUP_HANDLER	do_cleanup

static int
do_test (void)
{
  snprintf (name, sizeof (name), "/tst-mqueue2-%u", getpid ());

  char msg[8] = { 0x55 };

  struct mq_attr attr = { .mq_maxmsg = 1, .mq_msgsize = sizeof (msg) };
  mqd_t q = mq_open (name, O_CREAT | O_EXCL | O_RDWR, 0600, &attr);
  TEST_VERIFY_EXIT (q != (mqd_t) -1);

  struct timespec ts = { TYPE_MAXIMUM (time_t), 0 };

  {
    timer_t timer = support_create_timer (0, 100000000, false, NULL);
    TEST_COMPARE (mq_timedreceive (q, msg, sizeof (msg), NULL, &ts), -1);
    TEST_VERIFY (errno == EINTR || errno == EOVERFLOW);
    support_delete_timer (timer);
  }

  {
    timer_t timer = support_create_timer (0, 100000000, false, NULL);
    /* Fill the internal buffer first.  */
    TEST_COMPARE (mq_timedsend (q, msg, sizeof (msg), 0,
				&(struct timespec) { 0, 0 }), 0);
    TEST_COMPARE (mq_timedsend (q, msg, sizeof (msg), 0, &ts), -1);
    TEST_VERIFY (errno == EINTR || errno == EOVERFLOW);
    support_delete_timer (timer);
  }

  mq_unlink (name);

  return 0;
}

#include <support/test-driver.c>
