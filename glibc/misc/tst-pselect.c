/* Copyright (C) 2006-2021 Free Software Foundation, Inc.
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
#include <support/check.h>
#include <support/support.h>
#include <support/xsignal.h>
#include <support/xunistd.h>
#include <support/xtime.h>
#include <stdlib.h>

static volatile int handler_called;

static void
handler (int sig)
{
  handler_called = 1;
}


static void
test_pselect_basic (void)
{
  struct sigaction sa;
  sa.sa_handler = handler;
  sa.sa_flags = 0;
  sigemptyset (&sa.sa_mask);

  xsigaction (SIGUSR1, &sa, NULL);

  sa.sa_handler = SIG_IGN;
  xsigaction (SIGCHLD, &sa, NULL);

  sigset_t ss_usr1;
  sigemptyset (&ss_usr1);
  sigaddset (&ss_usr1, SIGUSR1);
  TEST_COMPARE (sigprocmask (SIG_BLOCK, &ss_usr1, NULL), 0);

  int fds[2][2];
  xpipe (fds[0]);
  xpipe (fds[1]);

  fd_set rfds;
  FD_ZERO (&rfds);

  sigset_t ss;
  TEST_COMPARE (sigprocmask (SIG_SETMASK, NULL, &ss), 0);
  sigdelset (&ss, SIGUSR1);

  struct timespec to = { .tv_sec = 0, .tv_nsec = 500000000 };

  pid_t parent = getpid ();
  pid_t p = xfork ();
  if (p == 0)
    {
      xclose (fds[0][1]);
      xclose (fds[1][0]);

      FD_SET (fds[0][0], &rfds);

      int e;
      do
	{
	  if (getppid () != parent)
	    FAIL_EXIT1 ("getppid()=%d != parent=%d", getppid(), parent);

	  errno = 0;
	  e = pselect (fds[0][0] + 1, &rfds, NULL, NULL, &to, &ss);
	}
      while (e == 0);

      TEST_COMPARE (e, -1);
      TEST_COMPARE (errno, EINTR);

      TEMP_FAILURE_RETRY (write (fds[1][1], "foo", 3));

      exit (0);
    }

  xclose (fds[0][0]);
  xclose (fds[1][1]);

  FD_SET (fds[1][0], &rfds);

  TEST_COMPARE (kill (p, SIGUSR1), 0);

  int e = pselect (fds[1][0] + 1, &rfds, NULL, NULL, NULL, &ss);
  TEST_COMPARE (e, 1);
  TEST_VERIFY (FD_ISSET (fds[1][0], &rfds));
}

static void
test_pselect_large_timeout (void)
{
  support_create_timer (0, 100000000, false, NULL);

  int fds[2];
  xpipe (fds);

  fd_set rfds;
  FD_ZERO (&rfds);
  FD_SET (fds[0], &rfds);

  sigset_t ss;
  TEST_COMPARE (sigprocmask (SIG_SETMASK, NULL, &ss), 0);
  sigdelset (&ss, SIGALRM);

  struct timespec ts = { TYPE_MAXIMUM (time_t), 0 };

  TEST_COMPARE (pselect (fds[0] + 1, &rfds, NULL, NULL, &ts, &ss), -1);
  TEST_VERIFY (errno == EINTR || errno == EOVERFLOW);
}

static int
do_test (void)
{
  test_pselect_basic ();

  test_pselect_large_timeout ();

  return 0;
}

#include <support/test-driver.c>
