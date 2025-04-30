/* Check that sigwait does not fail with EINTR (bug 22478).
   Copyright (C) 2017-2021 Free Software Foundation, Inc.
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
#include <signal.h>
#include <stdio.h>
#include <support/check.h>
#include <support/xunistd.h>
#include <time.h>
#include <unistd.h>

/* Handler for SIGUSR1.  */
static void
sigusr1_handler (int signo)
{
  TEST_VERIFY (signo == SIGUSR1);
}

/* Spawn a subprocess to send two signals: First SIGUSR1, then
   SIGUSR2.  Return the PID of the process.  */
static pid_t
signal_sender (void)
{
  pid_t pid = xfork ();
  if (pid == 0)
    {
      static const struct timespec delay = { 1, };
      if (nanosleep (&delay, NULL) != 0)
        FAIL_EXIT1 ("nanosleep: %m");
      if (kill (getppid (), SIGUSR1) != 0)
        FAIL_EXIT1 ("kill (SIGUSR1): %m");
      if (nanosleep (&delay, NULL) != 0)
        FAIL_EXIT1 ("nanosleep: %m");
      if (kill (getppid (), SIGUSR2) != 0)
        FAIL_EXIT1 ("kill (SIGUSR2): %m");
      _exit (0);
    }
  return pid;
}

static int
do_test (void)
{
  if (signal (SIGUSR1, sigusr1_handler) == SIG_ERR)
    FAIL_EXIT1 ("signal (SIGUSR1): %m\n");

  sigset_t sigs;
  sigemptyset (&sigs);
  sigaddset (&sigs, SIGUSR2);
  if (sigprocmask (SIG_BLOCK, &sigs, NULL) != 0)
    FAIL_EXIT1 ("sigprocmask (SIGBLOCK, SIGUSR2): %m");
  pid_t pid = signal_sender ();
  int signo = 0;
  int ret = sigwait (&sigs, &signo);
  if (ret != 0)
    {
      support_record_failure ();
      errno = ret;
      printf ("error: sigwait failed: %m (%d)\n", ret);
    }
  TEST_VERIFY (signo == SIGUSR2);

  int status;
  xwaitpid (pid, &status, 0);
  TEST_VERIFY (status == 0);

  return 0;
}

#include <support/test-driver.c>
