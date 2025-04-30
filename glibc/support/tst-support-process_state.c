/* Wait for process state tests.
   Copyright (C) 2020-2021 Free Software Foundation, Inc.
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

#include <stdio.h>
#include <sys/wait.h>
#include <unistd.h>
#include <errno.h>

#include <support/test-driver.h>
#include <support/process_state.h>
#include <support/check.h>
#include <support/xsignal.h>
#include <support/xunistd.h>

#ifndef WEXITED
# define WEXITED	0
#endif

static void
sigusr1_handler (int signo)
{
}

static void
test_child (void)
{
  xsignal (SIGUSR1, sigusr1_handler);

  raise (SIGSTOP);

  TEST_COMPARE (pause (), -1);
  TEST_COMPARE (errno, EINTR);

  while (1)
    asm ("");
}

static int
do_test (void)
{
  pid_t pid = xfork ();
  if (pid == 0)
    {
      test_child ();
      _exit (127);
    }

  /* Adding process_state_tracing_stop ('t') allows the test to work under
     trace programs such as ptrace.  */
  enum support_process_state stop_state = support_process_state_stopped
				    | support_process_state_tracing_stop;

  if (test_verbose)
    printf ("info: waiting pid %d, state_stopped/state_tracing_stop\n",
	    (int) pid);
  support_process_state_wait (pid, stop_state);

  if (kill (pid, SIGCONT) != 0)
    FAIL_RET ("kill (%d, SIGCONT): %m\n", pid);

  if (test_verbose)
    printf ("info: waiting pid %d, state_sleeping\n", (int) pid);
  support_process_state_wait (pid, support_process_state_sleeping);

  if (kill (pid, SIGUSR1) != 0)
    FAIL_RET ("kill (%d, SIGUSR1): %m\n", pid);

  if (test_verbose)
    printf ("info: waiting pid %d, state_running\n", (int) pid);
  support_process_state_wait (pid, support_process_state_running);

  if (kill (pid, SIGKILL) != 0)
    FAIL_RET ("kill (%d, SIGKILL): %m\n", pid);

  if (test_verbose)
    printf ("info: waiting pid %d, state_zombie\n", (int) pid);
  support_process_state_wait (pid, support_process_state_zombie);

  siginfo_t info;
  int r = waitid (P_PID, pid, &info, WEXITED);
  TEST_COMPARE (r, 0);
  TEST_COMPARE (info.si_signo, SIGCHLD);
  TEST_COMPARE (info.si_code, CLD_KILLED);
  TEST_COMPARE (info.si_status, SIGKILL);
  TEST_COMPARE (info.si_pid, pid);

  return 0;
}

#include <support/test-driver.c>
