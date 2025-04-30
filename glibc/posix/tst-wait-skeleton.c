/* Test framework for wait3 and wait4.
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

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/resource.h>
#include <signal.h>
#include <time.h>
#include <stdatomic.h>
#include <stdbool.h>

#include <support/xsignal.h>
#include <support/xunistd.h>
#include <support/check.h>
#include <support/process_state.h>

static void
test_child (void)
{
  /* First thing, we stop ourselves.  */
  raise (SIGSTOP);

  /* Hey, we got continued!  */
  while (1)
    pause ();
}

#ifndef WEXITED
# define WEXITED        0
# define WCONTINUED     0
# define WSTOPPED       WUNTRACED
#endif

/* Set with only SIGCHLD on do_test_waitid.  */
static sigset_t chldset;

#ifdef SA_SIGINFO
static void
sigchld (int signo, siginfo_t *info, void *ctx)
{
}
#endif

static void
check_sigchld (int code, int status, pid_t pid)
{
#ifdef SA_SIGINFO
  siginfo_t siginfo;
  TEST_COMPARE (sigwaitinfo (&chldset, &siginfo), SIGCHLD);

  TEST_COMPARE (siginfo.si_signo, SIGCHLD);
  TEST_COMPARE (siginfo.si_code, code);
  TEST_COMPARE (siginfo.si_status, status);
  TEST_COMPARE (siginfo.si_pid, pid);
#endif
}

static int
do_test_wait (pid_t pid)
{
  /* Adding process_state_tracing_stop ('t') allows the test to work under
     trace programs such as ptrace.  */
  enum support_process_state stop_state = support_process_state_stopped
                                          | support_process_state_tracing_stop;

  support_process_state_wait (pid, stop_state);

  check_sigchld (CLD_STOPPED, SIGSTOP, pid);

  pid_t ret;
  int wstatus;
  struct rusage rusage;

  ret = WAIT_CALL (pid, &wstatus, WUNTRACED|WCONTINUED|WNOHANG, NULL);
  if (ret == -1 && errno == ENOTSUP)
    FAIL_RET ("waitid WNOHANG on stopped: %m");
  TEST_COMPARE (ret, pid);
  TEST_VERIFY (WIFSTOPPED (wstatus));

  /* Issue again but with struct rusage input.  */
  ret = WAIT_CALL (pid, &wstatus, WUNTRACED|WCONTINUED|WNOHANG, &rusage);
  /* With WNOHANG and WUNTRACED, if the children has not changes its state
     since previous call the expected result it 0.  */
  TEST_COMPARE (ret, 0);

  /* Some sanity tests to check if 'wtatus' and 'rusage' possible
     input values.  */
  ret = WAIT_CALL (pid, NULL, WUNTRACED|WCONTINUED|WNOHANG, &rusage);
  TEST_COMPARE (ret, 0);
  ret = WAIT_CALL (pid, NULL, WUNTRACED|WCONTINUED|WNOHANG, NULL);
  TEST_COMPARE (ret, 0);

  if (kill (pid, SIGCONT) != 0)
    FAIL_RET ("kill (%d, SIGCONT): %m\n", pid);

  /* Wait for the child to have continued.  */
  support_process_state_wait (pid, support_process_state_sleeping);

#if WCONTINUED != 0
  check_sigchld (CLD_CONTINUED, SIGCONT, pid);

  ret = WAIT_CALL (pid, &wstatus, WCONTINUED|WNOHANG, NULL);
  TEST_COMPARE (ret, pid);
  TEST_VERIFY (WIFCONTINUED (wstatus));

  /* Issue again but with struct rusage input.  */
  ret = WAIT_CALL (pid, &wstatus, WUNTRACED|WCONTINUED|WNOHANG, &rusage);
  /* With WNOHANG and WUNTRACED, if the children has not changes its state
     since previous call the expected result it 0.  */
  TEST_COMPARE (ret, 0);

  /* Now stop him again and test waitpid with WCONTINUED.  */
  if (kill (pid, SIGSTOP) != 0)
    FAIL_RET ("kill (%d, SIGSTOP): %m\n", pid);

  /* Wait the child stop.  The waitid call below will block until it has
     stopped, but if we are real quick and enter the waitid system call
     before the SIGCHLD has been generated, then it will be discarded and
     never delivered.  */
  support_process_state_wait (pid, stop_state);

  ret = WAIT_CALL (pid, &wstatus, WUNTRACED|WNOHANG, &rusage);
  TEST_COMPARE (ret, pid);

  check_sigchld (CLD_STOPPED, SIGSTOP, pid);

  if (kill (pid, SIGCONT) != 0)
    FAIL_RET ("kill (%d, SIGCONT): %m\n", pid);

  /* Wait for the child to have continued.  */
  support_process_state_wait (pid, support_process_state_sleeping);

  check_sigchld (CLD_CONTINUED, SIGCONT, pid);

  ret = WAIT_CALL (pid, &wstatus, WCONTINUED|WNOHANG, NULL);
  TEST_COMPARE (ret, pid);
  TEST_VERIFY (WIFCONTINUED (wstatus));
#endif

  /* Die, child, die!  */
  if (kill (pid, SIGKILL) != 0)
    FAIL_RET ("kill (%d, SIGKILL): %m\n", pid);

  support_process_state_wait (pid, support_process_state_zombie);

  ret = WAIT_CALL (pid, &wstatus, 0, &rusage);
  TEST_COMPARE (ret, pid);
  TEST_VERIFY (WIFSIGNALED (wstatus));
  TEST_VERIFY (WTERMSIG (wstatus) == SIGKILL);

  check_sigchld (CLD_KILLED, SIGKILL, pid);

  return 0;
}

static int
do_test (void)
{
#ifdef SA_SIGINFO
  {
    struct sigaction sa;
    sa.sa_flags = SA_SIGINFO | SA_RESTART;
    sa.sa_sigaction = sigchld;
    sigemptyset (&sa.sa_mask);
    xsigaction (SIGCHLD, &sa, NULL);
  }
#endif

  sigemptyset (&chldset);
  sigaddset (&chldset, SIGCHLD);

  /* The SIGCHLD shall has blocked at the time of the call to sigwait;
     otherwise, the behavior is undefined.  */
  sigprocmask (SIG_BLOCK, &chldset, NULL);

  pid_t pid = xfork ();
  if (pid == 0)
    {
      test_child ();
      _exit (127);
    }

  do_test_wait (pid);

  xsignal (SIGCHLD, SIG_IGN);
  kill (pid, SIGKILL);          /* Make sure it's dead if we bailed early.  */

  return 0;
}

#include <support/test-driver.c>
