/* Tests for waitid.
   Copyright (C) 2004-2021 Free Software Foundation, Inc.
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
#include <signal.h>
#include <time.h>
#include <stdatomic.h>
#include <stdbool.h>

#include <support/xsignal.h>
#include <support/xunistd.h>
#include <support/check.h>
#include <support/process_state.h>

static void
test_child (bool setgroup)
{
  if (setgroup)
    TEST_COMPARE (setpgid (0, 0), 0);

  /* First thing, we stop ourselves.  */
  raise (SIGSTOP);

  /* Hey, we got continued!  */
  while (1)
    pause ();
}

#ifndef WEXITED
# define WEXITED	0
# define WCONTINUED	0
# define WSTOPPED	WUNTRACED
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
do_test_waitd_common (idtype_t type, pid_t pid)
{
  /* Adding process_state_tracing_stop ('t') allows the test to work under
     trace programs such as ptrace.  */
  enum support_process_state stop_state = support_process_state_stopped
					  | support_process_state_tracing_stop;

  support_process_state_wait (pid, stop_state);

  check_sigchld (CLD_STOPPED, SIGSTOP, pid);

  /* Now try a wait that should not succeed.  */
  siginfo_t info;
  int fail;

  info.si_signo = 0;		/* A successful call sets it to SIGCHLD.  */
  fail = waitid (P_PID, pid, &info, WEXITED|WCONTINUED|WNOHANG);
  if (fail == -1 && errno == ENOTSUP)
    FAIL_RET ("waitid WNOHANG on stopped: %m");
  TEST_COMPARE (fail, 0);
  TEST_COMPARE (info.si_signo, 0);

  /* Next the wait that should succeed right away.  */
  info.si_signo = 0;		/* A successful call sets it to SIGCHLD.  */
  info.si_pid = -1;
  info.si_status = -1;
  fail = waitid (P_PID, pid, &info, WSTOPPED|WNOHANG);
  if (fail == -1 && errno == ENOTSUP)
    FAIL_RET ("waitid WNOHANG on stopped: %m");
  TEST_COMPARE (fail, 0);
  TEST_COMPARE (info.si_signo, SIGCHLD);
  TEST_COMPARE (info.si_code, CLD_STOPPED);
  TEST_COMPARE (info.si_status, SIGSTOP);
  TEST_COMPARE (info.si_pid, pid);

  if (kill (pid, SIGCONT) != 0)
    FAIL_RET ("kill (%d, SIGCONT): %m\n", pid);

  /* Wait for the child to have continued.  */
  support_process_state_wait (pid, support_process_state_sleeping);

#if WCONTINUED != 0
  check_sigchld (CLD_CONTINUED, SIGCONT, pid);

  info.si_signo = 0;		/* A successful call sets it to SIGCHLD.  */
  info.si_pid = -1;
  info.si_status = -1;
  fail = waitid (P_PID, pid, &info, WCONTINUED|WNOWAIT);
  if (fail == -1 && errno == ENOTSUP)
    FAIL_RET ("waitid WCONTINUED|WNOWAIT on continued: %m");
  TEST_COMPARE (fail, 0);
  TEST_COMPARE (info.si_signo, SIGCHLD);
  TEST_COMPARE (info.si_code, CLD_CONTINUED);
  TEST_COMPARE (info.si_status, SIGCONT);
  TEST_COMPARE (info.si_pid, pid);

  /* That should leave the CLD_CONTINUED state waiting to be seen again.  */
  info.si_signo = 0;		/* A successful call sets it to SIGCHLD.  */
  info.si_pid = -1;
  info.si_status = -1;
  fail = waitid (P_PID, pid, &info, WCONTINUED);
  if (fail == -1 && errno == ENOTSUP)
    FAIL_RET ("waitid WCONTINUED on continued: %m");
  TEST_COMPARE (fail, 0);
  TEST_COMPARE (info.si_signo, SIGCHLD);
  TEST_COMPARE (info.si_code, CLD_CONTINUED);
  TEST_COMPARE (info.si_status, SIGCONT);
  TEST_COMPARE (info.si_pid, pid);

  /* Now try a wait that should not succeed.  */
  info.si_signo = 0;		/* A successful call sets it to SIGCHLD.  */
  fail = waitid (P_PID, pid, &info, WCONTINUED|WNOHANG);
  if (fail == -1 && errno == ENOTSUP)
    FAIL_RET ("waitid WCONTINUED|WNOHANG on waited continued: %m");
  TEST_COMPARE (fail, 0);
  TEST_COMPARE (info.si_signo, 0);

  /* Now stop him again and test waitpid with WCONTINUED.  */
  if (kill (pid, SIGSTOP) != 0)
    FAIL_RET ("kill (%d, SIGSTOP): %m\n", pid);

  /* Wait the child stop.  The waitid call below will block until it has
     stopped, but if we are real quick and enter the waitid system call
     before the SIGCHLD has been generated, then it will be discarded and
     never delivered.  */
  support_process_state_wait (pid, stop_state);

  fail = waitid (type, pid, &info, WEXITED|WSTOPPED);
  TEST_COMPARE (fail, 0);
  TEST_COMPARE (info.si_signo, SIGCHLD);
  TEST_COMPARE (info.si_code, CLD_STOPPED);
  TEST_COMPARE (info.si_status, SIGSTOP);
  TEST_COMPARE (info.si_pid, pid);

  check_sigchld (CLD_STOPPED, SIGSTOP, pid);

  if (kill (pid, SIGCONT) != 0)
    FAIL_RET ("kill (%d, SIGCONT): %m\n", pid);

  /* Wait for the child to have continued.  */
  support_process_state_wait (pid, support_process_state_sleeping);

  check_sigchld (CLD_CONTINUED, SIGCONT, pid);

  fail = waitid (type, pid, &info, WCONTINUED);
  TEST_COMPARE (fail, 0);
  TEST_COMPARE (info.si_signo, SIGCHLD);
  TEST_COMPARE (info.si_code, CLD_CONTINUED);
  TEST_COMPARE (info.si_status, SIGCONT);
  TEST_COMPARE (info.si_pid, pid);
#endif

  /* Die, child, die!  */
  if (kill (pid, SIGKILL) != 0)
    FAIL_RET ("kill (%d, SIGKILL): %m\n", pid);

#ifdef WNOWAIT
  info.si_signo = 0;		/* A successful call sets it to SIGCHLD.  */
  info.si_pid = -1;
  info.si_status = -1;
  fail = waitid (type, pid, &info, WEXITED|WNOWAIT);
  if (fail == -1 && errno == ENOTSUP)
    FAIL_RET ("waitid WNOHANG on killed: %m");
  TEST_COMPARE (fail, 0);
  TEST_COMPARE (info.si_signo, SIGCHLD);
  TEST_COMPARE (info.si_code, CLD_KILLED);
  TEST_COMPARE (info.si_status, SIGKILL);
  TEST_COMPARE (info.si_pid, pid);
#else
  support_process_state_wait (pid, support_process_state_zombie);
#endif
  check_sigchld (CLD_KILLED, SIGKILL, pid);

  info.si_signo = 0;		/* A successful call sets it to SIGCHLD.  */
  info.si_pid = -1;
  info.si_status = -1;
  fail = waitid (type, pid, &info, WEXITED | WNOHANG);
  TEST_COMPARE (fail, 0);
  TEST_COMPARE (info.si_signo, SIGCHLD);
  TEST_COMPARE (info.si_code, CLD_KILLED);
  TEST_COMPARE (info.si_status, SIGKILL);
  TEST_COMPARE (info.si_pid, pid);

  fail = waitid (P_PID, pid, &info, WEXITED);
  TEST_COMPARE (fail, -1);
  TEST_COMPARE (errno, ECHILD);

  return 0;
}

static int
do_test_waitid (idtype_t type)
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
      test_child (type == P_PGID || type == P_ALL);
      _exit (127);
    }

  int ret = do_test_waitd_common (type, pid);

  xsignal (SIGCHLD, SIG_IGN);
  kill (pid, SIGKILL);		/* Make sure it's dead if we bailed early.  */
  return ret;
}

static int
do_test (void)
{
  int ret = 0;

  ret |= do_test_waitid (P_PID);
  ret |= do_test_waitid (P_PGID);
  ret |= do_test_waitid (P_ALL);

  return ret;
}

#include <support/test-driver.c>
