/* Test if CLONE_VM does not change pthread pid/tid field (BZ #19957)
   Copyright (C) 2016-2021 Free Software Foundation, Inc.
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

#include <sched.h>
#include <signal.h>
#include <string.h>
#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>
#include <stddef.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <sys/syscall.h>

#include <stackinfo.h>  /* For _STACK_GROWS_{UP,DOWN}.  */

#include <support/check.h>

static int sig;
static int pipefd[2];

static int
f (void *a)
{
  close (pipefd[0]);

  pid_t ppid = getppid ();
  pid_t pid = getpid ();
  pid_t tid = syscall (__NR_gettid);

  if (write (pipefd[1], &ppid, sizeof ppid) != sizeof (ppid))
    FAIL_EXIT1 ("write ppid failed\n");
  if (write (pipefd[1], &pid, sizeof pid) != sizeof (pid))
    FAIL_EXIT1 ("write pid failed\n");
  if (write (pipefd[1], &tid, sizeof tid) != sizeof (tid))
    FAIL_EXIT1 ("write tid failed\n");

  return 0;
}


static int
do_test (void)
{
  sig = SIGRTMIN;
  sigset_t ss;
  sigemptyset (&ss);
  sigaddset (&ss, sig);
  if (sigprocmask (SIG_BLOCK, &ss, NULL) != 0)
    FAIL_EXIT1 ("sigprocmask failed: %m");

  if (pipe2 (pipefd, O_CLOEXEC))
    FAIL_EXIT1 ("pipe failed: %m");

  int clone_flags = 0;
#ifdef __ia64__
  extern int __clone2 (int (*__fn) (void *__arg), void *__child_stack_base,
		       size_t __child_stack_size, int __flags,
		       void *__arg, ...);
  char st[256 * 1024] __attribute__ ((aligned));
  pid_t p = __clone2 (f, st, sizeof (st), clone_flags, 0);
#else
  char st[128 * 1024] __attribute__ ((aligned));
#if _STACK_GROWS_DOWN
  pid_t p = clone (f, st + sizeof (st), clone_flags, 0);
#elif _STACK_GROWS_UP
  pid_t p = clone (f, st, clone_flags, 0);
#else
#error "Define either _STACK_GROWS_DOWN or _STACK_GROWS_UP"
#endif
#endif

  close (pipefd[1]);

  if (p == -1)
    FAIL_EXIT1("clone failed: %m");

  pid_t ppid, pid, tid;
  if (read (pipefd[0], &ppid, sizeof pid) != sizeof pid)
    {
      kill (p, SIGKILL);
      FAIL_EXIT1 ("read ppid failed: %m");
    }
  if (read (pipefd[0], &pid, sizeof pid) != sizeof pid)
    {
      kill (p, SIGKILL);
      FAIL_EXIT1 ("read pid failed: %m");
    }
  if (read (pipefd[0], &tid, sizeof tid) != sizeof tid)
    {
      kill (p, SIGKILL);
      FAIL_EXIT1 ("read tid failed: %m");
    }

  close (pipefd[0]);

  int ret = 0;

  pid_t own_pid = getpid ();
  pid_t own_tid = syscall (__NR_gettid);

  /* Some sanity checks for clone syscall: returned ppid should be current
     pid and both returned tid/pid should be different from current one.  */
  if ((ppid != own_pid) || (pid == own_pid) || (tid == own_tid))
    FAIL_RET ("ppid=%i pid=%i tid=%i | own_pid=%i own_tid=%i",
	      (int)ppid, (int)pid, (int)tid, (int)own_pid, (int)own_tid);

  int e;
  if (waitpid (p, &e, __WCLONE) != p)
    {
      kill (p, SIGKILL);
      FAIL_EXIT1 ("waitpid failed");
    }
  if (!WIFEXITED (e))
    {
      if (WIFSIGNALED (e))
	printf ("died from signal %s\n", strsignal (WTERMSIG (e)));
      else
	puts ("did not terminate correctly");
      exit (EXIT_FAILURE);
    }
  if (WEXITSTATUS (e) != 0)
    FAIL_EXIT1 ("exit code %d", WEXITSTATUS (e));

  return ret;
}

#include <support/test-driver.c>
