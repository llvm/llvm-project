/* Verify that the parent pid is unchanged by __clone_internal.
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

#include <sched.h>
#include <signal.h>
#include <string.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <clone_internal.h>
#include <support/xunistd.h>

#ifndef TEST_CLONE_FLAGS
#define TEST_CLONE_FLAGS 0
#endif

static int sig;

static int
f (void *a)
{
  puts ("in f");
  union sigval sival;
  sival.sival_int = getpid ();
  printf ("pid = %d\n", sival.sival_int);
  if (sigqueue (getppid (), sig, sival) != 0)
    return 1;
  return 0;
}


static int
do_test (void)
{
  int mypid = getpid ();

  sig = SIGRTMIN;
  sigset_t ss;
  sigemptyset (&ss);
  sigaddset (&ss, sig);
  if (sigprocmask (SIG_BLOCK, &ss, NULL) != 0)
    {
      printf ("sigprocmask failed: %m\n");
      return 1;
    }

#ifdef __ia64__
# define STACK_SIZE 256 * 1024
#else
# define STACK_SIZE 128 * 1024
#endif
  char st[STACK_SIZE] __attribute__ ((aligned));
  struct clone_args clone_args =
    {
      .flags = TEST_CLONE_FLAGS & ~CSIGNAL,
      .exit_signal = TEST_CLONE_FLAGS & CSIGNAL,
      .stack = (uintptr_t) st,
      .stack_size = sizeof (st),
    };
  pid_t p = __clone_internal (&clone_args, f, 0);
  if (p == -1)
    {
      printf("clone failed: %m\n");
      return 1;
    }
  printf ("new thread: %d\n", (int) p);

  siginfo_t si;
  do
    if (sigwaitinfo (&ss, &si) < 0)
      {
	printf("sigwaitinfo failed: %m\n");
	kill (p, SIGKILL);
	return 1;
      }
  while  (si.si_signo != sig || si.si_code != SI_QUEUE);

  int e;
  xwaitpid (p, &e, __WCLONE);
  if (!WIFEXITED (e))
    {
      if (WIFSIGNALED (e))
	printf ("died from signal %s\n", strsignal (WTERMSIG (e)));
      else
	puts ("did not terminate correctly");
      return 1;
    }
  if (WEXITSTATUS (e) != 0)
    {
      printf ("exit code %d\n", WEXITSTATUS (e));
      return 1;
    }

  if (si.si_int != (int) p)
    {
      printf ("expected PID %d, got si_int %d\n", (int) p, si.si_int);
      kill (p, SIGKILL);
      return 1;
    }

  if (si.si_pid != p)
    {
      printf ("expected PID %d, got si_pid %d\n", (int) p, (int) si.si_pid);
      kill (p, SIGKILL);
      return 1;
    }

  if (getpid () != mypid)
    {
      puts ("my PID changed");
      return 1;
    }

  return 0;
}

#include <support/test-driver.c>
