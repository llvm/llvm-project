/* Testcase checks, if setcontext(), swapcontext() restores signal-mask
   and if pending signals are delivered after those calls.
   Copyright (C) 2015-2021 Free Software Foundation, Inc.
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
#include <stdlib.h>
#include <sys/types.h>
#include <signal.h>
#include <ucontext.h>
#include <unistd.h>

volatile int global;
volatile sig_atomic_t handlerCalled;

static void
check (const char *funcName)
{
  sigset_t set;

  /* check if SIGUSR2 is unblocked after setcontext-call.  */
  sigprocmask (SIG_BLOCK, NULL, &set);

  if (sigismember (&set, SIGUSR2) != 0)
    {
      printf ("FAIL: SIGUSR2 is blocked after %s.\n", funcName);
      exit (1);
    }

  if (sigismember (&set, SIGUSR1) != 1)
    {
      printf ("FAIL: SIGUSR1 is not blocked after %s.\n", funcName);
      exit (1);
    }
}

static void
signalmask (int how, int signum)
{
  sigset_t set;
  sigemptyset (&set);
  sigaddset (&set, signum);
  if (sigprocmask (how, &set, NULL) != 0)
    {
      printf ("FAIL: sigprocmaks (%d, %d, NULL): %m\n", how, signum);
      exit (1);
    }
}

static void
signalpending (int signum, const char *msg)
{
  sigset_t set;
  sigemptyset (&set);
  if (sigpending (&set) != 0)
    {
      printf ("FAIL: sigpending: %m\n");
      exit (1);
    }
  if (sigismember (&set, SIGUSR2) != 1)
    {
      printf ("FAIL: Signal %d is not pending %s\n", signum, msg);
      exit (1);
    }
}

static void
handler (int __attribute__ ((unused)) signum)
{
  handlerCalled ++;
}

static int
do_test (void)
{
  ucontext_t ctx, oldctx;
  struct sigaction action;
  pid_t pid;

  pid = getpid ();

  /* unblock SIGUSR2 */
  signalmask (SIG_UNBLOCK, SIGUSR2);

  /* block SIGUSR1 */
  signalmask (SIG_BLOCK, SIGUSR1);

  /* register handler for SIGUSR2  */
  action.sa_flags = 0;
  action.sa_handler = handler;
  sigemptyset (&action.sa_mask);
  sigaction (SIGUSR2, &action, NULL);

  if (getcontext (&ctx) != 0)
    {
      printf ("FAIL: getcontext: %m\n");
      exit (1);
    }

  global++;

  if (global == 1)
    {
      puts ("after getcontext");

      /* block SIGUSR2  */
      signalmask (SIG_BLOCK, SIGUSR2);

      /* send SIGUSR2 to me  */
      handlerCalled = 0;
      kill (pid, SIGUSR2);

      /* was SIGUSR2 handler called?  */
      if (handlerCalled != 0)
	{
	  puts ("FAIL: signal handler was called, but signal was blocked.");
	  exit (1);
	}

      /* is SIGUSR2 pending?  */
      signalpending (SIGUSR2, "before setcontext");

      /* SIGUSR2 will be unblocked by setcontext-call.  */
      if (setcontext (&ctx) != 0)
	{
	  printf ("FAIL: setcontext: %m\n");
	  exit (1);
	}
    }
  else if (global == 2)
    {
      puts ("after setcontext");

      /* check SIGUSR1/2  */
      check ("setcontext");

      /* was SIGUSR2 handler called? */
      if (handlerCalled != 1)
	{
	  puts ("FAIL: signal handler was not called after setcontext.");
	  exit (1);
	}

      /* block SIGUSR2 */
      signalmask (SIG_BLOCK, SIGUSR2);

      /* send SIGUSR2 to me  */
      handlerCalled = 0;
      kill (pid, SIGUSR2);

      /* was SIGUSR2 handler called?  */
      if (handlerCalled != 0)
	{
	  puts ("FAIL: signal handler was called, but signal was blocked.");
	  exit (1);
	}

      /* is SIGUSR2 pending?  */
      signalpending (SIGUSR2, "before swapcontext");

      if (swapcontext (&oldctx, &ctx) != 0)
	{
	  printf ("FAIL: swapcontext: %m\n");
	  exit (1);
	}

      puts ("after returned from swapcontext");

      if (global != 3)
	{
	  puts ("FAIL: returned from swapcontext without ctx-context called.");
	  exit (1);
	}

      puts ("test succeeded");
      return 0;
    }
  else if ( global != 3 )
    {
      puts ("FAIL: 'global' not incremented three times");
      exit (1);
    }

  puts ("after swapcontext");
  /* check SIGUSR1/2  */
  check ("swapcontext");

  /* was SIGUSR2 handler called? */
  if (handlerCalled != 1)
    {
      puts ("FAIL: signal handler was not called after swapcontext.");
      exit (1);
    }

  /* check sigmask in old context of swapcontext-call  */
  if (sigismember (&oldctx.uc_sigmask, SIGUSR2) != 1)
    {
      puts ("FAIL: SIGUSR2 is not blocked in oldctx.uc_sigmask.");
      exit (1);
    }

  if (sigismember (&oldctx.uc_sigmask, SIGUSR1) != 1)
    {
      puts ("FAIL: SIGUSR1 is not blocked in oldctx.uc_sigmaks.");
      exit (1);
    }

  /* change to old context, which was gathered by swapcontext() call.  */
  setcontext (&oldctx);

  puts ("FAIL: returned from setcontext (&oldctx)");
  exit (1);
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
