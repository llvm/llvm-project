/* Smoke test for the tgkill system call.
   Copyright (C) 2019-2021 Free Software Foundation, Inc.
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
#include <support/check.h>
#include <support/namespace.h>
#include <support/xthread.h>
#include <unistd.h>

/* Number of times sigusr1_handler has been invoked.  */
static volatile sig_atomic_t signals_delivered;

/* Expected TID of the thread receiving the signal.  */
static pid_t expected_signal_tid;

static void
sigusr1_handler (int signo)
{
  TEST_COMPARE (expected_signal_tid, gettid ());
  ++signals_delivered;
}

struct pid_and_tid
{
  pid_t pid;
  pid_t tid;
};

/* Send signals from the subprocess which are not expected to be
   delivered.  There is no handler for SIGUSR2, so delivery will
   result in a test failure.  CLOSURE must point to a valid PID/TID
   combination that is still running.  */
static void
subprocess_no_tid_match (void *closure)
{
  struct pid_and_tid *ids = closure;
  TEST_COMPARE (tgkill (ids->pid, gettid (), SIGUSR2), -1);
  TEST_COMPARE (errno, ESRCH);

  TEST_COMPARE (tgkill (getpid (), ids->tid, SIGUSR2), -1);
  TEST_COMPARE (errno, ESRCH);

  TEST_COMPARE (tgkill (getppid (), gettid (), SIGUSR2), -1);
  TEST_COMPARE (errno, ESRCH);
}

/* Called from threadfunc below.  */
static void
subprocess (void *closure)
{
  int original_tid = expected_signal_tid;

  /* Do not expect that the folloing signals are delivered to the
     subprocess.  The parent process retains the original
     expected_signal_tid value.  */
  expected_signal_tid = 0;
  TEST_COMPARE (tgkill (getpid (), original_tid, SIGUSR1), -1);
  TEST_COMPARE (errno, ESRCH);
  TEST_COMPARE (tgkill (getppid (), gettid (), SIGUSR1), -1);
  TEST_COMPARE (errno, ESRCH);
  TEST_COMPARE (expected_signal_tid, 0);

  /* This call has the correct PID/TID combination and is therefore
     expected to suceed.  */
  TEST_COMPARE (tgkill (getppid (), original_tid, SIGUSR1), 0);
}

static void *
threadfunc (void *closure)
{
  TEST_VERIFY (gettid () != getpid ());
  expected_signal_tid = gettid ();
  TEST_COMPARE (tgkill (getpid (), gettid (), SIGUSR1), 0);
  TEST_COMPARE (signals_delivered, 1);
  signals_delivered = 0;

  support_isolate_in_subprocess (subprocess, NULL);

  /* Check that exactly one signal arrived from the subprocess.  */
  TEST_COMPARE (signals_delivered, 1);

  support_isolate_in_subprocess (subprocess_no_tid_match,
                                 &(struct pid_and_tid)
                                 {
                                   .pid = getpid (),
                                   .tid = gettid (),
                                 });

  support_isolate_in_subprocess (subprocess_no_tid_match,
                                 &(struct pid_and_tid)
                                 {
                                   .pid = getpid (),
                                   .tid = getpid (),
                                 });

  return NULL;
}

static int
do_test (void)
{
  TEST_VERIFY_EXIT (signal (SIGUSR1, sigusr1_handler) != SIG_ERR);

  expected_signal_tid = gettid ();
  TEST_COMPARE (gettid (), getpid ());
  TEST_COMPARE (tgkill (getpid (), gettid (), SIGUSR1), 0);
  TEST_COMPARE (signals_delivered, 1);
  signals_delivered = 0;

  xpthread_join (xpthread_create (NULL, threadfunc, NULL));

  TEST_VERIFY (signal (SIGUSR1, SIG_DFL) == sigusr1_handler);
  return 0;
}

#include <support/test-driver.c>
