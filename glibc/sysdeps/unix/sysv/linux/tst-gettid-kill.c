/* Verify the interaction of kill and thread groups.
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

/* This test demonstrates that a signal which is sent to a specified
   thread ID using the kill function is delivered to the entire thread
   group (as if it had been sent to the process group).  */

#include <errno.h>
#include <signal.h>
#include <support/check.h>
#include <support/xsignal.h>
#include <support/xthread.h>
#include <support/xunistd.h>

/* Signal set containing SIGUSR1.  */
static sigset_t sigusr1_set;

/* Used to synchronize the threads.  */
static pthread_barrier_t barrier;

/* TID of the thread to which the signal is sent.  */
static pid_t target_tid;

/* Thread which is expected to receive the SIGUSR1 signal.  */
static pthread_t signal_thread;

/* Pipe used to block and terminate the signal thread.  */
static int pipe_signal[2];

static volatile sig_atomic_t handler_tid;

static void
sigusr1_handler (int signo)
{
  TEST_COMPARE (signo, SIGUSR1);
  TEST_VERIFY (pthread_self () == signal_thread);
  TEST_COMPARE (handler_tid, 0);
  handler_tid = gettid ();
  TEST_VERIFY (handler_tid > 0);
  /* Ensure that the read system call in thread_read exits if the
     signal is delivered before the system call is invoked.  */
  char ch = 'X';
  xwrite (pipe_signal[1], &ch, 1);
}

/* Thread which calls pause without expecting it to return.  The TID
   of this thread is used as the target in the kill function call.  */
static void *
thread_pause_noreturn (void *closure)
{
  target_tid = gettid ();
  TEST_VERIFY (target_tid > 0);
  xpthread_barrier_wait (&barrier);
  pause ();
  FAIL_EXIT1 ("The pause function returned");
  return NULL;
}

/* Thread which is expected to receive the signal.  */
static void *
thread_read_signal (void *closure)
{
  xpthread_sigmask (SIG_UNBLOCK, &sigusr1_set, NULL);
  xpthread_barrier_wait (&barrier);
  TEST_VERIFY (target_tid > 0);
  TEST_VERIFY (gettid () != target_tid);
  char ch;
  ssize_t ret = read (pipe_signal[0], &ch, 1);
  if (ret == 1)
    /* The signal was delivered before we entered the read system
       call.  */
    TEST_COMPARE (ch, 'X');
  else
    {
      /* The signal was delivered while blocked in the read system
         call.  */
      TEST_COMPARE (ret, -1);
      TEST_COMPARE (errno, EINTR);
    }
  TEST_COMPARE (handler_tid, gettid ());
  return NULL;
}

static int
do_test (void)
{
  /* Block the SIGUSR1 signal in all threads.  */
  sigemptyset (&sigusr1_set);
  sigaddset (&sigusr1_set, SIGUSR1);
  xpthread_sigmask (SIG_BLOCK, &sigusr1_set, NULL);

  xsignal (SIGUSR1, sigusr1_handler);
  xpipe (pipe_signal);

  xpthread_barrier_init (&barrier, NULL, 3);

  pthread_t target_thread
    = xpthread_create (NULL, thread_pause_noreturn, NULL);
  signal_thread = xpthread_create (NULL, thread_read_signal, NULL);
  xpthread_barrier_wait (&barrier);

  /* Send the SIGUSR1 signal to the thread which has it blocked, and
     expect it to be delivered to the other thread.  */
  TEST_COMPARE (kill (target_tid, SIGUSR1), 0);

  xpthread_join (signal_thread);
  xpthread_cancel (target_thread);
  xpthread_join (target_thread);

  xpthread_barrier_destroy (&barrier);
  return 0;
}

#include <support/test-driver.c>
