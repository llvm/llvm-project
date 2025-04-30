/* Test case for async-signal-safe fork (with respect to malloc).
   Copyright (C) 2016-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public License as
   published by the Free Software Foundation; either version 2.1 of the
   License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; see the file COPYING.LIB.  If
   not, see <https://www.gnu.org/licenses/>.  */

/* This test will fail if the process is multi-threaded because we
   only have an async-signal-safe fork in the single-threaded case
   (where we skip acquiring the malloc heap locks).

   This test only checks async-signal-safety with regards to malloc;
   other, more rarely-used glibc subsystems could have locks which
   still make fork unsafe, even in single-threaded processes.  */

#include <errno.h>
#include <sched.h>
#include <signal.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>
#include <array_length.h>
#include <support/check.h>
#include <support/support.h>
#include <support/xthread.h>
#include <support/xunistd.h>

/* How many malloc objects to keep arond.  */
enum { malloc_objects = 1009 };

/* The maximum size of an object.  */
enum { malloc_maximum_size = 70000 };

/* How many iterations the test performs before exiting.  */
enum { iterations = 10000 };

/* Barrier for synchronization with the processes sending SIGUSR1
   signals, to make it more likely that the signals arrive during a
   fork/free/malloc call.  */
static struct { pthread_barrier_t barrier; } *shared;

/* Set to 1 if SIGUSR1 is received.  Used to detect a signal during
   fork/free/malloc.  */
static volatile sig_atomic_t sigusr1_received;

/* Periodically set to 1, to indicate that the process is making
   progress.  Checked by liveness_signal_handler.  */
static volatile sig_atomic_t progress_indicator = 1;

/* Set to 1 if an error occurs in the signal handler.  */
static volatile sig_atomic_t error_indicator = 0;

static void
sigusr1_handler (int signo)
{
  sigusr1_received = 1;

  /* Perform a fork with a trivial subprocess.  */
  pid_t pid = fork ();
  if (pid == -1)
    {
      write_message ("error: fork\n");
      error_indicator = 1;
      return;
    }
  if (pid == 0)
    _exit (0);
  int status;
  int ret = TEMP_FAILURE_RETRY (waitpid (pid, &status, 0));
  if (ret < 0)
    {
      write_message ("error: waitpid\n");
      error_indicator = 1;
      return;
    }
  if (status != 0)
    {
      write_message ("error: unexpected exit status from subprocess\n");
      error_indicator = 1;
      return;
    }
}

static void
liveness_signal_handler (int signo)
{
  if (progress_indicator)
    progress_indicator = 0;
  else
    write_message ("warning: process seems to be stuck\n");
}

/* Send SIGNO to the parent process.  If SLEEP, wait a second between
   signals, otherwise use barriers to delay sending signals.  */
static void
__attribute__ ((noreturn))
signal_sender (int signo, bool sleep)
{
  pid_t target = getppid ();
  while (true)
    {
      if (!sleep)
        xpthread_barrier_wait (&shared->barrier);
      if (kill (target, signo) != 0)
        {
          dprintf (STDOUT_FILENO, "error: kill: %m\n");
          abort ();
        }
      if (sleep)
        usleep (1 * 1000 * 1000);
      else
        xpthread_barrier_wait (&shared->barrier);
    }
}

/* Children processes.  */
static pid_t sigusr1_sender_pids[5] = { 0 };
static pid_t sigusr2_sender_pid = 0;

static void
kill_children (void)
{
  for (size_t i = 0; i < array_length (sigusr1_sender_pids); ++i)
    if (sigusr1_sender_pids[i] > 0)
      kill (sigusr1_sender_pids[i], SIGKILL);
  if (sigusr2_sender_pid > 0)
    kill (sigusr2_sender_pid, SIGKILL);
}

static int
do_test (void)
{
  atexit (kill_children);

  /* shared->barrier is intialized along with sigusr1_sender_pids
     below.  */
  shared = support_shared_allocate (sizeof (*shared));

  struct sigaction action =
    {
      .sa_handler = sigusr1_handler,
    };
  sigemptyset (&action.sa_mask);

  if (sigaction (SIGUSR1, &action, NULL) != 0)
    {
      printf ("error: sigaction: %m");
      return 1;
    }

  action.sa_handler = liveness_signal_handler;
  if (sigaction (SIGUSR2, &action, NULL) != 0)
    {
      printf ("error: sigaction: %m");
      return 1;
    }

  sigusr2_sender_pid = xfork ();
  if (sigusr2_sender_pid == 0)
    signal_sender (SIGUSR2, true);

  /* Send SIGUSR1 signals from several processes.  Hopefully, one
     signal will hit one of the ciritical functions.  Use a barrier to
     avoid sending signals while not running fork/free/malloc.  */
  {
    pthread_barrierattr_t attr;
    xpthread_barrierattr_init (&attr);
    xpthread_barrierattr_setpshared (&attr, PTHREAD_PROCESS_SHARED);
    xpthread_barrier_init (&shared->barrier, &attr,
                           array_length (sigusr1_sender_pids) + 1);
    xpthread_barrierattr_destroy (&attr);
  }
  for (size_t i = 0; i < array_length (sigusr1_sender_pids); ++i)
    {
      sigusr1_sender_pids[i] = xfork ();
      if (sigusr1_sender_pids[i] == 0)
        signal_sender (SIGUSR1, false);
    }

  void *objects[malloc_objects] = {};
  unsigned int fork_signals = 0;
  unsigned int free_signals = 0;
  unsigned int malloc_signals = 0;
  unsigned seed = 1;
  for (int i = 0; i < iterations; ++i)
    {
      progress_indicator = 1;
      int slot = rand_r (&seed) % malloc_objects;
      size_t size = rand_r (&seed) % malloc_maximum_size;

      /* Occasionally do a fork first, to catch deadlocks there as
         well (see bug 24161).  */
      bool do_fork = (rand_r (&seed) % 7) == 0;

      xpthread_barrier_wait (&shared->barrier);
      if (do_fork)
        {
          sigusr1_received = 0;
          pid_t pid = xfork ();
          if (sigusr1_received)
            ++fork_signals;
          if (pid == 0)
            _exit (0);
          int status;
          int ret = TEMP_FAILURE_RETRY (waitpid (pid, &status, 0));
          if (ret < 0)
            FAIL_EXIT1 ("waitpid: %m");
          TEST_COMPARE (status, 0);
        }
      sigusr1_received = 0;
      free (objects[slot]);
      if (sigusr1_received)
        ++free_signals;
      sigusr1_received = 0;
      objects[slot] = malloc (size);
      if (sigusr1_received)
        ++malloc_signals;
      xpthread_barrier_wait (&shared->barrier);

      if (objects[slot] == NULL || error_indicator != 0)
        {
          printf ("error: malloc: %m\n");
          for (size_t i = 0; i < array_length (sigusr1_sender_pids); ++i)
            kill (sigusr1_sender_pids[i], SIGKILL);
          kill (sigusr2_sender_pid, SIGKILL);
          return 1;
        }
    }

  /* Clean up allocations.  */
  for (int slot = 0; slot < malloc_objects; ++slot)
    free (objects[slot]);

  printf ("info: signals received during fork: %u\n", fork_signals);
  printf ("info: signals received during free: %u\n", free_signals);
  printf ("info: signals received during malloc: %u\n", malloc_signals);

  /* Do not destroy the barrier because of the SIGKILL above, which
     may have left the barrier in an inconsistent state.  */
  support_shared_free (shared);

  return 0;
}

#define TIMEOUT 100
#include <support/test-driver.c>
