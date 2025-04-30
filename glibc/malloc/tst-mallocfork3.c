/* Test case for async-signal-safe _Fork (with respect to malloc).
   Copyright (C) 2021 Free Software Foundation, Inc.
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

/* This test is similar to tst-mallocfork2.c, but specifically stress
   the async-signal-safeness of _Fork on multithread environment.  */

#include <array_length.h>
#include <errno.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <support/check.h>
#include <support/support.h>
#include <support/xsignal.h>
#include <support/xthread.h>
#include <support/xunistd.h>
#include <sys/wait.h>

/* How many malloc objects to keep arond.  */
enum { malloc_objects = 1009 };

/* The maximum size of an object.  */
enum { malloc_maximum_size = 70000 };

/* How many iterations the test performs before exiting.  */
enum { iterations = 10000 };

/* Barrier for synchronization with the threads sending SIGUSR1
   signals, to make it more likely that the signals arrive during a
   fork/free/malloc call.  */
static pthread_barrier_t barrier;

/* Set to 1 if SIGUSR1 is received.  Used to detect a signal during
   fork/free/malloc.  */
static volatile sig_atomic_t sigusr1_received;

/* Periodically set to 1, to indicate that the thread is making
   progress.  Checked by liveness_signal_handler.  */
static volatile sig_atomic_t progress_indicator = 1;

/* Set to 1 if an error occurs in the signal handler.  */
static volatile sig_atomic_t error_indicator = 0;

static void
sigusr1_handler (int signo)
{
  sigusr1_received = 1;

  /* Perform a fork with a trivial subprocess.  */
  pid_t pid = _Fork ();
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
    write_message ("warning: thread seems to be stuck\n");
}

struct signal_send_args
{
  pthread_t target;
  int signo;
  bool sleep;
};
#define SIGNAL_SEND_GET_ARG(arg, field) \
  (((struct signal_send_args *)(arg))->field)

/* Send SIGNO to the parent thread.  If SLEEP, wait a second between
   signals, otherwise use barriers to delay sending signals.  */
static void *
signal_sender (void *args)
{
  int signo = SIGNAL_SEND_GET_ARG (args, signo);
  bool sleep = SIGNAL_SEND_GET_ARG (args, sleep);

  pthread_t target = SIGNAL_SEND_GET_ARG (args, target);
  while (true)
    {
      if (!sleep)
        xpthread_barrier_wait (&barrier);
      xpthread_kill (target, signo);
      if (sleep)
        usleep (1 * 1000 * 1000);
      else
        xpthread_barrier_wait (&barrier);
    }
  return NULL;
}

static pthread_t sigusr1_sender[5];
static pthread_t sigusr2_sender;

static int
do_test (void)
{
  xsignal (SIGUSR1, sigusr1_handler);
  xsignal (SIGUSR2, liveness_signal_handler);

  pthread_t self = pthread_self ();

  struct signal_send_args sigusr2_args = { self, SIGUSR2, true };
  sigusr2_sender = xpthread_create (NULL, signal_sender, &sigusr2_args);

  /* Send SIGUSR1 signals from several threads.  Hopefully, one
     signal will hit one of the ciritical functions.  Use a barrier to
     avoid sending signals while not running fork/free/malloc.  */
  struct signal_send_args sigusr1_args = { self, SIGUSR1, false };
  xpthread_barrier_init (&barrier, NULL,
                         array_length (sigusr1_sender) + 1);
  for (size_t i = 0; i < array_length (sigusr1_sender); ++i)
    sigusr1_sender[i] = xpthread_create (NULL, signal_sender, &sigusr1_args);

  void *objects[malloc_objects] = {};
  unsigned int fork_signals = 0;
  unsigned int free_signals = 0;
  unsigned int malloc_signals = 0;
  unsigned int seed = 1;
  for (int i = 0; i < iterations; ++i)
    {
      progress_indicator = 1;
      int slot = rand_r (&seed) % malloc_objects;
      size_t size = rand_r (&seed) % malloc_maximum_size;

      /* Occasionally do a fork first, to catch deadlocks there as
         well (see bug 24161).  */
      bool do_fork = (rand_r (&seed) % 7) == 0;

      xpthread_barrier_wait (&barrier);
      if (do_fork)
        {
          sigusr1_received = 0;
          pid_t pid = _Fork ();
          TEST_VERIFY_EXIT (pid != -1);
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
      xpthread_barrier_wait (&barrier);

      if (objects[slot] == NULL || error_indicator != 0)
        {
          printf ("error: malloc: %m\n");
          return 1;
        }
    }

  /* Clean up allocations.  */
  for (int slot = 0; slot < malloc_objects; ++slot)
    free (objects[slot]);

  printf ("info: signals received during fork: %u\n", fork_signals);
  printf ("info: signals received during free: %u\n", free_signals);
  printf ("info: signals received during malloc: %u\n", malloc_signals);

  return 0;
}

#define TIMEOUT 100
#include <support/test-driver.c>
