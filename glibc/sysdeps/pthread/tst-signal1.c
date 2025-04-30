/* Copyright (C) 2002-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@redhat.com>, 2002.

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
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/wait.h>


static sigset_t ss;
static pthread_barrier_t *b;


static void *
tf (void *arg)
{
  sigdelset (&ss, SIGINT);

  if (pthread_sigmask (SIG_SETMASK, &ss, NULL) != 0)
    {
      puts ("2nd pthread_sigmask failed");
      exit (1);
    }

  pthread_barrier_wait (b);

  int sig;
  int res = sigwait (&ss, &sig);
  if (res == 0)
    {
      printf ("sigwait returned successfully with signal %d\n", sig);
      exit (1);
    }

  printf ("sigwait returned with %s (%d)\n", strerror (res), res);

  return NULL;
}


static void
receiver (void)
{
  pthread_t th;

  /* Make sure the process doesn't run forever.  */
  alarm (10);

  sigfillset (&ss);

  if (pthread_sigmask (SIG_SETMASK, &ss, NULL) != 0)
    {
      puts ("1st pthread_sigmask failed");
      exit (1);
    }

  if (pthread_create (&th, NULL, tf, NULL) != 0)
    {
      puts ("pthread_create failed");
      exit (1);
    }

  if (pthread_join (th, NULL) == 0)
    {
      puts ("thread joined?!");
      exit (1);
    }

  _exit (0);
}


static int
do_test (void)
{
  char tmp[] = "/tmp/tst-signal1-XXXXXX";

  int fd = mkstemp (tmp);
  if (fd == -1)
    {
      puts ("mkstemp failed");
      exit (1);
    }

  unlink (tmp);

  int i;
  for (i = 0; i < 20; ++i)
    write (fd, "foobar xyzzy", 12);

  b = mmap (NULL, sizeof (pthread_barrier_t), PROT_READ | PROT_WRITE,
	    MAP_SHARED, fd, 0);
  if (b == MAP_FAILED)
    {
      puts ("mmap failed");
      exit (1);
    }

  pthread_barrierattr_t ba;
  if (pthread_barrierattr_init (&ba) != 0)
    {
      puts ("barrierattr_init failed");
      exit (1);
    }

  if (pthread_barrierattr_setpshared (&ba, PTHREAD_PROCESS_SHARED) != 0)
    {
      puts ("barrierattr_setpshared failed");
      exit (1);
    }

  if (pthread_barrier_init (b, &ba, 2) != 0)
    {
      puts ("barrier_init failed");
      exit (1);
    }

  if (pthread_barrierattr_destroy (&ba) != 0)
    {
      puts ("barrierattr_destroy failed");
      exit (1);
    }

  pid_t pid = fork ();
  if (pid == -1)
    {
      puts ("fork failed");
      exit (1);
    }

  if (pid == 0)
    receiver ();

  pthread_barrier_wait (b);

  /* Wait a bit more.  */
  struct timespec ts = { .tv_sec = 0, .tv_nsec = 10000000 };
  nanosleep (&ts, NULL);

  /* Send the signal.  */
  puts ("sending the signal now");
  kill (pid, SIGINT);

  /* Wait for the process to terminate.  */
  int status;
  if (TEMP_FAILURE_RETRY (waitpid (pid, &status, 0)) != pid)
    {
      puts ("wrong child reported terminated");
      exit (1);
    }

  if (!WIFSIGNALED (status))
    {
      puts ("child wasn't signalled");
      exit (1);
    }

  if (WTERMSIG (status) != SIGINT)
    {
      puts ("child not terminated with SIGINT");
      exit (1);
    }

  return 0;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
