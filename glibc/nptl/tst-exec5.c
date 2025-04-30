/* Check if posix_spawn does not act as a cancellation entrypoint.
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

#include <errno.h>
#include <paths.h>
#include <pthread.h>
#include <signal.h>
#include <spawn.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>

static int do_test (void);
#define TEST_FUNCTION do_test ()
#include <test-skeleton.c>

static pthread_barrier_t b;

static pid_t pid;
static int pipefd[2];

static void *
tf (void *arg)
{
  xpthread_barrier_wait (&b);

  posix_spawn_file_actions_t a;
  if (posix_spawn_file_actions_init (&a) != 0)
    {
      puts ("error: spawn_file_actions_init failed");
      exit (1);
    }

  if (posix_spawn_file_actions_adddup2 (&a, pipefd[1], STDOUT_FILENO) != 0)
    {
      puts ("error: spawn_file_actions_adddup2 failed");
      exit (1);
    }

  if (posix_spawn_file_actions_addclose (&a, pipefd[0]) != 0)
    {
      puts ("error: spawn_file_actions_addclose");
      exit (1);
    }

  char *argv[] = { (char *) _PATH_BSHELL, (char *) "-c", (char *) "echo $$",
		   NULL };
  if (posix_spawn (&pid, _PATH_BSHELL, &a, NULL, argv, NULL) != 0)
    {
      puts ("error: spawn failed");
      exit (1);
    }

  return NULL;
}


static int
do_test (void)
{
  /* The test basically pipe a 'echo $$' created by a thread with a
     cancellation pending.  It then checks if the thread is not cancelled,
     the process is created and if the output is the expected one.  */

  if (pipe (pipefd) != 0)
    {
      puts ("error: pipe failed");
      exit (1);
    }

  /* Not interested in knowing when the pipe is closed.  */
  xsignal (SIGPIPE, SIG_IGN);

  /* To synchronize with the thread.  */
  if (pthread_barrier_init (&b, NULL, 2) != 0)
    {
      puts ("error: pthread_barrier_init failed");
      exit (1);
    }

  pthread_t th = xpthread_create (NULL, &tf, NULL);

  if (pthread_cancel (th) != 0)
    {
      puts ("error: pthread_cancel failed");
      return 1;
    }

  xpthread_barrier_wait (&b);

  if (xpthread_join (th) == PTHREAD_CANCELED)
    {
      puts ("error: thread cancelled");
      exit (1);
    }

  close (pipefd[1]);

  /* The global 'pid' should be set by thread posix_spawn calling.  Check
     below if it was executed correctly and with expected output.  */

  char buf[64];
  ssize_t n;
  bool seen_pid = false;
  while (TEMP_FAILURE_RETRY ((n = read (pipefd[0], buf, sizeof (buf)))) > 0)
    {
      /* We only expect to read the PID.  */
      char *endp;
      long int rpid = strtol (buf, &endp, 10);

      if (*endp != '\n')
	{
	  printf ("error: didn't parse whole line: \"%s\"\n", buf);
	  exit (1);
	}
      if (endp == buf)
	{
	  puts ("error: read empty line");
	  exit (1);
	}

      if (rpid != pid)
	{
	  printf ("error: found \"%s\", expected PID %ld\n", buf,
		 (long int) pid);
	  exit (1);
	}

      if (seen_pid)
	{
	  puts ("error: found more than one PID line");
	  exit (1);
	}

      seen_pid = true;
    }

  close (pipefd[0]);

  int status;
  int err = waitpid (pid, &status, 0);
  if (err != pid)
    {
      puts ("errnor: waitpid failed");
      exit (1);
    }

  if (!seen_pid)
    {
      puts ("error: didn't get PID");
      exit (1);
    }

  return 0;
}
