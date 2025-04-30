/* Thread calls exec.
   Copyright (C) 2002-2021 Free Software Foundation, Inc.
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
#include <paths.h>
#include <pthread.h>
#include <signal.h>
#include <spawn.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>
#include <support/xsignal.h>


static void *
tf (void *arg)
{
  execl (_PATH_BSHELL, _PATH_BSHELL, "-c", "echo $$", NULL);

  puts ("execl failed");
  exit (1);
}


static int
do_test (void)
{
  int fd[2];
  if (pipe (fd) != 0)
    {
      puts ("pipe failed");
      exit (1);
    }

  /* Not interested in knowing when the pipe is closed.  */
  xsignal (SIGPIPE, SIG_IGN);

  pid_t pid = fork ();
  if (pid == -1)
    {
      puts ("fork failed");
      exit (1);
    }

  if (pid == 0)
    {
      /* Use the fd for stdout.  This is kind of ugly because it
	 substitutes the fd of stdout but we know what we are doing
	 here...  */
      if (dup2 (fd[1], STDOUT_FILENO) != STDOUT_FILENO)
	{
	  puts ("dup2 failed");
	  exit (1);
	}

      close (fd[0]);

      pthread_t th;
      if (pthread_create (&th, NULL, tf, NULL) != 0)
	{
	  puts ("create failed");
	  exit (1);
	}

      if (pthread_join (th, NULL) == 0)
	{
	  puts ("join succeeded!?");
	  exit (1);
	}

      puts ("join returned!?");
      exit (1);
    }

  close (fd[1]);

  char buf[200];
  ssize_t n;
  bool seen_pid = false;
  while (TEMP_FAILURE_RETRY ((n = read (fd[0], buf, sizeof (buf)))) > 0)
    {
      /* We only expect to read the PID.  */
      char *endp;
      long int rpid = strtol (buf, &endp, 10);

      if (*endp != '\n')
	{
	  printf ("didn't parse whole line: \"%s\"\n", buf);
	  exit (1);
	}
      if (endp == buf)
	{
	  puts ("read empty line");
	  exit (1);
	}

      if (rpid != pid)
	{
	  printf ("found \"%s\", expected PID %ld\n", buf, (long int) pid);
	  exit (1);
	}

      if (seen_pid)
	{
	  puts ("found more than one PID line");
	  exit (1);
	}
      seen_pid = true;
    }

  close (fd[0]);

  int status;
  int err = waitpid (pid, &status, 0);
  if (err != pid)
    {
      puts ("waitpid failed");
      exit (1);
    }

  if (!seen_pid)
    {
      puts ("didn't get PID");
      exit (1);
    }

  return 0;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
