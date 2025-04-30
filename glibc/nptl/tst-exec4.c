/* Signal handler and mask set in thread which calls exec.
   Copyright (C) 2003-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@redhat.com>, 2003.

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

#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <support/xsignal.h>

static void *
tf (void *arg)
{
  /* Ignore SIGUSR1 and block SIGUSR2.  */
  xsignal (SIGUSR1, SIG_IGN);

  sigset_t ss;
  sigemptyset (&ss);
  sigaddset (&ss, SIGUSR2);
  if (pthread_sigmask (SIG_BLOCK, &ss, NULL) != 0)
    {
      puts ("1st run: sigmask failed");
      exit (1);
    }

  char **oldargv = (char **) arg;
  size_t n = 1;
  while (oldargv[n] != NULL)
    ++n;

  char **argv = (char **) alloca ((n + 1) * sizeof (char *));
  for (n = 0; oldargv[n + 1] != NULL; ++n)
    argv[n] = oldargv[n + 1];
  argv[n++] = (char *) "--direct";
  argv[n] = NULL;

  execv (argv[0], argv);

  puts ("execv failed");

  exit (1);
}


static int
do_test (int argc, char *argv[])
{
  if (argc == 1)
    {
      /* This is the second call.  Perform the test.  */
      struct sigaction sa;

      if (sigaction (SIGUSR1, NULL, &sa) != 0)
	{
	  puts ("2nd run: sigaction failed");
	  return 1;
	}
      if (sa.sa_handler != SIG_IGN)
	{
	  puts ("SIGUSR1 not ignored");
	  return 1;
	}

      sigset_t ss;
      if (pthread_sigmask (SIG_SETMASK, NULL, &ss) != 0)
	{
	  puts ("2nd run: sigmask failed");
	  return 1;
	}
      if (! sigismember (&ss, SIGUSR2))
	{
	  puts ("SIGUSR2 not blocked");
	  return 1;
	}

      return 0;
    }

  pthread_t th;
  if (pthread_create (&th, NULL, tf, argv) != 0)
    {
      puts ("create failed");
      exit (1);
    }

  /* This call should never return.  */
  pthread_join (th, NULL);

  puts ("join returned");

  return 1;
}

#define TEST_FUNCTION do_test (argc, argv)
#include "../test-skeleton.c"
