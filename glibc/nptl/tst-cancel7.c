/* Copyright (C) 2002-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Jakub Jelinek <jakub@redhat.com>, 2002.

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
#include <fcntl.h>
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <getopt.h>

#include <support/xthread.h>

const char *command;
const char *pidfile;
char pidfilename[] = "/tmp/tst-cancel7-XXXXXX";

static void *
tf (void *arg)
{
  const char *args = " --direct --pidfile ";
  char *cmd = alloca (strlen (command) + strlen (args)
		      + strlen (pidfilename) + 1);

  strcpy (stpcpy (stpcpy (cmd, command), args), pidfilename);
  system (cmd);
  /* This call should never return.  */
  return NULL;
}


static void
sl (void)
{
  FILE *f = fopen (pidfile, "w");
  if (f == NULL)
    exit (1);

  fprintf (f, "%lld\n", (long long) getpid ());
  fflush (f);

  struct flock fl =
    {
      .l_type = F_WRLCK,
      .l_start = 0,
      .l_whence = SEEK_SET,
      .l_len = 1
    };
  if (fcntl (fileno (f), F_SETLK, &fl) != 0)
    exit (1);

  sigset_t ss;
  sigfillset (&ss);
  sigsuspend (&ss);
  exit (0);
}


static void
do_prepare (int argc, char *argv[])
{
  if (command == NULL)
    command = argv[0];

  if (pidfile)
    sl ();

  int fd = mkstemp (pidfilename);
  if (fd == -1)
    {
      puts ("mkstemp failed");
      exit (1);
    }

  write (fd, " ", 1);
  close (fd);
}


static int
do_test (void)
{
  pthread_t th;
  if (pthread_create (&th, NULL, tf, NULL) != 0)
    {
      puts ("pthread_create failed");
      return 1;
    }

  do
    sleep (1);
  while (access (pidfilename, R_OK) != 0);

  xpthread_cancel (th);
  void *r = xpthread_join (th);

  sleep (1);

  FILE *f = fopen (pidfilename, "r+");
  if (f == NULL)
    {
      puts ("no pidfile");
      return 1;
    }

  long long ll;
  if (fscanf (f, "%lld\n", &ll) != 1)
    {
      puts ("could not read pid");
      unlink (pidfilename);
      return 1;
    }

  struct flock fl =
    {
      .l_type = F_WRLCK,
      .l_start = 0,
      .l_whence = SEEK_SET,
      .l_len = 1
    };
  if (fcntl (fileno (f), F_GETLK, &fl) != 0)
    {
      puts ("F_GETLK failed");
      unlink (pidfilename);
      return 1;
    }

  if (fl.l_type != F_UNLCK)
    {
      printf ("child %lld still running\n", (long long) fl.l_pid);
      if (fl.l_pid == ll)
	kill (fl.l_pid, SIGKILL);

      unlink (pidfilename);
      return 1;
    }

  fclose (f);

  unlink (pidfilename);

  return r != PTHREAD_CANCELED;
}

static void
do_cleanup (void)
{
  FILE *f = fopen (pidfilename, "r+");
  long long ll;

  if (f != NULL && fscanf (f, "%lld\n", &ll) == 1)
    {
      struct flock fl =
	{
	  .l_type = F_WRLCK,
	  .l_start = 0,
	  .l_whence = SEEK_SET,
	  .l_len = 1
	};
      if (fcntl (fileno (f), F_GETLK, &fl) == 0 && fl.l_type != F_UNLCK
	  && fl.l_pid == ll)
	kill (fl.l_pid, SIGKILL);

      fclose (f);
    }

  unlink (pidfilename);
}

#define OPT_COMMAND	10000
#define OPT_PIDFILE	10001
#define CMDLINE_OPTIONS \
  { "command", required_argument, NULL, OPT_COMMAND },	\
  { "pidfile", required_argument, NULL, OPT_PIDFILE },
static void
cmdline_process (int c)
{
  switch (c)
    {
    case OPT_COMMAND:
      command = optarg;
      break;
    case OPT_PIDFILE:
      pidfile = optarg;
      break;
    }
}
#define CMDLINE_PROCESS cmdline_process
#define CLEANUP_HANDLER do_cleanup
#define PREPARE do_prepare
#include <support/test-driver.c>
