/* Test all open message queues descriptors are closed during exec*.
   Copyright (C) 2004-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Jakub Jelinek <jakub@redhat.com>, 2004.

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
#include <mqueue.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#define OPT_AFTEREXEC 20000

static mqd_t after_exec = (mqd_t) -1;

#define CMDLINE_OPTIONS \
  { "after-exec", required_argument, NULL, OPT_AFTEREXEC },

#define CMDLINE_PROCESS \
  case OPT_AFTEREXEC:					\
    after_exec = (mqd_t) strtoul (optarg, NULL, 0);	\
    break;

static int
do_after_exec (void)
{
  int result = 0;

  struct mq_attr attr;
  if (mq_getattr (after_exec, &attr) == 0)
    {
      puts ("mq_getattr after exec unexpectedly succeeded");
      result = 1;
    }
  else if (errno != EBADF)
    {
      printf ("mq_getattr after exec did not fail with EBADF: %m\n");
      result = 1;
    }

  return result;
}

static int
do_test (int argc, char **argv)
{
  if (after_exec != (mqd_t) -1)
    return do_after_exec ();

  char name[sizeof "/tst-mqueue7-" + sizeof (pid_t) * 3];
  snprintf (name, sizeof (name), "/tst-mqueue7-%u", getpid ());

  struct mq_attr attr = { .mq_maxmsg = 10, .mq_msgsize = 1 };
  mqd_t q = mq_open (name, O_CREAT | O_EXCL | O_WRONLY, 0600, &attr);

  if (q == (mqd_t) -1)
    {
      printf ("mq_open failed with: %m\n");
      return 0;
    }
  else if (mq_unlink (name) != 0)
    {
      printf ("mq_unlink failed with: %m\n");
      return 1;
    }

  if (mq_getattr (q, &attr) != 0)
    {
      printf ("mq_getattr failed: %m\n");
      return 1;
    }

  char after_exec_arg[sizeof "--after-exec=0x" + sizeof (long) * 3];
  snprintf (after_exec_arg, sizeof (after_exec_arg),
	    "--after-exec=0x%lx", (long) q);

  const char *newargv[argc + 2];
  for (int i = 1; i < argc; ++i)
    newargv[i - 1] = argv[i];
  newargv[argc - 1] = "--direct";
  newargv[argc] = after_exec_arg;
  newargv[argc + 1] = NULL;

  /* Verify that exec* has the effect of mq_close (q).  */
  execv (newargv[0], (char * const *) newargv);
  printf ("execv failed: %m\n");
  return 1;
}

#include "../test-skeleton.c"
