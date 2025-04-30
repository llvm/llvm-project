/* Check posix_spawn add file actions.
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

#include <stdio.h>
#include <spawn.h>
#include <error.h>
#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/resource.h>
#include <fcntl.h>
#include <paths.h>

#include <support/check.h>
#include <support/temp_file.h>

static int
do_test (void)
{
  /* The test checks if posix_spawn open file action close the file descriptor
     before opening a new one in case the input file descriptor is already
     opened.  It does by exhausting all file descriptors on the process before
     issue posix_spawn.  It then issues a posix_spawn for '/bin/sh echo $$'
     and add two rules:

     1. Redirect stdout to a temporary filepath
     2. Redirect stderr to stdout

     If the implementation does not close the file 1. will fail with
     EMFILE.  */

  struct rlimit rl;
  int max_fd = 24;
  int ret;

  /* Set maximum number of file descriptor to a low value to avoid open
     too many files in environments where RLIMIT_NOFILE is large and to
     limit the array size to track the opened file descriptors.  */

  if (getrlimit (RLIMIT_NOFILE, &rl) == -1)
    FAIL_EXIT1 ("getrlimit (RLIMIT_NOFILE): %m");

  max_fd = (rl.rlim_cur < max_fd ? rl.rlim_cur : max_fd);
  rl.rlim_cur = max_fd;

  if (setrlimit (RLIMIT_NOFILE, &rl) == 1)
    FAIL_EXIT1 ("setrlimit (RLIMIT_NOFILE): %m");

  /* Exhauste the file descriptor limit with temporary files.  */
  int files[max_fd];
  int nfiles = 0;
  for (;;)
    {
      int fd = create_temp_file ("tst-spawn3.", NULL);
      if (fd == -1)
	{
	  if (errno != EMFILE)
	    FAIL_EXIT1 ("create_temp_file: %m");
	  break;
	}
      files[nfiles++] = fd;
    }

  posix_spawn_file_actions_t a;
  if (posix_spawn_file_actions_init (&a) != 0)
    FAIL_EXIT1 ("posix_spawn_file_actions_init");

  /* Executes a /bin/sh echo $$ 2>&1 > ${objpfx}tst-spawn3.pid .  */
  const char pidfile[] = OBJPFX "tst-spawn3.pid";
  if (posix_spawn_file_actions_addopen (&a, STDOUT_FILENO, pidfile, O_WRONLY
					| O_CREAT | O_TRUNC, 0644) != 0)
    FAIL_EXIT1 ("posix_spawn_file_actions_addopen");

  if (posix_spawn_file_actions_adddup2 (&a, STDOUT_FILENO, STDERR_FILENO) != 0)
    FAIL_EXIT1 ("posix_spawn_file_actions_adddup2");

  /* Since execve (called by posix_spawn) might require to open files to
     actually execute the shell script, setup to close the temporary file
     descriptors.  */
  for (int i=0; i<nfiles; i++)
    {
      if (posix_spawn_file_actions_addclose (&a, files[i]))
	FAIL_EXIT1 ("posix_spawn_file_actions_addclose");
    }

  char *spawn_argv[] = { (char *) _PATH_BSHELL, (char *) "-c",
			 (char *) "echo $$", NULL };
  pid_t pid;
  if ((ret = posix_spawn (&pid, _PATH_BSHELL, &a, NULL, spawn_argv, NULL))
       != 0)
    {
      errno = ret;
      FAIL_EXIT1 ("posix_spawn: %m");
    }

  int status;
  int err = waitpid (pid, &status, 0);
  if (err != pid)
    FAIL_EXIT1 ("waitpid: %m");

  /* Close the temporary files descriptor so it can check posix_spawn
     output.  */
  for (int i=0; i<nfiles; i++)
    {
      if (close (files[i]))
	FAIL_EXIT1 ("close: %m");
    }

  int pidfd = open (pidfile, O_RDONLY);
  if (pidfd == -1)
    FAIL_EXIT1 ("open: %m");

  char buf[64];
  ssize_t n;
  if ((n = read (pidfd, buf, sizeof (buf))) < 0)
    FAIL_EXIT1 ("read: %m");

  unlink (pidfile);

  /* We only expect to read the PID.  */
  char *endp;
  long int rpid = strtol (buf, &endp, 10);
  if (*endp != '\n')
    FAIL_EXIT1 ("*endp != \'n\'");
  if (endp == buf)
    FAIL_EXIT1 ("read empty line");

  if (rpid != pid)
    FAIL_EXIT1 ("found \"%s\", expected pid %ld\n", buf, (long int) pid);

  return 0;
}

#include <support/test-driver.c>
