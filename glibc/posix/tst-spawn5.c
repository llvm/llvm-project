/* Tests for posix_spawn signal handling.
   Copyright (C) 2021 Free Software Foundation, Inc.
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
   <http://www.gnu.org/licenses/>.  */

#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include <spawn.h>
#include <fcntl.h>
#include <sys/wait.h>
#include <dirent.h>
#include <stdbool.h>
#include <errno.h>
#include <limits.h>

#include <support/check.h>
#include <support/xunistd.h>
#include <support/support.h>

#include <arch-fd_to_filename.h>
#include <array_length.h>

/* Nonzero if the program gets called via `exec'.  */
static int restart;

/* Hold the four initial argument used to respawn the process, plus
   the extra '--direct' and '--restart', and a final NULL.  */
static char *initial_argv[7];
static int initial_argv_count;

#define CMDLINE_OPTIONS \
  { "restart", no_argument, &restart, 1 },

#define NFDS 100

static int
open_multiple_temp_files (void)
{
  /* Check if the temporary file descriptor has no no gaps.  */
  int lowfd = xopen ("/dev/null", O_RDONLY, 0600);
  for (int i = 1; i <= NFDS; i++)
    TEST_COMPARE (xopen ("/dev/null", O_RDONLY, 0600),
		  lowfd + i);
  return lowfd;
}

static int
parse_fd (const char *str)
{
  char *endptr;
  long unsigned int fd = strtoul (str, &endptr, 10);
  if (*endptr != '\0' || fd > INT_MAX)
    FAIL_EXIT1 ("invalid file descriptor value: %s", str);
  return fd;
}

/* Called on process re-execution.  The arguments are the expected opened
   file descriptors.  */
_Noreturn static void
handle_restart (int argc, char *argv[])
{
  TEST_VERIFY (argc > 0);
  int lowfd = parse_fd (argv[0]);

  size_t nfds = argc > 1 ? argc - 1 : 0;
  struct fd_t
  {
    int fd;
    _Bool found;
  } *fds = xmalloc (sizeof (struct fd_t) * nfds);
  for (int i = 0; i < nfds; i++)
    {
      fds[i].fd = parse_fd (argv[i + 1]);
      fds[i].found = false;
    }

  DIR *dirp = opendir (FD_TO_FILENAME_PREFIX);
  if (dirp == NULL)
    FAIL_EXIT1 ("opendir (\"" FD_TO_FILENAME_PREFIX "\"): %m");

  while (true)
    {
      errno = 0;
      struct dirent64 *e = readdir64 (dirp);
      if (e == NULL)
        {
          if (errno != 0)
            FAIL_EXIT1 ("readdir: %m");
          break;
        }

      if (e->d_name[0] == '.')
        continue;

      char *endptr;
      long int fd = strtol (e->d_name, &endptr, 10);
      if (*endptr != '\0' || fd < 0 || fd > INT_MAX)
        FAIL_EXIT1 ("readdir: invalid file descriptor name: /proc/self/fd/%s",
                    e->d_name);

      /* Ignore the descriptors not in the range of the opened files.  */
      if (fd < lowfd || fd == dirfd (dirp))
	continue;

      bool found = false;
      for (int i = 0; i < nfds; i++)
	if (fds[i].fd == fd)
	  fds[i].found = found = true;

      if (!found)
	{
	  char *path = xasprintf ("/proc/self/fd/%s", e->d_name);
	  char *resolved = xreadlink (path);
	  FAIL_EXIT1 ("unexpected open file descriptor %ld: %s", fd, resolved);
	}
    }
  closedir (dirp);

  for (int i = 0; i < nfds; i++)
    if (!fds[i].found)
      FAIL_EXIT1 ("file descriptor %d not opened", fds[i].fd);

  free (fds);

  exit (EXIT_SUCCESS);
}

static void
spawn_closefrom_test (posix_spawn_file_actions_t *fa, int lowfd, int highfd,
		      int *extrafds, size_t nextrafds)
{
  /* 3 or 7 elements from initial_argv:
       + path to ld.so          optional
       + --library-path         optional
       + the library path       optional
       + application name
       + --direct
       + --restart
       + lowest opened file descriptor
       + up to 2 * maximum_fd arguments (the expected open file descriptors),
	 plus NULL.  */

  int argv_size = initial_argv_count + 2 * NFDS + 1;
  char *args[argv_size];
  int argc = 0;

  for (char **arg = initial_argv; *arg != NULL; arg++)
    args[argc++] = *arg;

  args[argc++] = xasprintf ("%d", lowfd);

  for (int i = lowfd; i < highfd; i++)
    args[argc++] = xasprintf ("%d", i);

  for (int i = 0; i < nextrafds; i++)
    args[argc++] = xasprintf ("%d", extrafds[i]);

  args[argc] = NULL;
  TEST_VERIFY (argc < argv_size);

  pid_t pid;
  int status;

  TEST_COMPARE (posix_spawn (&pid, args[0], fa, NULL, args, environ), 0);
  TEST_COMPARE (xwaitpid (pid, &status, 0), pid);
  TEST_VERIFY (WIFEXITED (status));
  TEST_VERIFY (!WIFSIGNALED (status));
  TEST_COMPARE (WEXITSTATUS (status), 0);
}

static void
do_test_closefrom (void)
{
  int lowfd = open_multiple_temp_files ();
  const int half_fd = lowfd + NFDS / 2;

  /* Close half of the descriptors and check result.  */
  {
    posix_spawn_file_actions_t fa;
    TEST_COMPARE (posix_spawn_file_actions_init (&fa), 0);

    int ret = posix_spawn_file_actions_addclosefrom_np (&fa, half_fd);
    if (ret == EINVAL)
      /* Hurd currently does not support closefrom fileaction.  */
      FAIL_UNSUPPORTED ("posix_spawn_file_actions_addclosefrom_np unsupported");
    TEST_COMPARE (ret, 0);

    spawn_closefrom_test (&fa, lowfd, half_fd, NULL, 0);

    TEST_COMPARE (posix_spawn_file_actions_destroy (&fa), 0);
  }

  /* Create some gaps, close up to a threshold, and check result.  */
  xclose (lowfd + 57);
  xclose (lowfd + 78);
  xclose (lowfd + 81);
  xclose (lowfd + 82);
  xclose (lowfd + 84);
  xclose (lowfd + 90);

  {
    posix_spawn_file_actions_t fa;
    TEST_COMPARE (posix_spawn_file_actions_init (&fa), 0);

    TEST_COMPARE (posix_spawn_file_actions_addclosefrom_np (&fa, half_fd), 0);

    spawn_closefrom_test (&fa, lowfd, half_fd, NULL, 0);

    TEST_COMPARE (posix_spawn_file_actions_destroy (&fa), 0);
  }

  /* Close the remaining but the last one.  */
  {
    posix_spawn_file_actions_t fa;
    TEST_COMPARE (posix_spawn_file_actions_init (&fa), 0);

    TEST_COMPARE (posix_spawn_file_actions_addclosefrom_np (&fa, lowfd + 1), 0);

    spawn_closefrom_test (&fa, lowfd, lowfd + 1, NULL, 0);

    TEST_COMPARE (posix_spawn_file_actions_destroy (&fa), 0);
  }

  /* Close everything.  */
  {
    posix_spawn_file_actions_t fa;
    TEST_COMPARE (posix_spawn_file_actions_init (&fa), 0);

    TEST_COMPARE (posix_spawn_file_actions_addclosefrom_np (&fa, lowfd), 0);

    spawn_closefrom_test (&fa, lowfd, lowfd, NULL, 0);

    TEST_COMPARE (posix_spawn_file_actions_destroy (&fa), 0);
  }

  /* Close a range and add some file actions.  */
  {
    posix_spawn_file_actions_t fa;
    TEST_COMPARE (posix_spawn_file_actions_init (&fa), 0);

    TEST_COMPARE (posix_spawn_file_actions_addclosefrom_np (&fa, lowfd + 1), 0);
    TEST_COMPARE (posix_spawn_file_actions_addopen (&fa, lowfd, "/dev/null",
						    0666, O_RDONLY), 0);
    TEST_COMPARE (posix_spawn_file_actions_adddup2 (&fa, lowfd, lowfd + 1), 0);
    TEST_COMPARE (posix_spawn_file_actions_addopen (&fa, lowfd, "/dev/null",
						    0666, O_RDONLY), 0);

    spawn_closefrom_test (&fa, lowfd, lowfd, (int[]){lowfd, lowfd + 1}, 2);

    TEST_COMPARE (posix_spawn_file_actions_destroy (&fa), 0);
  }
}

static int
do_test (int argc, char *argv[])
{
  /* We must have either:

     - one or four parameters if called initially:
       + argv[1]: path for ld.so        optional
       + argv[2]: "--library-path"      optional
       + argv[3]: the library path      optional
       + argv[4]: the application name

     - six parameters left if called through re-execution:
       + argv[1]: the application name
       + argv[2]: the lowest file descriptor expected
       + argv[3]: first expected open file descriptor   optional
       + argv[n]: last expected open file descritptor   optional

     * When built with --enable-hardcoded-path-in-tests or issued without
       using the loader directly.  */

  if (restart)
    /* Ignore the application name. */
    handle_restart (argc - 1, &argv[1]);

  TEST_VERIFY_EXIT (argc == 2 || argc == 5);

  int i;

  for (i = 0; i < argc - 1; i++)
    initial_argv[i] = argv[i + 1];
  initial_argv[i++] = (char *) "--direct";
  initial_argv[i++] = (char *) "--restart";

  initial_argv_count = i;

  do_test_closefrom ();

  return 0;
}

#define TEST_FUNCTION_ARGV do_test
#include <support/test-driver.c>
