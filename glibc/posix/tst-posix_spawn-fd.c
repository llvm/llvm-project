/* Test that spawn file action functions work without file limit.
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
#include <fcntl.h>
#include <spawn.h>
#include <stdbool.h>
#include <stdio.h>
#include <sys/resource.h>
#include <unistd.h>

/* _SC_OPEN_MAX value.  */
static long maxfd;

/* A positive but unused file descriptor, used for testing
   purposes.  */
static int invalid_fd;

/* Indicate that errors have been encountered.  */
static bool errors;

static posix_spawn_file_actions_t actions;

static void
one_test (const char *name, int (*func) (int), int fd,
          bool expect_success)
{
  int ret = func (fd);
  if (expect_success)
    {
      if (ret != 0)
        {
          errno = ret;
          printf ("error: posix_spawn_file_actions_%s (%d): %m\n", name, fd);
          errors = true;
        }
    }
  else if (ret != EBADF)
    {
      if (ret == 0)
          printf ("error: posix_spawn_file_actions_%s (%d):"
                  " unexpected success\n", name, fd);
      else
        {
          errno = ret;
          printf ("error: posix_spawn_file_actions_%s (%d): %m\n", name, fd);
        }
      errors = true;
    }
}

static void
all_tests (const char *name, int (*func) (int))
{
  one_test (name, func, 0, true);
  one_test (name, func, invalid_fd, true);
  one_test (name, func, -1, false);
  one_test (name, func, -2, false);
  if (maxfd >= 0)
    one_test (name, func, maxfd, false);
}

static int
addopen (int fd)
{
  return posix_spawn_file_actions_addopen
    (&actions, fd, "/dev/null", O_RDONLY, 0);
}

static int
adddup2 (int fd)
{
  return posix_spawn_file_actions_adddup2 (&actions, fd, 1);
}

static int
adddup2_reverse (int fd)
{
  return posix_spawn_file_actions_adddup2 (&actions, 1, fd);
}

static int
addclose (int fd)
{
  return posix_spawn_file_actions_addclose (&actions, fd);
}

static void
all_functions (void)
{
  all_tests ("addopen", addopen);
  all_tests ("adddup2", adddup2);
  all_tests ("adddup2", adddup2_reverse);
  all_tests ("adddup2", addclose);
}

static int
do_test (void)
{
  /* Try to eliminate the file descriptor limit.  */
  {
    struct rlimit limit;
    if (getrlimit (RLIMIT_NOFILE, &limit) < 0)
      {
        printf ("error: getrlimit: %m\n");
        return 1;
      }
    limit.rlim_cur = RLIM_INFINITY;
    if (setrlimit (RLIMIT_NOFILE, &limit) < 0)
      printf ("warning: setrlimit: %m\n");
  }

  maxfd = sysconf (_SC_OPEN_MAX);
  printf ("info: _SC_OPEN_MAX: %ld\n", maxfd);

  invalid_fd = dup (0);
  if (invalid_fd < 0)
    {
      printf ("error: dup: %m\n");
      return 1;
    }
  if (close (invalid_fd) < 0)
    {
      printf ("error: close: %m\n");
      return 1;
    }

  int ret = posix_spawn_file_actions_init (&actions);
  if (ret != 0)
    {
      errno = ret;
      printf ("error: posix_spawn_file_actions_init: %m\n");
      return 1;
    }

  all_functions ();

  ret = posix_spawn_file_actions_destroy (&actions);
  if (ret != 0)
    {
      errno = ret;
      printf ("error: posix_spawn_file_actions_destroy: %m\n");
      return 1;
    }

  return errors;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
