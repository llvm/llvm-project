/* Test for sched_getaffinity and sched_setaffinity, PID version.
   Copyright (C) 2015-2021 Free Software Foundation, Inc.
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

/* Function definitions for the benefit of tst-skeleton-affinity.c.
   This variant forks a child process which then invokes
   sched_getaffinity and sched_setaffinity on the parent PID.  */

#include <errno.h>
#include <stdlib.h>
#include <sched.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/wait.h>
#include <unistd.h>

static int
write_fully (int fd, const void *buffer, size_t length)
{
  const void *end = buffer + length;
  while (buffer < end)
    {
      ssize_t bytes_written = TEMP_FAILURE_RETRY
        (write (fd, buffer, end - buffer));
      if (bytes_written < 0)
        return -1;
      if (bytes_written == 0)
        {
          errno = ENOSPC;
          return -1;
        }
      buffer += bytes_written;
    }
  return 0;
}

static ssize_t
read_fully (int fd, void *buffer, size_t length)
{
  const void *start = buffer;
  const void *end = buffer + length;
  while (buffer < end)
    {
      ssize_t bytes_read = TEMP_FAILURE_RETRY
        (read (fd, buffer, end - buffer));
      if (bytes_read < 0)
        return -1;
      if (bytes_read == 0)
        return buffer - start;
      buffer += bytes_read;
    }
  return length;
}

static int
process_child_response (int *pipes, pid_t child,
                        cpu_set_t *set, size_t size)
{
  close (pipes[1]);

  int value_from_child;
  ssize_t bytes_read = read_fully
    (pipes[0], &value_from_child, sizeof (value_from_child));
  if (bytes_read < 0)
    {
      printf ("error: read from child: %m\n");
      exit (1);
    }
  if (bytes_read != sizeof (value_from_child))
    {
      printf ("error: not enough bytes from child: %zd\n", bytes_read);
      exit (1);
    }
  if (value_from_child == 0)
    {
      bytes_read = read_fully (pipes[0], set, size);
      if (bytes_read < 0)
        {
          printf ("error: read: %m\n");
          exit (1);
        }
      if (bytes_read != size)
        {
          printf ("error: not enough bytes from child: %zd\n", bytes_read);
          exit (1);
        }
    }

  int status;
  if (waitpid (child, &status, 0) < 0)
    {
      printf ("error: waitpid: %m\n");
      exit (1);
    }
  if (!(WIFEXITED (status) && WEXITSTATUS (status) == 0))
    {
      printf ("error: invalid status from : %m\n");
      exit (1);
    }

  close (pipes[0]);

  if (value_from_child != 0)
    {
      errno = value_from_child;
      return -1;
    }
  return 0;
}

static int
getaffinity (size_t size, cpu_set_t *set)
{
  int pipes[2];
  if (pipe (pipes) < 0)
    {
      printf ("error: pipe: %m\n");
      exit (1);
    }

  int ret = fork ();
  if (ret < 0)
    {
      printf ("error: fork: %m\n");
      exit (1);
    }
  if (ret == 0)
    {
      /* Child.  */
      int ret = sched_getaffinity (getppid (), size, set);
      if (ret < 0)
        ret = errno;
      if (write_fully (pipes[1], &ret, sizeof (ret)) < 0
          || write_fully (pipes[1], set, size) < 0
          || (ret == 0 && write_fully (pipes[1], set, size) < 0))
        {
          printf ("error: write: %m\n");
          _exit (1);
        }
      _exit (0);
    }

  /* Parent.  */
  return process_child_response (pipes, ret, set, size);
}

static int
setaffinity (size_t size, const cpu_set_t *set)
{
  int pipes[2];
  if (pipe (pipes) < 0)
    {
      printf ("error: pipe: %m\n");
      exit (1);
    }

  int ret = fork ();
  if (ret < 0)
    {
      printf ("error: fork: %m\n");
      exit (1);
    }
  if (ret == 0)
    {
      /* Child.  */
      int ret = sched_setaffinity (getppid (), size, set);
      if (write_fully (pipes[1], &ret, sizeof (ret)) < 0)
        {
          printf ("error: write: %m\n");
          _exit (1);
        }
      _exit (0);
    }

  /* Parent.  There is no affinity mask to read from the child, so the
     size is 0.  */
  return process_child_response (pipes, ret, NULL, 0);
}

struct conf;
static bool early_test (struct conf *unused)
{
  return true;
}

#include "tst-skeleton-affinity.c"
