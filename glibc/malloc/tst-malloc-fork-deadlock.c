/* Test concurrent fork, getline, and fflush (NULL).
   Copyright (C) 2016-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public License as
   published by the Free Software Foundation; either version 2.1 of the
   License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; see the file COPYING.LIB.  If
   not, see <https://www.gnu.org/licenses/>.  */

#include <sys/wait.h>
#include <unistd.h>
#include <errno.h>
#include <stdio.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdlib.h>
#include <malloc.h>
#include <time.h>
#include <string.h>
#include <signal.h>

#include <support/xthread.h>
#include <support/temp_file.h>
#include <support/test-driver.h>

enum {
  /* Number of threads which call fork.  */
  fork_thread_count = 4,
  /* Number of threads which call getline (and, indirectly,
     malloc).  */
  read_thread_count = 8,
};

static bool termination_requested;

static void *
fork_thread_function (void *closure)
{
  while (!__atomic_load_n (&termination_requested, __ATOMIC_RELAXED))
    {
      pid_t pid = fork ();
      if (pid < 0)
        {
          printf ("error: fork: %m\n");
          abort ();
        }
      else if (pid == 0)
        _exit (17);

      int status;
      if (waitpid (pid, &status, 0) < 0)
        {
          printf ("error: waitpid: %m\n");
          abort ();
        }
      if (!WIFEXITED (status) || WEXITSTATUS (status) != 17)
        {
          printf ("error: waitpid returned invalid status: %d\n", status);
          abort ();
        }
    }
  return NULL;
}

static char *file_to_read;

static void *
read_thread_function (void *closure)
{
  FILE *f = fopen (file_to_read, "r");
  if (f == NULL)
    {
      printf ("error: fopen (%s): %m\n", file_to_read);
      abort ();
    }

  while (!__atomic_load_n (&termination_requested, __ATOMIC_RELAXED))
    {
      rewind (f);
      char *line = NULL;
      size_t line_allocated = 0;
      ssize_t ret = getline (&line, &line_allocated, f);
      if (ret < 0)
        {
          printf ("error: getline: %m\n");
          abort ();
        }
      free (line);
    }
  fclose (f);

  return NULL;
}

static void *
flushall_thread_function (void *closure)
{
  while (!__atomic_load_n (&termination_requested, __ATOMIC_RELAXED))
    if (fflush (NULL) != 0)
      {
        printf ("error: fflush (NULL): %m\n");
        abort ();
      }
  return NULL;
}

static void
create_threads (pthread_t *threads, size_t count, void *(*func) (void *))
{
  for (size_t i = 0; i < count; ++i)
    threads[i] = xpthread_create (NULL, func, NULL);
}

static void
join_threads (pthread_t *threads, size_t count)
{
  for (size_t i = 0; i < count; ++i)
    xpthread_join (threads[i]);
}

/* Create a file which consists of a single long line, and assigns
   file_to_read.  The hope is that this triggers an allocation in
   getline which needs a lock.  */
static void
create_file_with_large_line (void)
{
  int fd = create_temp_file ("bug19431-large-line", &file_to_read);
  if (fd < 0)
    {
      printf ("error: create_temp_file: %m\n");
      abort ();
    }
  FILE *f = fdopen (fd, "w+");
  if (f == NULL)
    {
      printf ("error: fdopen: %m\n");
      abort ();
    }
  for (int i = 0; i < 50000; ++i)
    fputc ('x', f);
  fputc ('\n', f);
  if (ferror (f))
    {
      printf ("error: fputc: %m\n");
      abort ();
    }
  if (fclose (f) != 0)
    {
      printf ("error: fclose: %m\n");
      abort ();
    }
}

static int
do_test (void)
{
  /* Make sure that we do not exceed the arena limit with the number
     of threads we configured.  */
  if (mallopt (M_ARENA_MAX, 400) == 0)
    {
      printf ("error: mallopt (M_ARENA_MAX) failed\n");
      return 1;
    }

  /* Leave some room for shutting down all threads gracefully.  */
  int timeout = 3;
  if (timeout > DEFAULT_TIMEOUT)
    timeout = DEFAULT_TIMEOUT - 1;

  create_file_with_large_line ();

  pthread_t fork_threads[fork_thread_count];
  create_threads (fork_threads, fork_thread_count, fork_thread_function);
  pthread_t read_threads[read_thread_count];
  create_threads (read_threads, read_thread_count, read_thread_function);
  pthread_t flushall_threads[1];
  create_threads (flushall_threads, 1, flushall_thread_function);

  struct timespec ts = {timeout, 0};
  if (nanosleep (&ts, NULL))
    {
      printf ("error: error: nanosleep: %m\n");
      abort ();
    }

  __atomic_store_n (&termination_requested, true, __ATOMIC_RELAXED);

  join_threads (flushall_threads, 1);
  join_threads (read_threads, read_thread_count);
  join_threads (fork_threads, fork_thread_count);

  free (file_to_read);

  return 0;
}

#include <support/test-driver.c>
