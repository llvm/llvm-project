/* Test driver for malloc interposition tests.
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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#if INTERPOSE_THREADS
#include <pthread.h>
#endif

static int do_test (void);
#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"

/* Fills BUFFER with a test string.  */
static void
line_string (int number, char *buffer, size_t length)
{
  for (size_t i = 0; i < length - 2; ++i)
    buffer[i] = 'A' + ((number + i) % 26);
  buffer[length - 2] = '\n';
  buffer[length - 1] = '\0';
}

/* Perform the tests.  */
static void *
run_tests (void *closure)
{
  char *temp_file_path;
  int fd = create_temp_file ("tst-malloc-interpose", &temp_file_path);
  if (fd < 0)
    _exit (1);

  /* Line lengths excluding the line terminator.  */
  static const int line_lengths[] = { 0, 45, 80, 2, 8201, 0, 17, -1 };

  /* Fill the test file with data.  */
  {
    FILE *fp = fdopen (fd, "w");
    for (int lineno = 0; line_lengths[lineno] >= 0; ++lineno)
      {
        char buffer[line_lengths[lineno] + 2];
        line_string (lineno, buffer, sizeof (buffer));
        fprintf (fp, "%s", buffer);
      }

    if (ferror (fp))
      {
        printf ("error: fprintf: %m\n");
        _exit (1);
      }
    if (fclose (fp) != 0)
      {
        printf ("error: fclose: %m\n");
        _exit (1);
      }
  }

  /* Read the test file.  This tests libc-internal allocation with
     realloc.  */
  {
    FILE *fp = fopen (temp_file_path, "r");

    char *actual = NULL;
    size_t actual_size = 0;
    for (int lineno = 0; ; ++lineno)
      {
        errno = 0;
        ssize_t result = getline (&actual, &actual_size, fp);
        if (result == 0)
          {
            printf ("error: invalid return value 0 from getline\n");
            _exit (1);
          }
        if (result < 0 && errno != 0)
          {
            printf ("error: getline: %m\n");
            _exit (1);
          }
        if (result < 0 && line_lengths[lineno] >= 0)
          {
            printf ("error: unexpected end of file after line %d\n", lineno);
            _exit (1);
          }
        if (result > 0 && line_lengths[lineno] < 0)
          {
            printf ("error: no end of file after line %d\n", lineno);
            _exit (1);
          }
        if (result == -1 && line_lengths[lineno] == -1)
          /* End of file reached as expected.  */
          break;

        if (result != line_lengths[lineno] + 1)
          {
            printf ("error: line length mismatch: expected %d, got %zd\n",
                    line_lengths[lineno], result);
            _exit (1);
          }

        char expected[line_lengths[lineno] + 2];
        line_string (lineno, expected, sizeof (expected));
        if (strcmp (actual, expected) != 0)
          {
            printf ("error: line mismatch\n");
            printf ("error:   expected: [[%s]]\n", expected);
            printf ("error:   actual:   [[%s]]\n", actual);
            _exit (1);
          }
      }

    if (fclose (fp) != 0)
      {
        printf ("error: fclose (after reading): %m\n");
        _exit (1);
      }
  }

  free (temp_file_path);

  /* Make sure that fork is working.  */
  pid_t pid = fork ();
  if (pid == -1)
    {
      printf ("error: fork: %m\n");
      _exit (1);
    }
  enum { exit_code = 55 };
  if (pid == 0)
    _exit (exit_code);
  int status;
  int ret = waitpid (pid, &status, 0);
  if (ret < 0)
    {
      printf ("error: waitpid: %m\n");
      _exit (1);
    }
  if (!WIFEXITED (status) || WEXITSTATUS (status) != exit_code)
    {
      printf ("error: unexpected exit status from child process: %d\n",
              status);
      _exit (1);
    }

  return NULL;
}

/* This is used to detect if malloc has not been successfully
   interposed.  The interposed malloc does not use brk/sbrk.  */
static void *initial_brk;
__attribute__ ((constructor))
static void
set_initial_brk (void)
{
  initial_brk = sbrk (0);
}

/* Terminate the process if the break value has been changed.  */
__attribute__ ((destructor))
static void
check_brk (void)
{
  void *current = sbrk (0);
  if (current != initial_brk)
    {
      printf ("error: brk changed from %p to %p; no interposition?\n",
              initial_brk, current);
      _exit (1);
    }
}

static int
do_test (void)
{
  check_brk ();

#if INTERPOSE_THREADS
  pthread_t thr = xpthread_create (NULL, run_tests, NULL);
  xpthread_join (thr);
#else
  run_tests (NULL);
#endif

  check_brk ();

  return 0;
}
