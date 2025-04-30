/* Basic tests for _Fork.
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
   <https://www.gnu.org/licenses/>.  */

#include <array_length.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <stdbool.h>
#include <support/check.h>
#include <support/xsignal.h>
#include <support/temp_file.h>
#include <support/xunistd.h>
#include <support/xthread.h>

/* For single-thread, _Fork behaves like fork.  */
static int
singlethread_test (void)
{
  const char testdata1[] = "abcdefghijklmnopqrtuvwxz";
  enum { testdatalen1 = array_length (testdata1) };
  const char testdata2[] = "01234567890";
  enum { testdatalen2 = array_length (testdata2) };

  pid_t ppid = getpid ();

  int tempfd = create_temp_file ("tst-_Fork", NULL);

  /* Check if the opened file is shared between process by read and write
     some data on parent and child processes.  */
  xwrite (tempfd, testdata1, testdatalen1);
  off_t off = xlseek (tempfd, 0, SEEK_CUR);
  TEST_COMPARE (off, testdatalen1);

  pid_t pid = _Fork ();
  TEST_VERIFY_EXIT (pid != -1);
  if (pid == 0)
    {
      TEST_VERIFY_EXIT (getpid () != ppid);
      TEST_COMPARE (getppid(), ppid);

      TEST_COMPARE (xlseek (tempfd, 0, SEEK_CUR), testdatalen1);

      xlseek (tempfd, 0, SEEK_SET);
      char buf[testdatalen1];
      TEST_COMPARE (read (tempfd, buf, sizeof (buf)), testdatalen1);
      TEST_COMPARE_BLOB (buf, testdatalen1, testdata1, testdatalen1);

      xlseek (tempfd, 0, SEEK_SET);
      xwrite (tempfd, testdata2, testdatalen2);

      xclose (tempfd);

      _exit (EXIT_SUCCESS);
    }

  int status;
  xwaitpid (pid, &status, 0);
  TEST_VERIFY (WIFEXITED (status));
  TEST_COMPARE (WEXITSTATUS (status), EXIT_SUCCESS);

  TEST_COMPARE (xlseek (tempfd, 0, SEEK_CUR), testdatalen2);

  xlseek (tempfd, 0, SEEK_SET);
  char buf[testdatalen2];
  TEST_COMPARE (read (tempfd, buf, sizeof (buf)), testdatalen2);

  TEST_COMPARE_BLOB (buf, testdatalen2, testdata2, testdatalen2);

  return 0;
}


static volatile sig_atomic_t sigusr1_handler_ran;
#define SIG_PID_EXIT_CODE 20

static bool atfork_prepare_var;
static bool atfork_parent_var;
static bool atfork_child_var;

static void
atfork_prepare (void)
{
  atfork_prepare_var = true;
}

static void
atfork_parent (void)
{
  atfork_parent_var = true;
}

static void
atfork_child (void)
{
  atfork_child_var = true;
}

/* Different than fork, _Fork does not execute any pthread_atfork
   handlers.  */
static int
singlethread_atfork_test (void)
{
  pthread_atfork (atfork_prepare, atfork_parent, atfork_child);
  singlethread_test ();
  TEST_VERIFY (!atfork_prepare_var);
  TEST_VERIFY (!atfork_parent_var);
  TEST_VERIFY (!atfork_child_var);

  return 0;
}

static void *
mt_atfork_test (void *args)
{
  singlethread_atfork_test ();

  return NULL;
}

static int
multithread_atfork_test (void)
{
  pthread_t thr = xpthread_create (NULL, mt_atfork_test, NULL);
  xpthread_join (thr);

  return 0;
}


static int
do_test (void)
{
  singlethread_atfork_test ();
  multithread_atfork_test ();

  return 0;
}

#include <support/test-driver.c>
