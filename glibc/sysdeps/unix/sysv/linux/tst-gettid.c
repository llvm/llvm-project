/* Smoke test for the gettid system call.
   Copyright (C) 2019-2021 Free Software Foundation, Inc.
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

#include <support/check.h>
#include <support/namespace.h>
#include <support/xthread.h>
#include <support/xunistd.h>

/* TID of the initial (main) thread.  */
static pid_t initial_tid;

/* Check that PID and TID are the same in a subprocess.  */
static void
subprocess (void *closure)
{
  TEST_COMPARE (getpid (), gettid ());
  TEST_VERIFY (gettid () != initial_tid);
}

/* Check that the TID changes in a new thread.  */
static void *
threadfunc (void *closure)
{
  TEST_VERIFY (getpid () != gettid ());
  TEST_VERIFY (gettid () != initial_tid);
  return NULL;
}

/* Check for interactions with vfork.  */
static void
test_vfork (void)
{
  pid_t proc = vfork ();
  if (proc == 0)
    {
      if (getpid () != gettid ())
        _exit (1);
      if (gettid () == initial_tid)
        _exit (2);
      _exit (0);
    }
  int status;
  xwaitpid (proc, &status, 0);
  TEST_COMPARE (status, 0);
}

static int
do_test (void)
{
  initial_tid = gettid ();

  /* The main thread has the same TID as the PID.  */
  TEST_COMPARE (getpid (), gettid ());

  test_vfork ();

  support_isolate_in_subprocess (subprocess, NULL);

  xpthread_join (xpthread_create (NULL, threadfunc, NULL));

  return 0;
}

#include <support/test-driver.c>
