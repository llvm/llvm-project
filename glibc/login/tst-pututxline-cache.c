/* Test case for cache invalidation after concurrent write (bug 24882).
   Copyright (C) 2019-2021 Free Software Foundation, Inc.
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
   not, see <http://www.gnu.org/licenses/>.  */

/* This test writes an entry to the utmpx file, reads it (so that it
   is cached) in process1, and overwrites the same entry in process2
   with something that does not match the search criteria.  At this
   point, the cache of the first process is stale, and when process1
   attempts to write a new record which would have gone to the same
   place (as indicated by the cache), it needs to realize that it has
   to pick a different slot because the old slot is now used for
   something else.  */

#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include <support/check.h>
#include <support/namespace.h>
#include <support/support.h>
#include <support/temp_file.h>
#include <support/xthread.h>
#include <support/xunistd.h>
#include <utmp.h>
#include <utmpx.h>

/* Set to the path of the utmp file.  */
static char *utmp_file;

/* Used to synchronize the subprocesses.  The barrier itself is
   allocated in shared memory.  */
static pthread_barrier_t *barrier;

/* setutxent with error checking.  */
static void
xsetutxent (void)
{
  errno = 0;
  setutxent ();
  TEST_COMPARE (errno, 0);
}

/* getutxent with error checking.  */
static struct utmpx *
xgetutxent (void)
{
  errno = 0;
  struct utmpx *result = getutxent ();
  if (result == NULL)
    FAIL_EXIT1 ("getutxent: %m");
  return result;
}

static void
put_entry (const char *id, pid_t pid, const char *user, const char *line)
{
  struct utmpx ut =
    {
     .ut_type = LOGIN_PROCESS,
     .ut_pid = pid,
     .ut_host = "localhost",
    };
  strcpy (ut.ut_id, id);
  strncpy (ut.ut_user, user, sizeof (ut.ut_user));
  strncpy (ut.ut_line, line, sizeof (ut.ut_line));
  TEST_VERIFY (pututxline (&ut) != NULL);
}

/* Use two cooperating subprocesses to avoid issues related to
   unlock-on-close semantics of POSIX advisory locks.  */

static __attribute__ ((noreturn)) void
process1 (void)
{
  TEST_COMPARE (utmpname (utmp_file), 0);

  /* Create an entry.  */
  xsetutxent ();
  put_entry ("1", 101, "root", "process1");

  /* Retrieve the entry.  This will fill the internal cache.  */
  {
    errno = 0;
    setutxent ();
    TEST_COMPARE (errno, 0);
    struct utmpx ut =
      {
       .ut_type = LOGIN_PROCESS,
       .ut_line = "process1",
      };
    struct utmpx *result = getutxline (&ut);
    if (result == NULL)
      FAIL_EXIT1 ("getutxline (\"process1\"): %m");
    TEST_COMPARE (result->ut_pid, 101);
  }

  /* Signal the other process to overwrite the entry.  */
  xpthread_barrier_wait (barrier);

  /* Wait for the other process to complete the write operation.  */
  xpthread_barrier_wait (barrier);

  /* Add another entry.  Note: This time, there is no setutxent call.  */
  put_entry ("1", 103, "root", "process1");

  _exit (0);
}

static void
process2 (void *closure)
{
  /* Wait for the first process to write its entry.  */
  xpthread_barrier_wait (barrier);

  /* Truncate the file.  The glibc interface does not support
     re-purposing records, but an external expiration mechanism may
     trigger this.  */
  TEST_COMPARE (truncate64 (utmp_file, 0), 0);

  /* Write the replacement entry.  */
  TEST_COMPARE (utmpname (utmp_file), 0);
  xsetutxent ();
  put_entry ("2", 102, "user", "process2");

  /* Signal the other process that the entry has been replaced.  */
  xpthread_barrier_wait (barrier);
}

static int
do_test (void)
{
  xclose (create_temp_file ("tst-tumpx-cache-write-", &utmp_file));
  {
    pthread_barrierattr_t attr;
    xpthread_barrierattr_init (&attr);
    xpthread_barrierattr_setpshared (&attr, PTHREAD_SCOPE_PROCESS);
    barrier = support_shared_allocate (sizeof (*barrier));
    xpthread_barrier_init (barrier, &attr, 2);
  }

  /* Run both subprocesses in parallel.  */
  {
    pid_t pid1 = xfork ();
    if (pid1 == 0)
      process1 ();
    support_isolate_in_subprocess (process2, NULL);
    int status;
    xwaitpid (pid1, &status, 0);
    TEST_COMPARE (status, 0);
  }

  /* Check that the utmpx database contains the expected records.  */
  {
    TEST_COMPARE (utmpname (utmp_file), 0);
    xsetutxent ();

    struct utmpx *ut = xgetutxent ();
    TEST_COMPARE_STRING (ut->ut_id, "2");
    TEST_COMPARE (ut->ut_pid, 102);
    TEST_COMPARE_STRING (ut->ut_user, "user");
    TEST_COMPARE_STRING (ut->ut_line, "process2");

    ut = xgetutxent ();
    TEST_COMPARE_STRING (ut->ut_id, "1");
    TEST_COMPARE (ut->ut_pid, 103);
    TEST_COMPARE_STRING (ut->ut_user, "root");
    TEST_COMPARE_STRING (ut->ut_line, "process1");

    if (getutxent () != NULL)
      FAIL_EXIT1 ("additional utmpx entry");
  }

  xpthread_barrier_destroy (barrier);
  support_shared_free (barrier);
  free (utmp_file);

  return 0;
}

#include <support/test-driver.c>
