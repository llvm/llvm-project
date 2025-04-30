/* Test the lock upgrade path in tst-pututxline.
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
   not, see <https://www.gnu.org/licenses/>.  */

/* pututxline upgrades the read lock on the file to a write lock.
   This test verifies that if the lock upgrade fails, the utmp
   subsystem remains in a consistent state, so that pututxline can be
   called again.  */

#include <errno.h>
#include <fcntl.h>
#include <stdlib.h>
#include <support/check.h>
#include <support/namespace.h>
#include <support/support.h>
#include <support/temp_file.h>
#include <support/xthread.h>
#include <support/xunistd.h>
#include <unistd.h>
#include <utmp.h>
#include <utmpx.h>

/* Path to the temporary utmp file.   */
static char *path;

/* Used to synchronize the subprocesses.  The barrier itself is
   allocated in shared memory.  */
static pthread_barrier_t *barrier;

/* Use pututxline to write an entry for PID.  */
static struct utmpx *
write_entry (pid_t pid)
{
  struct utmpx ut =
    {
     .ut_type = LOGIN_PROCESS,
     .ut_id = "1",
     .ut_user = "root",
     .ut_pid = pid,
     .ut_line = "entry",
     .ut_host = "localhost",
    };
  return pututxline (&ut);
}

/* Create the initial entry in a subprocess, so that the utmp
   subsystem in the original process is not disturbed.  */
static void
subprocess_create_entry (void *closure)
{
  TEST_COMPARE (utmpname (path), 0);
  TEST_VERIFY (write_entry (101) != NULL);
}

/* Acquire an advisory read lock on PATH.  */
__attribute__ ((noreturn)) static void
subprocess_lock_file (void)
{
  int fd = xopen (path, O_RDONLY, 0);

  struct flock64 fl =
    {
     .l_type = F_RDLCK,
     fl.l_whence = SEEK_SET,
    };
  TEST_COMPARE (fcntl64 (fd, F_SETLKW, &fl), 0);

  /* Signal to the main process that the lock has been acquired.  */
  xpthread_barrier_wait (barrier);

  /* Wait for the unlock request from the main process.  */
  xpthread_barrier_wait (barrier);

  /* Implicitly unlock the file.  */
  xclose (fd);

  /* Overwrite the existing entry.  */
  TEST_COMPARE (utmpname (path), 0);
  errno = 0;
  setutxent ();
  TEST_COMPARE (errno, 0);
  TEST_VERIFY (write_entry (102) != NULL);
  errno = 0;
  endutxent ();
  TEST_COMPARE (errno, 0);

  _exit (0);
}

static int
do_test (void)
{
  xclose (create_temp_file ("tst-pututxline-lockfail-", &path));

  {
    pthread_barrierattr_t attr;
    xpthread_barrierattr_init (&attr);
    xpthread_barrierattr_setpshared (&attr, PTHREAD_SCOPE_PROCESS);
    barrier = support_shared_allocate (sizeof (*barrier));
    xpthread_barrier_init (barrier, &attr, 2);
    xpthread_barrierattr_destroy (&attr);
  }

  /* Write the initial entry.  */
  support_isolate_in_subprocess (subprocess_create_entry, NULL);

  pid_t locker_pid = xfork ();
  if (locker_pid == 0)
    subprocess_lock_file ();

  /* Wait for the file locking to complete.  */
  xpthread_barrier_wait (barrier);

  /* Try to add another entry.  This attempt will fail, with EINTR or
     EAGAIN.  */
  TEST_COMPARE (utmpname (path), 0);
  TEST_VERIFY (write_entry (102) == NULL);
  if (errno != EINTR)
    TEST_COMPARE (errno, EAGAIN);

  /* Signal the subprocess to overwrite the entry.  */
  xpthread_barrier_wait (barrier);

  /* Wait for write and unlock to complete.  */
  {
    int status;
    xwaitpid (locker_pid, &status, 0);
    TEST_COMPARE (status, 0);
  }

  /* The file is no longer locked, so this operation will succeed.  */
  TEST_VERIFY (write_entry (103) != NULL);
  errno = 0;
  endutxent ();
  TEST_COMPARE (errno, 0);

  /* Check that there is just one entry with the expected contents.
     If pututxline becomes desynchronized internally, the entry is not
     overwritten (bug 24902).  */
  errno = 0;
  setutxent ();
  TEST_COMPARE (errno, 0);
  struct utmpx *ut = getutxent ();
  TEST_VERIFY_EXIT (ut != NULL);
  TEST_COMPARE (ut->ut_type, LOGIN_PROCESS);
  TEST_COMPARE_STRING (ut->ut_id, "1");
  TEST_COMPARE_STRING (ut->ut_user, "root");
  TEST_COMPARE (ut->ut_pid, 103);
  TEST_COMPARE_STRING (ut->ut_line, "entry");
  TEST_COMPARE_STRING (ut->ut_host, "localhost");
  TEST_VERIFY (getutxent () == NULL);
  errno = 0;
  endutxent ();
  TEST_COMPARE (errno, 0);

  xpthread_barrier_destroy (barrier);
  support_shared_free (barrier);
  free (path);
  return 0;
}

#include <support/test-driver.c>
