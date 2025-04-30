/* Test that closing O_PATH descriptors does not release POSIX advisory locks.
   Copyright (C) 2020-2021 Free Software Foundation, Inc.
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
#include <stdbool.h>
#include <stdlib.h>
#include <support/check.h>
#include <support/namespace.h>
#include <support/support.h>
#include <support/temp_file.h>
#include <support/xunistd.h>

/* The subprocess writes the errno value of the lock operation
   here.  */
static int *shared_errno;

/* The path of the temporary file which is locked.  */
static char *path;

/* Try to obtain an exclusive lock on the file at path.  */
static void
subprocess (void *closure)
{
  int fd = xopen (path, O_RDWR, 0);
  struct flock64 lock = { .l_type = F_WRLCK, };
  int ret = fcntl64 (fd, F_SETLK, &lock);
  if (ret == 0)
    *shared_errno = 0;
  else
    *shared_errno = errno;
  xclose (fd);
}

/* Return true if the file at path is currently locked, false if
   not.  */
static bool
probe_lock (void)
{
  *shared_errno = -1;
  support_isolate_in_subprocess (subprocess, NULL);
  if (*shared_errno == 0)
    /* Lock was aquired by the subprocess, so this process has not
       locked it.  */
    return false;
  else
    {
      /* POSIX allows both EACCES and EAGAIN.  Linux use EACCES.  */
      TEST_COMPARE (*shared_errno, EAGAIN);
      return true;
    }
}

static int
do_test (void)
{
  shared_errno = support_shared_allocate (sizeof (*shared_errno));
  int fd = create_temp_file ("tst-o_path-locks-", &path);

  /* The file is not locked initially.  */
  TEST_VERIFY (!probe_lock ());

  struct flock64 lock = { .l_type = F_WRLCK, };
  TEST_COMPARE (fcntl64 (fd, F_SETLK, &lock), 0);

  /* The lock has been acquired.  */
  TEST_VERIFY (probe_lock ());

  /* Closing the same file via a different descriptor releases the
     lock.  */
  xclose (xopen (path, O_RDONLY, 0));
  TEST_VERIFY (!probe_lock ());

  /* But not if it is an O_PATH descriptor.  */
  TEST_COMPARE (fcntl64 (fd, F_SETLK, &lock), 0);
  xclose (xopen (path, O_PATH, 0));
  TEST_VERIFY (probe_lock ());

  xclose (fd);
  free (path);
  support_shared_free (shared_errno);
  return 0;
}

#include <support/test-driver.c>
