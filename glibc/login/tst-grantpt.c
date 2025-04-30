/* Test for grantpt, unlockpt error corner cases.
   Copyright (C) 2001-2021 Free Software Foundation, Inc.
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

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>
#include <unistd.h>
#include <support/check.h>
#include <support/temp_file.h>
#include <support/xunistd.h>

/* Test grantpt, unlockpt with a closed descriptor.  */
static void
test_ebadf (void)
{
  int fd, ret, err;

  fd = posix_openpt (O_RDWR);
  if (fd == -1)
    FAIL_EXIT1 ("posix_openpt(O_RDWR) failed\nerrno %d (%m)\n", errno);
  TEST_COMPARE (unlockpt (fd), 0);

  xclose (fd);
  ret = grantpt (fd);
  err = errno;
  if (ret != -1 || err != EBADF)
    {
      support_record_failure ();
      printf ("grantpt(): expected: return = %d, errno = %d\n", -1, EBADF);
      printf ("           got: return = %d, errno = %d\n", ret, err);
    }

  TEST_COMPARE (unlockpt (fd), -1);
  TEST_COMPARE (errno, EBADF);
}

/* Test grantpt, unlockpt on a regular file.  */
static void
test_einval (void)
{
  int fd, ret, err;

  fd = create_temp_file ("tst-grantpt-", NULL);
  TEST_VERIFY_EXIT (fd >= 0);

  ret = grantpt (fd);
  err = errno;
  if (ret != -1 || err != EINVAL)
    {
      support_record_failure ();
      printf ("grantpt(): expected: return = %d, errno = %d\n", -1, EINVAL);
      printf ("           got: return = %d, errno = %d\n", ret, err);
    }

  TEST_COMPARE (unlockpt (fd), -1);
  TEST_COMPARE (errno, EINVAL);

  xclose (fd);
}

/* Test grantpt, unlockpt on a non-ptmx pseudo-terminal.  */
static void
test_not_ptmx (void)
{
  int ptmx = posix_openpt (O_RDWR);
  TEST_VERIFY_EXIT (ptmx >= 0);
  TEST_COMPARE (grantpt (ptmx), 0);
  TEST_COMPARE (unlockpt (ptmx), 0);

  /* A second unlock succeeds as well.  */
  TEST_COMPARE (unlockpt (ptmx), 0);

  const char *name = ptsname (ptmx);
  TEST_VERIFY_EXIT (name != NULL);
  int pts = open (name, O_RDWR | O_NOCTTY);
  TEST_VERIFY_EXIT (pts >= 0);

  TEST_COMPARE (grantpt (pts), -1);
  TEST_COMPARE (errno, EINVAL);

  TEST_COMPARE (unlockpt (pts), -1);
  TEST_COMPARE (errno, EINVAL);

  xclose (pts);
  xclose (ptmx);
}

static int
do_test (void)
{
  test_ebadf ();
  test_einval ();
  test_not_ptmx ();
  return 0;
}

#include <support/test-driver.c>
