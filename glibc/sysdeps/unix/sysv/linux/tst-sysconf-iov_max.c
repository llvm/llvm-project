/* Check IOV_MAX definition for consistency (bug 22321).
   Copyright (C) 2017-2021 Free Software Foundation, Inc.
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

/* Defined in tst-sysconf-iov_max-uapi.c.  */
long uio_maxiov_value (void);


#include <limits.h>
#include <support/check.h>
#include <sys/uio.h>
#include <unistd.h>

static int
do_test (void)
{
  TEST_VERIFY (_XOPEN_IOV_MAX == 16); /* Value required by POSIX.  */
  TEST_VERIFY (uio_maxiov_value () >= _XOPEN_IOV_MAX);
  TEST_VERIFY (IOV_MAX == uio_maxiov_value ());
  TEST_VERIFY (UIO_MAXIOV == uio_maxiov_value ());
  TEST_VERIFY (sysconf (_SC_UIO_MAXIOV) == uio_maxiov_value ());
  TEST_VERIFY (sysconf (_SC_IOV_MAX) == uio_maxiov_value ());
  return 0;
}

#include <support/test-driver.c>
