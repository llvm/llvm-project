/* Check that gettimeofday does not clobber errno.
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

#include <errno.h>
#include <stddef.h>
#include <support/check.h>
#include <sys/time.h>

static int
do_test (void)
{
  for (int original_errno = 0; original_errno < 2; ++original_errno)
    {
      errno = original_errno;
      struct timeval tv;
      gettimeofday (&tv, NULL);
      TEST_COMPARE (errno, original_errno);
    }
  return 0;
}

#include <support/test-driver.c>
