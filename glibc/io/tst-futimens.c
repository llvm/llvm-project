/* Test for futimens
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

#include <support/check.h>
#include <support/xunistd.h>
#include <sys/stat.h>

#ifndef struct_stat
# define struct_stat struct stat64
#endif

static int
test_futimens_helper (const char *file, int fd, const struct timespec *ts)
{
  int result = futimens (fd, ts);
  TEST_VERIFY_EXIT (result == 0);

  struct_stat st;
  xfstat (fd, &st);

  /* Check if seconds for atime match */
  TEST_COMPARE (st.st_atime, ts[0].tv_sec);

  /* Check if seconds for mtime match */
  TEST_COMPARE (st.st_mtime, ts[1].tv_sec);

  return 0;
}

#define TEST_CALL(fname, fd, lname, v1, v2) \
  test_futimens_helper (fname, fd, (struct timespec[]) { { v1, 0 }, \
							 { v2, 0 } })

#include "tst-utimensat-skeleton.c"
