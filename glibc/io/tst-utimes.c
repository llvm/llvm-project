/* Test for utimes
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
#include <sys/time.h>
#include <time.h>

#ifndef struct_stat
# define struct_stat struct stat64
#endif

static int
test_utimes_helper (const char *file, int fd, const struct timeval *tv)
{
  int result = utimes (file, tv);
  TEST_VERIFY_EXIT (result == 0);

  struct_stat st;
  xfstat (fd, &st);

  /* Check if seconds for atime match */
  TEST_COMPARE (st.st_atime, tv[0].tv_sec);

  /* Check if seconds for mtime match */
  TEST_COMPARE (st.st_mtime, tv[1].tv_sec);

  return 0;
}

#define TEST_CALL(fname, fd, lname, v1, v2) \
  test_utimes_helper (fname, fd, (struct timeval[]) { { v1, 0 }, \
						      { v2, 0 } })

#include "tst-utimensat-skeleton.c"
