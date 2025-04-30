/* Test for lutimes.
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

#ifndef struct_stat
# define struct_stat struct stat64
#endif

static int
test_lutimes_helper (const char *testfile, int fd, const char *testlink,
                     const struct timeval *tv)
{
  struct_stat stfile_orig;
  xlstat (testfile, &stfile_orig);

  TEST_VERIFY_EXIT (lutimes (testlink, tv) == 0);

  struct_stat stlink;
  xlstat (testlink, &stlink);

  TEST_COMPARE (stlink.st_atime, tv[0].tv_sec);
  TEST_COMPARE (stlink.st_mtime, tv[1].tv_sec);

  /* Check if the timestamp from original file is not changed.  */
  struct_stat stfile;
  xlstat (testfile, &stfile);

  TEST_COMPARE (stfile_orig.st_atime, stfile.st_atime);
  TEST_COMPARE (stfile_orig.st_mtime, stfile.st_mtime);

  return 0;
}

#define TEST_CALL(fname, fd, lname, v1, v2) \
  test_lutimes_helper (fname, fd, lname, (struct timeval[]) { { v1, 0 }, \
                                                              { v2, 0 } })

#include "tst-utimensat-skeleton.c"
