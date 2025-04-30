/* Test for utime.
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

#include <utime.h>
#include <support/check.h>
#include <support/xunistd.h>
#include <sys/stat.h>

#ifndef struct_stat
# define struct_stat struct stat64
#endif

static int
test_utime_helper (const char *file, int fd, const struct utimbuf *ut)
{
  int result = utime (file, ut);
  TEST_VERIFY_EXIT (result == 0);

  struct_stat st;
  xfstat (fd, &st);

  /* Check if seconds for actime match */
  TEST_COMPARE (st.st_atime, ut->actime);

  /* Check if seconds for modtime match */
  TEST_COMPARE (st.st_mtime, ut->modtime);

  return 0;
}

#define TEST_CALL(fname, fd, lname, v1, v2) \
  test_utime_helper (fname, fd, &(struct utimbuf) { (v1), (v2) })

#include "tst-utimensat-skeleton.c"
