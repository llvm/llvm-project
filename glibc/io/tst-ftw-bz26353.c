/* test ftw bz26353: Check whether stack overflow occurs when the value
   of the nopenfd parameter is too large.

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

#include <stdlib.h>
#include <stdio.h>
#include <ftw.h>
#include <errno.h>
#include <sys/resource.h>

#include <support/temp_file.h>
#include <support/capture_subprocess.h>
#include <support/check.h>

static int
my_func (const char *file, const struct stat *sb, int flag)
{
  return 0;
}

static int
get_large_nopenfd (void)
{
  struct rlimit r;
  TEST_COMPARE (getrlimit (RLIMIT_STACK, &r), 0);
  if (r.rlim_cur == RLIM_INFINITY)
    {
      r.rlim_cur = 8 * 1024 * 1024;
      TEST_COMPARE (setrlimit (RLIMIT_STACK, &r), 0);
    }
  return (int) r.rlim_cur;
}

static void
do_ftw (void *unused)
{
  char *tempdir = support_create_temp_directory ("tst-bz26353");
  int large_nopenfd = get_large_nopenfd ();
  TEST_COMPARE (ftw (tempdir, my_func, large_nopenfd), 0);
  free (tempdir);
}

/* Check whether stack overflow occurs.  */
static int
do_test (void)
{
  struct support_capture_subprocess result;
  result = support_capture_subprocess (do_ftw, NULL);
  support_capture_subprocess_check (&result, "bz26353", 0, sc_allow_none);
  support_capture_subprocess_free (&result);
  return 0;
}

#include <support/test-driver.c>
