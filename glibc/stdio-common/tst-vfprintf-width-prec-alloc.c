/* Test large width or precision does not involve large allocation.
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

#include <stdio.h>
#include <sys/resource.h>
#include <support/check.h>

char test_string[] = "test";

static int
do_test (void)
{
  struct rlimit limit;
  TEST_VERIFY_EXIT (getrlimit (RLIMIT_AS, &limit) == 0);
  limit.rlim_cur = 200 * 1024 * 1024;
  TEST_VERIFY_EXIT (setrlimit (RLIMIT_AS, &limit) == 0);
  FILE *fp = fopen ("/dev/null", "w");
  TEST_VERIFY_EXIT (fp != NULL);
  TEST_COMPARE (fprintf (fp, "%1000000000d", 1), 1000000000);
  TEST_COMPARE (fprintf (fp, "%.1000000000s", test_string), 4);
  TEST_COMPARE (fprintf (fp, "%1000000000d %1000000000d", 1, 2), 2000000001);
  TEST_COMPARE (fprintf (fp, "%2$.*1$s", 0x7fffffff, test_string), 4);
  return 0;
}

#include <support/test-driver.c>
