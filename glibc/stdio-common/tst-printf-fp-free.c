/* Test double free bug in __printf_fp_l (bug 26214).
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

#include <mcheck.h>
#include <stdio.h>
#include <sys/resource.h>
#include <support/check.h>

static int
do_test (void)
{
  mtrace ();
  FILE *fp = fopen ("/dev/full", "w");
  TEST_VERIFY_EXIT (fp != NULL);
  char buf[131072];
  TEST_VERIFY_EXIT (setvbuf (fp, buf, _IOFBF, sizeof buf) == 0);
  TEST_COMPARE (fprintf (fp, "%-1000000.65536f", 1.0), -1);
  fclose (fp);
  return 0;
}

#include <support/test-driver.c>
