/* Common tests for utimensat routines.
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

#include <array_length.h>
#include <inttypes.h>
#include <support/support.h>
#include <support/temp_file.h>
#include <stdio.h>

static int temp_fd = -1;
static char *testfile;
static char *testlink;

const static struct {
  int64_t v1;
  int64_t v2;
} tests[] = {
  /* Y2038 threshold minus 2 and 1 seconds.  */
  { 0x7FFFFFFELL, 0x7FFFFFFFLL },
  /* Y2038 threshold plus 1 and 2 seconds.  */
  { 0x80000001LL, 0x80000002LL },
  /* Around Y2038 threshold.  */
  { 0x7FFFFFFELL, 0x80000002LL },
  /* Y2106 threshold minus 2 and 1 seconds.  */
  { 0x100000000LL, 0xFFFFFFFELL },
  /* Y2106 threshold plus 1 and 2 seconds.  */
  { 0x100000001LL, 0x100000002LL },
  /* Around Y2106 threshold.  */
  { 0xFFFFFFFELL, 0xFFFFFFFELL },
};

#define PREPARE do_prepare
static void
do_prepare (int argc, char *argv[])
{
  temp_fd = create_temp_file ("utime", &testfile);
  TEST_VERIFY_EXIT (temp_fd > 0);

  testlink = xasprintf ("%s-symlink", testfile);
  xsymlink (testfile, testlink);
  add_temp_file (testlink);
}

static int
do_test (void)
{
  if (!support_path_support_time64 (testfile))
    FAIL_UNSUPPORTED ("File %s does not support 64-bit timestamps",
		      testfile);

  bool y2106 = support_path_support_time64_value (testfile,
						  0x100000001LL,
						  0x100000002LL);

  for (int i = 0; i < array_length (tests); i++)
    {
      /* Check if we run on port with 32 bit time_t size.  */
      time_t t;
      if (__builtin_add_overflow (tests[i].v1, 0, &t)
	  || __builtin_add_overflow (tests[i].v2, 0, &t))
        {
          printf ("warning: skipping tests[%d] { %" PRIx64 ", %" PRIx64 " }: "
		  "time_t overflows\n", i, tests[i].v1, tests[i].v2);
	  continue;
        }

      if (tests[i].v1 >= 0x100000000LL && !y2106)
	{
          printf ("warning: skipping tests[%d] { %" PRIx64 ", %" PRIx64 " }: "
		  "unsupported timestamp value\n",
		  i, tests[i].v1, tests[i].v2);
	  continue;
	}

      TEST_CALL (testfile, temp_fd, testlink, tests[i].v1, tests[i].v2);
    }

  return 0;
}

#include <support/test-driver.c>
