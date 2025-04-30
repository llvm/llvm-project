/* Basic test for the TEST_COMPARE_STRING macro.
   Copyright (C) 2018-2021 Free Software Foundation, Inc.
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

#include <string.h>
#include <support/check.h>
#include <support/capture_subprocess.h>

static void
subprocess (void *closure)
{
  /* These tests should fail.  They were chosen to cover differences
     in length (with the same contents), single-bit mismatches, and
     mismatching null pointers.  */
  TEST_COMPARE_STRING ("", NULL);             /* Line 29.  */
  TEST_COMPARE_STRING ("X", "");              /* Line 30.  */
  TEST_COMPARE_STRING (NULL, "X");            /* Line 31.  */
  TEST_COMPARE_STRING ("abcd", "abcD");       /* Line 32.  */
  TEST_COMPARE_STRING ("abcd", NULL);         /* Line 33.  */
  TEST_COMPARE_STRING (NULL, "abcd");         /* Line 34.  */
}

/* Same contents, different addresses.  */
char buffer_abc_1[] = "abc";
char buffer_abc_2[] = "abc";

static int
do_test (void)
{
  /* This should succeed.  Even if the pointers and array contents are
     different, zero-length inputs are not different.  */
  TEST_COMPARE_STRING (NULL, NULL);
  TEST_COMPARE_STRING ("", "");
  TEST_COMPARE_STRING (buffer_abc_1, buffer_abc_2);
  TEST_COMPARE_STRING (buffer_abc_1, "abc");

  struct support_capture_subprocess proc = support_capture_subprocess
    (&subprocess, NULL);

  /* Discard the reported error.  */
  support_record_failure_reset ();

  puts ("info: *** subprocess output starts ***");
  fputs (proc.out.buffer, stdout);
  puts ("info: *** subprocess output ends ***");

  TEST_VERIFY
    (strcmp (proc.out.buffer,
"tst-test_compare_string.c:29: error: string comparison failed\n"
"  left string: 0 bytes\n"
"  right string: NULL\n"
"tst-test_compare_string.c:30: error: string comparison failed\n"
"  left string: 1 bytes\n"
"  right string: 0 bytes\n"
"  left (evaluated from \"X\"):\n"
"      \"X\"\n"
"      58\n"
"tst-test_compare_string.c:31: error: string comparison failed\n"
"  left string: NULL\n"
"  right string: 1 bytes\n"
"  right (evaluated from \"X\"):\n"
"      \"X\"\n"
"      58\n"
"tst-test_compare_string.c:32: error: string comparison failed\n"
"  string length: 4 bytes\n"
"  left (evaluated from \"abcd\"):\n"
"      \"abcd\"\n"
"      61 62 63 64\n"
"  right (evaluated from \"abcD\"):\n"
"      \"abcD\"\n"
"      61 62 63 44\n"
"tst-test_compare_string.c:33: error: string comparison failed\n"
"  left string: 4 bytes\n"
"  right string: NULL\n"
"  left (evaluated from \"abcd\"):\n"
"      \"abcd\"\n"
"      61 62 63 64\n"
"tst-test_compare_string.c:34: error: string comparison failed\n"
"  left string: NULL\n"
"  right string: 4 bytes\n"
"  right (evaluated from \"abcd\"):\n"
"      \"abcd\"\n"
"      61 62 63 64\n"
             ) == 0);

  /* Check that there is no output on standard error.  */
  support_capture_subprocess_check (&proc, "TEST_COMPARE_STRING",
                                    0, sc_allow_stdout);

  return 0;
}

#include <support/test-driver.c>
