/* Basic test for the TEST_COMPARE_BLOB macro.
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
  TEST_COMPARE_BLOB ("", 0, "", 1);    /* Line 29.  */
  TEST_COMPARE_BLOB ("X", 1, "", 1);   /* Line 30.  */
  TEST_COMPARE_BLOB ("abcd", 3, "abcd", 4); /* Line 31.  */
  TEST_COMPARE_BLOB ("abcd", 4, "abcD", 4); /* Line 32.  */
  TEST_COMPARE_BLOB ("abcd", 4, NULL, 0); /* Line 33.  */
  TEST_COMPARE_BLOB (NULL, 0, "abcd", 4); /* Line 34.  */
}

/* Same contents, different addresses.  */
char buffer_abc_1[] = "abc";
char buffer_abc_2[] = "abc";

static int
do_test (void)
{
  /* This should succeed.  Even if the pointers and array contents are
     different, zero-length inputs are not different.  */
  TEST_COMPARE_BLOB ("", 0, "", 0);
  TEST_COMPARE_BLOB ("", 0, buffer_abc_1, 0);
  TEST_COMPARE_BLOB (buffer_abc_1, 0, "", 0);
  TEST_COMPARE_BLOB (NULL, 0, "", 0);
  TEST_COMPARE_BLOB ("", 0, NULL, 0);
  TEST_COMPARE_BLOB (NULL, 0, NULL, 0);

  /* Check equality of blobs containing a single NUL byte.  */
  TEST_COMPARE_BLOB ("", 1, "", 1);
  TEST_COMPARE_BLOB ("", 1, &buffer_abc_1[3], 1);

  /* Check equality of blobs of varying lengths.  */
  for (size_t i = 0; i <= sizeof (buffer_abc_1); ++i)
    TEST_COMPARE_BLOB (buffer_abc_1, i, buffer_abc_2, i);

  struct support_capture_subprocess proc = support_capture_subprocess
    (&subprocess, NULL);

  /* Discard the reported error.  */
  support_record_failure_reset ();

  puts ("info: *** subprocess output starts ***");
  fputs (proc.out.buffer, stdout);
  puts ("info: *** subprocess output ends ***");

  TEST_VERIFY
    (strcmp (proc.out.buffer,
"tst-test_compare_blob.c:29: error: blob comparison failed\n"
"  left length:  0 bytes (from 0)\n"
"  right length: 1 bytes (from 1)\n"
"  right (evaluated from \"\"):\n"
"      \"\\000\"\n"
"      00\n"
"tst-test_compare_blob.c:30: error: blob comparison failed\n"
"  blob length: 1 bytes\n"
"  left (evaluated from \"X\"):\n"
"      \"X\"\n"
"      58\n"
"  right (evaluated from \"\"):\n"
"      \"\\000\"\n"
"      00\n"
"tst-test_compare_blob.c:31: error: blob comparison failed\n"
"  left length:  3 bytes (from 3)\n"
"  right length: 4 bytes (from 4)\n"
"  left (evaluated from \"abcd\"):\n"
"      \"abc\"\n"
"      61 62 63\n"
"  right (evaluated from \"abcd\"):\n"
"      \"abcd\"\n"
"      61 62 63 64\n"
"tst-test_compare_blob.c:32: error: blob comparison failed\n"
"  blob length: 4 bytes\n"
"  left (evaluated from \"abcd\"):\n"
"      \"abcd\"\n"
"      61 62 63 64\n"
"  right (evaluated from \"abcD\"):\n"
"      \"abcD\"\n"
"      61 62 63 44\n"
"tst-test_compare_blob.c:33: error: blob comparison failed\n"
"  left length:  4 bytes (from 4)\n"
"  right length: 0 bytes (from 0)\n"
"  left (evaluated from \"abcd\"):\n"
"      \"abcd\"\n"
"      61 62 63 64\n"
"tst-test_compare_blob.c:34: error: blob comparison failed\n"
"  left length:  0 bytes (from 0)\n"
"  right length: 4 bytes (from 4)\n"
"  right (evaluated from \"abcd\"):\n"
"      \"abcd\"\n"
"      61 62 63 64\n"
             ) == 0);

  /* Check that there is no output on standard error.  */
  support_capture_subprocess_check (&proc, "TEST_COMPARE_BLOB",
                                    0, sc_allow_stdout);

  return 0;
}

#include <support/test-driver.c>
