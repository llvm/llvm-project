/* Basic test for the TEST_COMPARE macro.
   Copyright (C) 2017-2021 Free Software Foundation, Inc.
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
  char ch = 1;
  /* These tests should fail.  */
  TEST_COMPARE (ch, -1);         /* Line 28.  */
  TEST_COMPARE (2LL, -2LL);      /* Line 29.  */
  TEST_COMPARE (3LL, (short) -3); /* Line 30.  */
}

struct bitfield
{
  int i2 : 2;
  int i3 : 3;
  unsigned int u2 : 2;
  unsigned int u3 : 3;
  int i31 : 31;
  unsigned int u31 : 31 ;
  long long int i63 : 63;
  unsigned long long int u63 : 63;
};

/* Functions which return signed sizes are common, so test that these
   results can readily checked using TEST_COMPARE.  */

static int
return_ssize_t (void)
{
  return 4;
}

static int
return_int (void)
{
  return 4;
}


static int
do_test (void)
{
  /* This should succeed.  */
  TEST_COMPARE (1, 1);
  TEST_COMPARE (2LL, 2U);
  {
    char i8 = 3;
    unsigned short u16 = 3;
    TEST_COMPARE (i8, u16);
  }
  TEST_COMPARE (return_ssize_t (), sizeof (char[4]));
  TEST_COMPARE (return_int (), sizeof (char[4]));

  struct bitfield bitfield = { 0 };
  TEST_COMPARE (bitfield.i2, bitfield.i3);
  TEST_COMPARE (bitfield.u2, bitfield.u3);
  TEST_COMPARE (bitfield.u2, bitfield.i3);
  TEST_COMPARE (bitfield.u3, bitfield.i3);
  TEST_COMPARE (bitfield.i2, bitfield.u3);
  TEST_COMPARE (bitfield.i3, bitfield.u2);
  TEST_COMPARE (bitfield.i63, bitfield.i63);
  TEST_COMPARE (bitfield.u63, bitfield.u63);
  TEST_COMPARE (bitfield.i31, bitfield.i63);
  TEST_COMPARE (bitfield.i63, bitfield.i31);

  struct support_capture_subprocess proc = support_capture_subprocess
    (&subprocess, NULL);

  /* Discard the reported error.  */
  support_record_failure_reset ();

  puts ("info: *** subprocess output starts ***");
  fputs (proc.out.buffer, stdout);
  puts ("info: *** subprocess output ends ***");

  TEST_VERIFY
    (strcmp (proc.out.buffer,
             "tst-test_compare.c:28: numeric comparison failure\n"
             "   left: 1 (0x1); from: ch\n"
             "  right: -1 (0xffffffff); from: -1\n"
             "tst-test_compare.c:29: numeric comparison failure\n"
             "   left: 2 (0x2); from: 2LL\n"
             "  right: -2 (0xfffffffffffffffe); from: -2LL\n"
             "tst-test_compare.c:30: numeric comparison failure"
             " (widths 64 and 32)\n"
             "   left: 3 (0x3); from: 3LL\n"
             "  right: -3 (0xfffffffd); from: (short) -3\n") == 0);

  /* Check that there is no output on standard error.  */
  support_capture_subprocess_check (&proc, "TEST_COMPARE", 0, sc_allow_stdout);

  return 0;
}

#include <support/test-driver.c>
