/* Test iconv buffer handling with the IGNORE error handler.
   Copyright (C) 2019-2021 Free Software Foundation, Inc.
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

/* Derived from BZ #18830 */
#include <errno.h>
#include <iconv.h>
#include <stdio.h>
#include <support/check.h>

static int
do_test (void)
{
  /* This conversion needs two steps, from ASCII to INTERNAL to ASCII.  */
  iconv_t cd = iconv_open ("ASCII//IGNORE", "ASCII");
  TEST_VERIFY_EXIT (cd != (iconv_t) -1);

  /* Convert some irreversible sequence, enough to trigger an overflow of
     the output buffer before the irreversible character in the second
     step, but after going past the irreversible character in the first
     step.  */
  char input[4 + 4] = { '0', '1', '2', '3', '4', '5', '\266', '7' };
  char *inptr = input;
  size_t insize = sizeof (input);
  char output[4];
  char *outptr = output;
  size_t outsize = sizeof (output);

  /* The conversion should fail.  */
  TEST_VERIFY (iconv (cd, &inptr, &insize, &outptr, &outsize) == (size_t) -1);
  TEST_VERIFY (errno == E2BIG);
  /* The conversion should not consume more than it was able to store in
     the output buffer.  */
  TEST_COMPARE (inptr - input, sizeof (output) - outsize);

  TEST_VERIFY_EXIT (iconv_close (cd) != -1);

  return 0;
}

#include <support/test-driver.c>
