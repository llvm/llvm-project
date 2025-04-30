/* Test iconv behavior on UCS4 conversions with //IGNORE.
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
   <http://www.gnu.org/licenses/>.  */

/* Derived from BZ #26923 */
#include <errno.h>
#include <iconv.h>
#include <stdio.h>
#include <support/check.h>

static int
do_test (void)
{
  iconv_t cd = iconv_open ("UTF-8//IGNORE", "ISO-10646/UCS4/");
  TEST_VERIFY_EXIT (cd != (iconv_t) -1);

  /*
   * Convert sequence beginning with an irreversible character into buffer that
   * is too small.
   */
  char input[12] = "\xe1\x80\xa1" "AAAAAAAAA";
  char *inptr = input;
  size_t insize = sizeof (input);
  char output[6];
  char *outptr = output;
  size_t outsize = sizeof (output);

  TEST_VERIFY (iconv (cd, &inptr, &insize, &outptr, &outsize) == -1);
  TEST_VERIFY (errno == E2BIG);

  TEST_VERIFY_EXIT (iconv_close (cd) != -1);

  return 0;
}

#include <support/test-driver.c>
