/* bug 24973: Test EUC-KR module
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

#include <errno.h>
#include <iconv.h>
#include <stdio.h>
#include <support/check.h>

static int
do_test (void)
{
  iconv_t cd = iconv_open ("UTF-8//IGNORE", "EUC-KR");
  TEST_VERIFY_EXIT (cd != (iconv_t) -1);

  /* 0xfe (->0x7e : row 94) and 0xc9 (->0x49 : row 41) are user-defined
     areas, which are not allowed and should be skipped over due to
     //IGNORE.  The trailing 0xfe also is an incomplete sequence, which
     should be checked first.  */
  char input[4] = { '\xc9', '\xa1', '\0', '\xfe' };
  char *inptr = input;
  size_t insize = sizeof (input);
  char output[4];
  char *outptr = output;
  size_t outsize = sizeof (output);

  /* This used to crash due to buffer overrun.  */
  TEST_VERIFY (iconv (cd, &inptr, &insize, &outptr, &outsize) == (size_t) -1);
  TEST_VERIFY (errno == EINVAL);
  /* The conversion should produce one character, the converted null
     character.  */
  TEST_VERIFY (sizeof (output) - outsize == 1);

  TEST_VERIFY_EXIT (iconv_close (cd) != -1);

  return 0;
}

#include <support/test-driver.c>
