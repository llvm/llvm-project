/* Test c16rtomb handling of surrogate pairs (DR#488, bug 23794).
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

#include <errno.h>
#include <locale.h>
#include <stdio.h>
#include <string.h>
#include <uchar.h>
#include <wchar.h>
#include <array_length.h>
#include <support/check.h>

static int
do_test (void)
{
  TEST_VERIFY_EXIT (setlocale (LC_ALL, "de_DE.UTF-8") != NULL);
  /* Test conversions of surrogate pairs.  */
  for (char32_t c = 0x10000; c <= 0x10ffff; c += 0x123)
    {
      char32_t c_pos = c - 0x10000;
      char16_t c_hi = (c_pos >> 10) + 0xd800;
      char16_t c_lo = (c_pos & 0x3ff) + 0xdc00;
      printf ("testing U+0x%08x (0x%x 0x%x)\n",
	      (unsigned int) c, (unsigned int) c_hi, (unsigned int) c_lo);
      char buf[16] = { 0 };
      size_t ret_hi = c16rtomb (buf, c_hi, NULL);
      TEST_COMPARE (ret_hi, 0);
      size_t ret_lo = c16rtomb (buf, c_lo, NULL);
      TEST_COMPARE (ret_lo, 4);
      wchar_t wc = 0;
      size_t ret_wc = mbrtowc (&wc, buf, 4, NULL);
      TEST_COMPARE (ret_wc, 4);
      TEST_COMPARE (wc, (wchar_t) c);
    }
  /* Test errors for invalid conversions.  */
  static const char16_t err_cases[][2] =
    {
      /* High surrogate followed by non-surrogate.  */
      { 0xd800, 0x1 },
      /* High surrogate followed by another high surrogate.  */
      { 0xd800, 0xd800 },
      /* Low surrogate not following high surrogate.  */
      { 0xdc00, 0 }
    };
  for (size_t i = 0; i < array_length (err_cases); i++)
    {
      char16_t c_hi = err_cases[i][0];
      char16_t c_lo = err_cases[i][1];
      printf ("testing error case: 0x%x 0x%x\n", (unsigned int) c_hi,
	      (unsigned int) c_lo);
      c16rtomb (NULL, 0, NULL);
      char buf[16] = { 0 };
      errno = 0;
      size_t ret_hi = c16rtomb (buf, c_hi, NULL);
      if (c_lo == 0)
	{
	  /* Unmatched low surrogate in first place.  */
	  TEST_COMPARE (ret_hi, (size_t) -1);
	  TEST_COMPARE (errno, EILSEQ);
	}
      else
	{
	  /* High surrogate; error in second place.  */
	  TEST_COMPARE (ret_hi, 0);
	  errno = 0;
	  size_t ret_lo = c16rtomb (buf, c_lo, NULL);
	  TEST_COMPARE (ret_lo, (size_t) -1);
	  TEST_COMPARE (errno, EILSEQ);
	}
    }
  return 0;
}

#include <support/test-driver.c>
