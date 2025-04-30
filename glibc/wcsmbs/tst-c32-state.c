/* Test mbrtowc and mbrtoc32 do not share state (bug 23793).
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

#include <locale.h>
#include <uchar.h>
#include <wchar.h>
#include <support/check.h>

static int
do_test (void)
{
  TEST_VERIFY_EXIT (setlocale (LC_ALL, "de_DE.UTF-8") != NULL);
  const char buf[] = "\u00ff";
  wchar_t wc = 0;
  char32_t c32 = 0;
  size_t ret = mbrtowc (&wc, buf, 1, NULL);
  TEST_COMPARE (ret, (size_t) -2);
  ret = mbrtoc32 (&c32, buf, 1, NULL);
  TEST_COMPARE (ret, (size_t) -2);
  ret = mbrtowc (&wc, buf + 1, 1, NULL);
  TEST_COMPARE (ret, 1);
  TEST_COMPARE (wc, 0xff);
  ret = mbrtoc32 (&c32, buf + 1, 1, NULL);
  TEST_COMPARE (ret, 1);
  TEST_COMPARE (c32, 0xff);
  return 0;
}

#include <support/test-driver.c>
