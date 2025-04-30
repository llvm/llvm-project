/* Test for fnmatch handling of collating elements
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

#include <fnmatch.h>
#include <locale.h>
#include <stdio.h>
#include <string.h>
#include <support/check.h>

#define LENGTH 20000000

static char pattern[LENGTH + 7];

static int
do_test (void)
{
  TEST_VERIFY_EXIT (setlocale (LC_ALL, "en_US.UTF-8") != NULL);

  pattern[0] = '[';
  pattern[1] = '[';
  pattern[2] = '.';
  memset (pattern + 3, 'a', LENGTH);
  pattern[LENGTH + 3] = '.';
  pattern[LENGTH + 4] = ']';
  pattern[LENGTH + 5] = ']';
  TEST_VERIFY (fnmatch (pattern, "a", 0) != 0);

  return 0;
}

#include <support/test-driver.c>
