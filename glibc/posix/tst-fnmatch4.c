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

#include <stdio.h>
#include <locale.h>
#include <fnmatch.h>
#include <support/check.h>

static void
do_test_locale (const char *locale)
{
  TEST_VERIFY_EXIT (setlocale (LC_ALL, locale) != NULL);

  TEST_VERIFY (fnmatch ("[[.ch.]]", "ch", 0) == 0);
}

static int
do_test (void)
{
  do_test_locale ("cs_CZ.ISO-8859-2");
  do_test_locale ("cs_CZ.UTF-8");

  return 0;
}

#include <support/test-driver.c>
