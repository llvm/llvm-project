/* Test for the long double variants of strfmon* functions.
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

#include <locale/locale.h>
#include <monetary.h>

#include <support/check.h>

static int
do_test (void)
{
  size_t written;
  char buffer[64];
  char *bufptr = buffer;
  locale_t loc;

  /* Using the C locale is enough for the purpose of this test case,
     i.e.: to test that strfmon correctly reads long double values with
     binary128 format.  Grouping and currency are irrelevant, here.  */
  setlocale (LC_MONETARY, "C");
  loc = newlocale (LC_MONETARY_MASK, "C", (locale_t) 0);

  /* Write to the buffer.  */
  written = strfmon (bufptr, 32, "%.10i, %.10Li\n",
                     (double) -2, (long double) -1);
  if (written < 0)
    support_record_failure ();
  else
    bufptr += written;
  written = strfmon_l (bufptr, 32, loc, "%.10i, %.10Li\n",
                       (double) -2, (long double) -1);
  if (written < 0)
    support_record_failure ();

  /* Compare against the expected output.  */
  const char *expected =
    "-2.0000000000, -1.0000000000\n"
    "-2.0000000000, -1.0000000000\n";
  TEST_COMPARE_STRING (expected, buffer);

  return 0;
}

#include <support/test-driver.c>
