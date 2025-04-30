/* Test for the long double variants of strfroml and strtold.
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

#include <stdlib.h>

#include <support/check.h>

static int
do_test (void)
{
  int written;
  char buffer[64];
  char *bufptr = buffer;
  const char *expected = "-1.0000000000";
  long double read;

  /* Write to the buffer.  */
  written = strfroml (bufptr, 64, "%.10f", (long double) -1);
  if (written < 0)
    support_record_failure ();

  /* Compare against the expected output.  */
  TEST_COMPARE_STRING (expected, buffer);

  /* Read from the buffer.  */
  read = strtold (expected, NULL);

  if (read != -1.0L)
    support_record_failure ();

  return 0;
}

#include <support/test-driver.c>
