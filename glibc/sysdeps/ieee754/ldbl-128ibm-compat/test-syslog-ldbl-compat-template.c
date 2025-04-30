/* Test for the long double variants of *syslog* functions.
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

#include <stdarg.h>
#include <stddef.h>
#include <syslog.h>

#include <support/capture_subprocess.h>
#include <support/check.h>

static void
do_test_one_call (void *last, ...)
{
  long double ld = -1;
  va_list ap;

  /* Make syslog functions write to stderr with LOG_PERROR, so that it
     can be captured by support_capture_subprocess and verified.  */
  openlog ("test-syslog", LOG_PERROR, LOG_USER);

  /* Call syslog functions that take a format string.  */
  SYSLOG_FUNCTION SYSLOG_FUNCTION_PARAMS;
  va_start (ap, last);
  VSYSLOG_FUNCTION VSYSLOG_FUNCTION_PARAMS;
  va_end (ap);
}

static void
do_test_call (void)
{
  long double ld = -1;
  do_test_one_call (NULL, ld);
}

static int
do_test (void)
{
  struct support_capture_subprocess result;
  result = support_capture_subprocess ((void *) &do_test_call, NULL);

  do_test_call ();

  /* Compare against the expected output.  */
  const char *expected =
    "test-syslog: -1.000000\n"
    "test-syslog: -1.000000\n";
  TEST_COMPARE_STRING (expected, result.err.buffer);

  return 0;
}

#include <support/test-driver.c>
