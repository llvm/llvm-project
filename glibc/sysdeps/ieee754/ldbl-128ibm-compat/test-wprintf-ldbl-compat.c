/* Test for the long double variants of *w*printf functions.
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
#include <stdint.h>
#include <stdio.h>
#include <wchar.h>

#include <support/capture_subprocess.h>
#include <support/check.h>

static void
do_test_call_varg (FILE *stream, const wchar_t *format, ...)
{
  wchar_t string[128];
  va_list args;

  wprintf (L"%15Ls", L"vfwprintf: ");
  va_start (args, format);
  vfwprintf (stream, format, args);
  va_end (args);
  wprintf (L"\n");

  wprintf (L"%15Ls", L"vswprintf: ");
  va_start (args, format);
  vswprintf (string, 127, format, args);
  va_end (args);
  wprintf (L"%Ls", string);
  wprintf (L"\n");

  wprintf (L"%15Ls", L"vwprintf: ");
  va_start (args, format);
  vwprintf (format, args);
  va_end (args);
  wprintf (L"\n");
}

static void
do_test_call_rarg (FILE *stream, const wchar_t *format, long double ld,
		   double d)
{
  wchar_t string[128];

  wprintf (L"%15Ls", L"fwprintf: ");
  fwprintf (stream, format, ld, d);
  wprintf (L"\n");

  wprintf (L"%15Ls", L"swprintf: ");
  swprintf (string, 127, format, ld, d);
  wprintf (L"%Ls", string);
  wprintf (L"\n");

  wprintf (L"%15Ls", L"wprintf: ");
  wprintf (format, ld, d);
  wprintf (L"\n");
}

static void
do_test_call (void)
{
  long double ld = -1;
  double d = -1;

  /* Print in decimal notation.  */
  do_test_call_rarg (stdout, L"%.10Lf, %.10f", ld, d);
  do_test_call_varg (stdout, L"%.10Lf, %.10f", ld, d);

  /* Print in hexadecimal notation.  */
  do_test_call_rarg (stdout, L"%.10La, %.10a", ld, d);
  do_test_call_varg (stdout, L"%.10La, %.10a", ld, d);

  /* Test positional parameters.  */
  do_test_call_varg (stdout, L"%3$Lf, %2$Lf, %1$f",
		     (double) 1, (long double) 2, (long double) 3);
}

static int
do_test (void)
{
  struct support_capture_subprocess result;
  result = support_capture_subprocess ((void *) &do_test_call, NULL);

  /* Compare against the expected output.  */
  const char *expected =
    "     fwprintf: -1.0000000000, -1.0000000000\n"
    "     swprintf: -1.0000000000, -1.0000000000\n"
    "      wprintf: -1.0000000000, -1.0000000000\n"
    "    vfwprintf: -1.0000000000, -1.0000000000\n"
    "    vswprintf: -1.0000000000, -1.0000000000\n"
    "     vwprintf: -1.0000000000, -1.0000000000\n"
    "     fwprintf: -0x1.0000000000p+0, -0x1.0000000000p+0\n"
    "     swprintf: -0x1.0000000000p+0, -0x1.0000000000p+0\n"
    "      wprintf: -0x1.0000000000p+0, -0x1.0000000000p+0\n"
    "    vfwprintf: -0x1.0000000000p+0, -0x1.0000000000p+0\n"
    "    vswprintf: -0x1.0000000000p+0, -0x1.0000000000p+0\n"
    "     vwprintf: -0x1.0000000000p+0, -0x1.0000000000p+0\n"
    "    vfwprintf: 3.000000, 2.000000, 1.000000\n"
    "    vswprintf: 3.000000, 2.000000, 1.000000\n"
    "     vwprintf: 3.000000, 2.000000, 1.000000\n";
  TEST_COMPARE_STRING (expected, result.out.buffer);

  return 0;
}

#include <support/test-driver.c>
