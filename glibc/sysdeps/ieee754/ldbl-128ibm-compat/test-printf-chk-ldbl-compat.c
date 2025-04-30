/* Test for the long double variants of *printf_chk functions.
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

#define _FORTIFY_SOURCE 2

#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <support/capture_subprocess.h>
#include <support/check.h>

static void
do_test_call_varg (FILE *stream, const char *format, ...)
{
  char *buffer = NULL;
  char string[128];
  int res;
  va_list args;

  printf ("%20s", "__vasprintf_chk: ");
  va_start (args, format);
  res = __vasprintf_chk (&buffer, 1, format, args);
  va_end (args);
  if (res == -1)
    printf ("Error using vasprintf\n");
  if (buffer == NULL)
    printf ("Error using vasprintf\n");
  else
    {
      printf ("%s", buffer);
      free (buffer);
    }
  printf ("\n");

  printf ("%20s", "__vdprintf_chk: ");
  va_start (args, format);
  __vdprintf_chk (fileno (stream), 1, format, args);
  va_end (args);
  printf ("\n");

  printf ("%20s", "__vfprintf_chk: ");
  va_start (args, format);
  __vfprintf_chk (stream, 1, format, args);
  va_end (args);
  printf ("\n");

  printf ("%20s", "__vprintf_chk: ");
  va_start (args, format);
  __vprintf_chk (1, format, args);
  va_end (args);
  printf ("\n");

  printf ("%20s", "__vsnprintf_chk: ");
  va_start (args, format);
  __vsnprintf_chk (string, 79, 1, 127, format, args);
  va_end (args);
  printf ("%s", string);
  printf ("\n");

  printf ("%20s", "__vsprintf_chk: ");
  va_start (args, format);
  __vsprintf_chk (string, 1, 127, format, args);
  va_end (args);
  printf ("%s", string);
  printf ("\n");
}

static void
do_test_call_rarg (FILE *stream, const char *format, long double ld,
		   double d)
{
  char *buffer = NULL;
  char string[128];
  int res;

  printf ("%20s", "__asprintf_chk: ");
  res = __asprintf_chk (&buffer, 1, format, ld, d);
  if (res == -1)
    printf ("Error using vasprintf\n");
  if (buffer == NULL)
    printf ("Error using asprintf\n");
  else
    {
      printf ("%s", buffer);
      free (buffer);
    }
  printf ("\n");

  printf ("%20s", "__dprintf_chk: ");
  __dprintf_chk (fileno (stream), 1, format, ld, d);
  printf ("\n");

  printf ("%20s", "__fprintf_chk: ");
  __fprintf_chk (stdout, 1, format, ld, d);
  printf ("\n");

  printf ("%20s", "__printf_chk: ");
  __printf_chk (1, format, ld, d);
  printf ("\n");

  printf ("%20s", "__snprintf_chk: ");
  __snprintf_chk (string, 79, 1, 127, format, ld, d);
  printf ("%s", string);
  printf ("\n");

  printf ("%20s", "__sprintf_chk: ");
  __sprintf_chk (string, 1, 127, format, ld, d);
  printf ("%s", string);
  printf ("\n");
}

static void
do_test_call (void)
{
  long double ld = -1;
  double d = -1;

  /* Print in decimal notation.  */
  do_test_call_rarg (stdout, "%.10Lf, %.10f", ld, d);
  do_test_call_varg (stdout, "%.10Lf, %.10f", ld, d);

  /* Print in hexadecimal notation.  */
  do_test_call_rarg (stdout, "%.10La, %.10a", ld, d);
  do_test_call_varg (stdout, "%.10La, %.10a", ld, d);

  /* Test positional parameters.  */
  do_test_call_varg (stdout, "%3$Lf, %2$Lf, %1$f",
		     (double) 1, (long double) 2, (long double) 3);
}

static int
do_test (void)
{
  struct support_capture_subprocess result;
  result = support_capture_subprocess ((void *) &do_test_call, NULL);

  /* Compare against the expected output.  */
  const char *expected =
    "    __asprintf_chk: -1.0000000000, -1.0000000000\n"
    "     __dprintf_chk: -1.0000000000, -1.0000000000\n"
    "     __fprintf_chk: -1.0000000000, -1.0000000000\n"
    "      __printf_chk: -1.0000000000, -1.0000000000\n"
    "    __snprintf_chk: -1.0000000000, -1.0000000000\n"
    "     __sprintf_chk: -1.0000000000, -1.0000000000\n"
    "   __vasprintf_chk: -1.0000000000, -1.0000000000\n"
    "    __vdprintf_chk: -1.0000000000, -1.0000000000\n"
    "    __vfprintf_chk: -1.0000000000, -1.0000000000\n"
    "     __vprintf_chk: -1.0000000000, -1.0000000000\n"
    "   __vsnprintf_chk: -1.0000000000, -1.0000000000\n"
    "    __vsprintf_chk: -1.0000000000, -1.0000000000\n"
    "    __asprintf_chk: -0x1.0000000000p+0, -0x1.0000000000p+0\n"
    "     __dprintf_chk: -0x1.0000000000p+0, -0x1.0000000000p+0\n"
    "     __fprintf_chk: -0x1.0000000000p+0, -0x1.0000000000p+0\n"
    "      __printf_chk: -0x1.0000000000p+0, -0x1.0000000000p+0\n"
    "    __snprintf_chk: -0x1.0000000000p+0, -0x1.0000000000p+0\n"
    "     __sprintf_chk: -0x1.0000000000p+0, -0x1.0000000000p+0\n"
    "   __vasprintf_chk: -0x1.0000000000p+0, -0x1.0000000000p+0\n"
    "    __vdprintf_chk: -0x1.0000000000p+0, -0x1.0000000000p+0\n"
    "    __vfprintf_chk: -0x1.0000000000p+0, -0x1.0000000000p+0\n"
    "     __vprintf_chk: -0x1.0000000000p+0, -0x1.0000000000p+0\n"
    "   __vsnprintf_chk: -0x1.0000000000p+0, -0x1.0000000000p+0\n"
    "    __vsprintf_chk: -0x1.0000000000p+0, -0x1.0000000000p+0\n"
    "   __vasprintf_chk: 3.000000, 2.000000, 1.000000\n"
    "    __vdprintf_chk: 3.000000, 2.000000, 1.000000\n"
    "    __vfprintf_chk: 3.000000, 2.000000, 1.000000\n"
    "     __vprintf_chk: 3.000000, 2.000000, 1.000000\n"
    "   __vsnprintf_chk: 3.000000, 2.000000, 1.000000\n"
    "    __vsprintf_chk: 3.000000, 2.000000, 1.000000\n";
  TEST_COMPARE_STRING (expected, result.out.buffer);

  return 0;
}

#include <support/test-driver.c>
