/* Test the sprintf (buf, "%s", buf) does not override buf.
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
#include <stdio.h>
#include <string.h>

#include <support/check.h>

enum
{
  FUNCTION_FIRST,
  FUNCTION_SPRINTF = FUNCTION_FIRST,
  FUNCTION_SNPRINF,
  FUNCTION_VSPRINTF,
  FUNCTION_VSNPRINTF,
  FUNCTION_LAST
};

static void
do_one_test (int function, char *buf, ...)
{
  va_list args;
  char *arg;

  /* Expected results for fortified and non-fortified sprintf.  */
#if defined _FORTIFY_SOURCE && _FORTIFY_SOURCE > 1
  const char *expected = "CD";
#else
  const char *expected = "ABCD";
#endif

  va_start (args, buf);
  arg = va_arg (args, char *);
  va_end (args);

  switch (function)
    {
      /* The regular sprintf and vsprintf functions do not override the
         destination buffer, even if source and destination overlap.  */
      case FUNCTION_SPRINTF:
        sprintf (buf, "%sCD", buf);
        TEST_COMPARE_STRING (buf, expected);
        break;
      case FUNCTION_VSPRINTF:
        va_start (args, buf);
        vsprintf (arg, "%sCD", args);
        TEST_COMPARE_STRING (arg, expected);
        va_end (args);
        break;
      /* On the other hand, snprint and vsnprint override overlapping
         source and destination buffers.  */
      case FUNCTION_SNPRINF:
        snprintf (buf, 3, "%sCD", buf);
        TEST_COMPARE_STRING (buf, "CD");
        break;
      case FUNCTION_VSNPRINTF:
        va_start (args, buf);
        vsnprintf (arg, 3, "%sCD", args);
        TEST_COMPARE_STRING (arg, "CD");
        va_end (args);
        break;
      default:
        support_record_failure ();
    }
}

static int
do_test (void)
{
  char buf[8];
  int i;

  /* For each function in the enum do:
       - reset the buffer to the initial state "AB";
       - call the function with buffer as source and destination;
       - check for the desired behavior.  */
  for (i = FUNCTION_FIRST; i < FUNCTION_LAST; i++)
    {
      strncpy (buf, "AB", 3);
      do_one_test (i, buf, buf);
    }

  return 0;
}

#include <support/test-driver.c>
