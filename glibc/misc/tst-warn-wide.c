/* Test wide output conversion for warn.
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

#include <err.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <support/check.h>
#include <support/xmemstream.h>
#include <wchar.h>

/* Used to trigger the large-string path in __fxprintf.  */
#define PADDING \
  "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX" \
  "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX" \
  "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"

#define LPADDING \
  L"XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"  \
  "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX" \
  "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"


static void
one_test (const char *message, int error_code, const wchar_t *expected)
{
  wchar_t *buffer = NULL;
  size_t length = 0;
  FILE *fp = open_wmemstream (&buffer, &length);
  TEST_VERIFY_EXIT (fp != NULL);
  FILE *old_stderr = stderr;
  stderr = fp;
  errno = error_code;
  switch (error_code)
    {
    case E2BIG:
      warn ("%s with padding " PADDING, message);
      break;
    case EAGAIN:
      warn ("%s", message);
      break;
    case -1:
      warnx ("%s", message);
      break;
    case -2:
      warnx ("%s with padding " PADDING, message);
      break;
    }
  stderr = old_stderr;
  TEST_VERIFY_EXIT (!ferror (fp));
  TEST_COMPARE (fclose (fp), 0);
  if (wcscmp (buffer, expected) != 0)
    FAIL_EXIT1 ("unexpected output: %ls", buffer);
  free (buffer);
}

static int
do_test (void)
{
  one_test ("no errno", -1,
            L"tst-warn-wide: no errno\n");
  one_test ("no errno", -2,
            L"tst-warn-wide: no errno with padding " PADDING "\n");
  one_test ("with errno", EAGAIN,
            L"tst-warn-wide: with errno: Resource temporarily unavailable\n");
  one_test ("with errno", E2BIG,
            L"tst-warn-wide: with errno with padding " PADDING
            ": Argument list too long\n");
  return 0;
}

#include <support/test-driver.c>
