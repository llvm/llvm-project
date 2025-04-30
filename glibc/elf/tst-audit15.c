/* Main program with DT_AUDIT and DT_DEPAUDIT.  Two audit modules.
   Copyright (C) 2020-2021 Free Software Foundation, Inc.
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
#include <string.h>
#include <support/check.h>
#include <support/xstdio.h>

static int
do_test (void)
{
  /* Verify what the audit modules have written.  This test assumes
     that standard output has been redirected to a regular file.  */
  FILE *fp = xfopen ("/dev/stdout", "r");

  char *buffer = NULL;
  size_t buffer_length = 0;
  size_t line_length = xgetline (&buffer, &buffer_length, fp);
  const char *message = "info: tst-auditlogmod-1.so loaded\n";
  TEST_COMPARE_BLOB (message, strlen (message), buffer, line_length);

  line_length = xgetline (&buffer, &buffer_length, fp);
  message = "info: tst-auditlogmod-2.so loaded\n";
  TEST_COMPARE_BLOB (message, strlen (message), buffer, line_length);

  /* No more audit module output.  */
  line_length = xgetline (&buffer, &buffer_length, fp);
  TEST_COMPARE_BLOB ("", 0, buffer, line_length);

  free (buffer);
  xfclose (fp);
  return 0;
}

#include <support/test-driver.c>
