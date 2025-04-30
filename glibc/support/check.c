/* Support code for reporting test results.
   Copyright (C) 2016-2021 Free Software Foundation, Inc.
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

#include <support/check.h>

#include <errno.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <support/test-driver.h>

static void
print_failure (const char *file, int line, const char *format, va_list ap)
{
  int saved_errno = errno;
  printf ("error: %s:%d: ", file, line);
  vprintf (format, ap);
  puts ("");
  errno = saved_errno;
}

int
support_print_failure_impl (const char *file, int line,
                            const char *format, ...)
{
  support_record_failure ();
  va_list ap;
  va_start (ap, format);
  print_failure (file, line, format, ap);
  va_end (ap);
  return 1;
}

void
support_exit_failure_impl (int status, const char *file, int line,
                           const char *format, ...)
{
  if (status != EXIT_SUCCESS && status != EXIT_UNSUPPORTED)
    support_record_failure ();
  va_list ap;
  va_start (ap, format);
  print_failure (file, line, format, ap);
  va_end (ap);
  exit (status);
}
