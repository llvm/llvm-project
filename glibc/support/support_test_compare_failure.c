/* Reporting a numeric comparison failure.
   Copyright (C) 2017-2021 Free Software Foundation, Inc.
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

#include <errno.h>
#include <stdio.h>
#include <support/check.h>

static void
report (const char *which, const char *expr, long long value, int positive,
        int size)
{
  printf ("  %s: ", which);
  if (positive)
    printf ("%llu", (unsigned long long) value);
  else
    printf ("%lld", value);
  unsigned long long mask
    = (~0ULL) >> (8 * (sizeof (unsigned long long) - size));
  printf (" (0x%llx); from: %s\n", (unsigned long long) value & mask, expr);
}

void
support_test_compare_failure (const char *file, int line,
                              const char *left_expr,
                              long long left_value,
                              int left_positive,
                              int left_size,
                              const char *right_expr,
                              long long right_value,
                              int right_positive,
                              int right_size)
{
  int saved_errno = errno;
  support_record_failure ();
  if (left_size != right_size)
    printf ("%s:%d: numeric comparison failure (widths %d and %d)\n",
            file, line, left_size * 8, right_size * 8);
  else
    printf ("%s:%d: numeric comparison failure\n", file, line);
  report (" left", left_expr, left_value, left_positive, left_size);
  report ("right", right_expr, right_value, right_positive, right_size);
  errno = saved_errno;
}
