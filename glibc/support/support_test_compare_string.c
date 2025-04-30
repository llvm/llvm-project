/* Check two strings for equality.
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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <support/check.h>
#include <support/support.h>
#include <support/xmemstream.h>

static void
report_length (const char *what, const char *str, size_t length)
{
  if (str == NULL)
    printf ("  %s string: NULL\n", what);
  else
    printf ("  %s string: %zu bytes\n", what, length);
}

static void
report_string (const char *what, const unsigned char *blob,
               size_t length, const char *expr)
{
  if (length > 0)
    {
      printf ("  %s (evaluated from %s):\n", what, expr);
      char *quoted = support_quote_blob (blob, length);
      printf ("      \"%s\"\n", quoted);
      free (quoted);

      fputs ("     ", stdout);
      for (size_t i = 0; i < length; ++i)
        printf (" %02X", blob[i]);
      putc ('\n', stdout);
    }
}

static size_t
string_length_or_zero (const char *str)
{
  if (str == NULL)
    return 0;
  else
    return strlen (str);
}

void
support_test_compare_string (const char *left, const char *right,
                             const char *file, int line,
                             const char *left_expr, const char *right_expr)
{
  /* Two null pointers are accepted.  */
  if (left == NULL && right == NULL)
    return;

  size_t left_length = string_length_or_zero (left);
  size_t right_length = string_length_or_zero (right);

  if (left_length != right_length || left == NULL || right == NULL
      || memcmp (left, right, left_length) != 0)
    {
      support_record_failure ();
      printf ("%s:%d: error: string comparison failed\n", file, line);
      if (left_length == right_length && right != NULL && left != NULL)
        printf ("  string length: %zu bytes\n", left_length);
      else
        {
          report_length ("left", left, left_length);
          report_length ("right", right, right_length);
        }
      report_string ("left", (const unsigned char *) left,
                     left_length, left_expr);
      report_string ("right", (const unsigned char *) right,
                     right_length, right_expr);
    }
}
