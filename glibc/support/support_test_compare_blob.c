/* Check two binary blobs for equality.
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
report_length (const char *what, unsigned long int length, const char *expr)
{
  printf ("  %s %lu bytes (from %s)\n", what, length, expr);
}

static void
report_blob (const char *what, const unsigned char *blob,
             unsigned long int length, const char *expr)
{
  if (blob == NULL && length > 0)
    printf ("  %s (evaluated from %s): NULL\n", what, expr);
  else if (length > 0)
    {
      printf ("  %s (evaluated from %s):\n", what, expr);
      char *quoted = support_quote_blob (blob, length);
      printf ("      \"%s\"\n", quoted);
      free (quoted);

      fputs ("     ", stdout);
      for (unsigned long i = 0; i < length; ++i)
        printf (" %02X", blob[i]);
      putc ('\n', stdout);
    }
}

void
support_test_compare_blob (const void *left, unsigned long int left_length,
                           const void *right, unsigned long int right_length,
                           const char *file, int line,
                           const char *left_expr, const char *left_len_expr,
                           const char *right_expr, const char *right_len_expr)
{
  /* No differences are possible if both lengths are null.  */
  if (left_length == 0 && right_length == 0)
    return;

  if (left_length != right_length || left == NULL || right == NULL
      || memcmp (left, right, left_length) != 0)
    {
      support_record_failure ();
      printf ("%s:%d: error: blob comparison failed\n", file, line);
      if (left_length == right_length)
        printf ("  blob length: %lu bytes\n", left_length);
      else
        {
          report_length ("left length: ", left_length, left_len_expr);
          report_length ("right length:", right_length, right_len_expr);
        }
      report_blob ("left", left, left_length, left_expr);
      report_blob ("right", right, right_length, right_expr);
    }
}
