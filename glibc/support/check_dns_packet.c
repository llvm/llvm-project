/* Check that a DNS packet buffer has the expected contents.
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

#include <support/check_nss.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <support/check.h>
#include <support/format_nss.h>
#include <support/run_diff.h>

void
check_dns_packet (const char *query_description,
                  const unsigned char *buffer, size_t length,
                  const char *expected)
{
  char *formatted = support_format_dns_packet (buffer, length);
  if (strcmp (formatted, expected) != 0)
    {
      support_record_failure ();
      printf ("error: packet comparison failure\n");
      if (query_description != NULL)
        printf ("query: %s\n", query_description);
      support_run_diff ("expected", expected, "actual", formatted);
    }
  free (formatted);
}
