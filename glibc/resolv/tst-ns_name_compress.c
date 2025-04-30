/* Test ns_name_compress corner cases.
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

#include <resolv.h>
#include <stdlib.h>
#include <string.h>
#include <support/check.h>
#include <support/support.h>

/* Check that we can process names which fit into the destination
   buffer exactly.  See bug 21359.  */
static void
test_exact_fit (const char *name, size_t length)
{
  unsigned char *buf = xmalloc (length + 1);
  memset (buf, '$', length + 1);
  enum { ptr_count = 5 };
  const unsigned char *dnptrs[ptr_count] = { buf, };
  int ret = ns_name_compress (name, buf, length,
                          dnptrs, dnptrs + ptr_count);
  if (ret < 0)
    {
      support_record_failure ();
      printf ("error: ns_name_compress for %s/%zu failed\n", name, length);
      return;
    }
  if ((size_t) ret != length)
    {
      support_record_failure ();
      printf ("error: ns_name_compress for %s/%zu result mismatch: %d\n",
              name, length, ret);
    }
  if (buf[length] != '$')
    {
      support_record_failure ();
      printf ("error: ns_name_compress for %s/%zu padding write\n",
              name, length);
    }
  free (buf);
}

static int
do_test (void)
{
  test_exact_fit ("abc", 5);
  test_exact_fit ("abc.", 5);
  {
    char long_name[]
      = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa."
      "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa."
      "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa."
      "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.";
    TEST_VERIFY (strlen (long_name) == NS_MAXCDNAME - 1);
    test_exact_fit (long_name, NS_MAXCDNAME);
    long_name[sizeof (long_name) - 1] = '\0';
    test_exact_fit (long_name, NS_MAXCDNAME);
  }
  return 0;
}

#include <support/test-driver.c>
