/* Test re_search with multi-byte characters in EUC-JP.
   Copyright (C) 2012-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Stanislav Brabec <sbrabec@suse.cz>, 2012.

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

#define _GNU_SOURCE 1
#include <locale.h>
#include <regex.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int
do_test (void)
{
  struct re_pattern_buffer r;
  struct re_registers s;
  int e, rc = 0;
  if (setlocale (LC_CTYPE, "ja_JP.EUC-JP") == NULL)
    {
      puts ("setlocale failed");
      return 1;
    }
  memset (&r, 0, sizeof (r));
  memset (&s, 0, sizeof (s));

  /* The bug cannot be reproduced without initialized fastmap (it is SBC_MAX
     value from regex_internal.h).  */
  r.fastmap = malloc (UCHAR_MAX + 1);

                     /* 圭 */
  re_compile_pattern ("\xb7\xbd", 2, &r);

                /* aaaaa件a新処, \xb7\xbd constitutes a false match */
  e = re_search (&r, "\x61\x61\x61\x61\x61\xb7\xef\x61\xbf\xb7\xbd\xe8",
                 12, 0, 12, &s);
  if (e != -1)
    {
      printf ("bug-regex33.1: false match or error: re_search() returned %d, should return -1\n", e);
      rc = 1;
    }

                /* aaaa件a新処, \xb7\xbd constitutes a false match,
                 * this is a reproducer of BZ #13637 */
  e = re_search (&r, "\x61\x61\x61\x61\xb7\xef\x61\xbf\xb7\xbd\xe8",
                 11, 0, 11, &s);
  if (e != -1)
    {
      printf ("bug-regex33.2: false match or error: re_search() returned %d, should return -1\n", e);
      rc = 1;
    }

                /* aaa件a新処, \xb7\xbd constitutes a false match,
                 * this is a reproducer of BZ #13637 */
  e = re_search (&r, "\x61\x61\x61\xb7\xef\x61\xbf\xb7\xbd\xe8",
                 10, 0, 10, &s);
  if (e != -1)
    {
      printf ("bug-regex33.3: false match or error: re_search() returned %d, should return -1\n", e);
      rc = 1;
    }

                /* aa件a新処, \xb7\xbd constitutes a false match */
  e = re_search (&r, "\x61\x61\xb7\xef\x61\xbf\xb7\xbd\xe8",
                 9, 0, 9, &s);
  if (e != -1)
    {
      printf ("bug-regex33.4: false match or error: re_search() returned %d, should return -1\n", e);
      rc = 1;
    }

                /* a件a新処, \xb7\xbd constitutes a false match */
  e = re_search (&r, "\x61\xb7\xef\x61\xbf\xb7\xbd\xe8",
                 8, 0, 8, &s);
  if (e != -1)
    {
      printf ("bug-regex33.5: false match or error: re_search() returned %d, should return -1\n", e);
      rc = 1;
    }

                /* 新処圭新処, \xb7\xbd here really matches 圭, but second occurrence is a false match,
                 * this is a reproducer of bug-regex25 and BZ #13637 */
  e = re_search (&r, "\xbf\xb7\xbd\xe8\xb7\xbd\xbf\xb7\xbd\xe8",
                 10, 0, 10, &s);
  if (e != 4)
    {
      printf ("bug-regex33.6: no match or false match: re_search() returned %d, should return 4\n", e);
      rc = 1;
    }

                /* 新処圭新, \xb7\xbd here really matches 圭,
                 * this is a reproducer of bug-regex25 */
  e = re_search (&r, "\xbf\xb7\xbd\xe8\xb7\xbd\xbf\xb7",
                 9, 0, 9, &s);
  if (e != 4)
    {
      printf ("bug-regex33.7: no match or false match: re_search() returned %d, should return 4\n", e);
      rc = 1;
    }

  return rc;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
