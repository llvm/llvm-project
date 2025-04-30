/* Tests for <support/blob_repeat.h>
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
#include <string.h>
#include <support/blob_repeat.h>
#include <support/check.h>

static int
do_test (void)
{
  struct support_blob_repeat repeat
    = support_blob_repeat_allocate ("5", 1, 5);
  TEST_COMPARE_BLOB (repeat.start, repeat.size, "55555", 5);
  support_blob_repeat_free (&repeat);

  repeat = support_blob_repeat_allocate ("ABC", 3, 3);
  TEST_COMPARE_BLOB (repeat.start, repeat.size, "ABCABCABC", 9);
  support_blob_repeat_free (&repeat);

  repeat = support_blob_repeat_allocate ("abc", 4, 3);
  TEST_COMPARE_BLOB (repeat.start, repeat.size, "abc\0abc\0abc", 12);
  support_blob_repeat_free (&repeat);

  size_t gigabyte = 1U << 30;
  repeat = support_blob_repeat_allocate ("X", 1, gigabyte + 1);
  if (repeat.start == NULL)
    puts ("warning: not enough memory for 1 GiB mapping");
  else
    {
      TEST_COMPARE (repeat.size, gigabyte + 1);
      {
        unsigned char *p = repeat.start;
        for (size_t i = 0; i < gigabyte + 1; ++i)
          if (p[i] != 'X')
            FAIL_EXIT1 ("invalid byte 0x%02x at %zu", p[i], i);

        /* Check that there is no sharing across the mapping.  */
        p[0] = 'Y';
        p[1U << 24] = 'Z';
        for (size_t i = 0; i < gigabyte + 1; ++i)
          if (i == 0)
            TEST_COMPARE (p[i], 'Y');
          else if (i == 1U << 24)
            TEST_COMPARE (p[i], 'Z');
          else if (p[i] != 'X')
            FAIL_EXIT1 ("invalid byte 0x%02x at %zu", p[i], i);
      }
    }
  support_blob_repeat_free (&repeat);

  for (int do_shared = 0; do_shared < 2; ++do_shared)
    {
      if (do_shared)
        repeat = support_blob_repeat_allocate_shared ("012345678", 9,
                                                      10 * 1000 * 1000);
      else
        repeat = support_blob_repeat_allocate ("012345678", 9,
                                               10 * 1000 * 1000);
      if (repeat.start == NULL)
        puts ("warning: not enough memory for large mapping");
      else
        {
          unsigned char *p = repeat.start;
          for (int i = 0; i < 10 * 1000 * 1000; ++i)
            for (int j = 0; j <= 8; ++j)
              if (p[i * 9 + j] != '0' + j)
                {
                  printf ("error: element %d index %d\n", i, j);
                  TEST_COMPARE (p[i * 9 + j], '0' + j);
                }

          enum { total_size = 9 * 10 * 1000 * 1000 };
          p[total_size - 1] = '\0';
          asm ("" ::: "memory");
          if (do_shared)
            /* The write is repeated in multiple places earlier in the
               string due to page sharing.  */
            TEST_VERIFY (strlen (repeat.start) < total_size - 1);
          else
            TEST_COMPARE (strlen (repeat.start), total_size - 1);
        }
      support_blob_repeat_free (&repeat);
    }

  return 0;
}

#include <support/test-driver.c>
