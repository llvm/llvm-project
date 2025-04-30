/* Test integer wraparound in hcreate.
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

#include <errno.h>
#include <limits.h>
#include <search.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/resource.h>

static void
test_size (size_t size)
{
  int res = hcreate (size);
  if (res == 0)
    {
      if (errno == ENOMEM)
        return;
      printf ("error: hcreate (%zu): %m\n", size);
      exit (1);
    }
  char *keys[100];
  for (int i = 0; i < 100; ++i)
    {
      if (asprintf (keys + i, "%d", i) < 0)
        {
          printf ("error: asprintf: %m\n");
          exit (1);
        }
      ENTRY e = { keys[i], (char *) "value" };
      if (hsearch (e, ENTER) == NULL)
        {
          printf ("error: hsearch (\"%s\"): %m\n", keys[i]);
          exit (1);
        }
    }
  hdestroy ();

  for (int i = 0; i < 100; ++i)
    free (keys[i]);
}

static int
do_test (void)
{
  /* Limit the size of the process, so that memory allocation will
     fail without impacting the entire system.  */
  {
    struct rlimit limit;
    if (getrlimit (RLIMIT_AS, &limit) != 0)
      {
        printf ("getrlimit (RLIMIT_AS) failed: %m\n");
        return 1;
      }
    long target = 100 * 1024 * 1024;
    if (limit.rlim_cur == RLIM_INFINITY || limit.rlim_cur > target)
      {
        limit.rlim_cur = target;
        if (setrlimit (RLIMIT_AS, &limit) != 0)
          {
            printf ("setrlimit (RLIMIT_AS) failed: %m\n");
            return 1;
          }
      }
  }

  test_size (500);
  test_size (-1);
  test_size (-3);
  test_size (INT_MAX - 2);
  test_size (INT_MAX - 1);
  test_size (INT_MAX);
  test_size (((unsigned) INT_MAX) + 1);
  test_size (UINT_MAX - 2);
  test_size (UINT_MAX - 1);
  test_size (UINT_MAX);
  return 0;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
