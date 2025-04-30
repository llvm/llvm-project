/* Test for memory leak with large width and precision.
   Copyright (C) 1991-2021 Free Software Foundation, Inc.
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

#include <mcheck.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/resource.h>
#include <wchar.h>

static int
do_test (void)
{
  mtrace ();

  int ret;
  {
    char *result;
    ret = asprintf (&result, "%133000.133001x", 17);
    if (ret < 0)
      {
        printf ("error: asprintf: %m\n");
        return 1;
      }
    free (result);
  }
  {
    wchar_t *result = calloc (ret + 1, sizeof (wchar_t));
    if (result == NULL)
      {
        printf ("error: calloc (%d, %zu): %m", ret + 1, sizeof (wchar_t));
        return 1;
      }

    ret = swprintf (result, ret + 1, L"%133000.133001x", 17);
    if (ret < 0)
      {
        printf ("error: swprintf: %d (%m)\n", ret);
        return 1;
      }
    free (result);
  }

  /* Limit the size of the process, so that the second allocation will
     fail.  */
  {
    struct rlimit limit;
    if (getrlimit (RLIMIT_AS, &limit) != 0)
      {
        printf ("getrlimit (RLIMIT_AS) failed: %m\n");
        return 1;
      }
    long target = 200 * 1024 * 1024;
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

  {
    char *result;
    ret = asprintf (&result, "%133000.999999999x", 17);
    if (ret >= 0)
      {
        printf ("error: asprintf: incorrect result %d\n", ret);
        return 1;
      }
  }
  {
    wchar_t result[100];
    if (result == NULL)
      {
        printf ("error: calloc (%d, %zu): %m", ret + 1, sizeof (wchar_t));
        return 1;
      }

    ret = swprintf (result, 100, L"%133000.999999999x", 17);
    if (ret >= 0)
      {
        printf ("error: swprintf: incorrect result %d\n", ret);
        return 1;
      }
  }

  return 0;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
