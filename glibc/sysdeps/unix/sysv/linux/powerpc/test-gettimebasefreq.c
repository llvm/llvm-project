/* Check __ppc_get_timebase_freq() for architecture changes
   Copyright (C) 2012-2021 Free Software Foundation, Inc.
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

/* Test if __ppc_get_timebase_freq() works and is different from zero.  A read
   failure might indicate a Linux Kernel change.
   This test also use the frequency to compute the real elapsed time.  */

#include <inttypes.h>
#include <stdio.h>
#include <stdint.h>

#include <sys/platform/ppc.h>

/* Maximum value of the Time Base Register: 2^60 - 1.  */
#define MAX_TB 0xFFFFFFFFFFFFFFF

static int
do_test (void)
{
  uint64_t t1, t2, f, diff;

  t1 = __ppc_get_timebase ();
  printf ("t1 = %"PRIu64"\n", t1);

  f = __ppc_get_timebase_freq ();
  printf ("Time Base frequency = %"PRIu64" Hz\n", f);

  if (f == 0) {
      printf ("Fail: The time base frequency can't be zero.\n");
      return 1;
  }

  t2 = __ppc_get_timebase ();
  printf ("t2 = %"PRIu64"\n", t2);

  if (t2 > t1) {
    diff = t2 - t1;
  } else {
    diff = (MAX_TB - t2) + t1;
  }

  printf ("Elapsed time  = %1.2f usecs\n", (double) diff * 1000000 / f );

  return 0;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
