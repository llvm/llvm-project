/* Check __ppc_get_timebase() for architecture changes
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

/* Test if __ppc_get_timebase() is compatible with the current processor and if
   it's changing between reads.  A read failure might indicate a Power ISA or
   binutils change.  */

#include <inttypes.h>
#include <stdio.h>
#include <stdint.h>

#include <sys/platform/ppc.h>

static int
do_test (void)
{
  uint64_t t1, t2, t3;
  t1 = __ppc_get_timebase ();
  printf ("Time Base = %"PRIu64"\n", t1);
  t2 = __ppc_get_timebase ();
  printf ("Time Base = %"PRIu64"\n", t2);
  t3 = __ppc_get_timebase ();
  printf ("Time Base = %"PRIu64"\n", t3);
  if (t1 != t2 && t1 != t3 && t2 != t3)
    return 0;

  printf ("Fail: timebase reads should always be different.\n");
  return 1;
}

#include <support/test-driver.c>
