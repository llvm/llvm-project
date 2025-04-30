/* Tests of functions to access `dev_t' values.
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

#include <sys/types.h>
#include <sys/sysmacros.h>
#include <stdio.h>
#include <inttypes.h>

/* Confirm that makedev (major (d), minor (d)) == d.  */
static int
do_test_split_combine (dev_t d1)
{
  unsigned int maj = major (d1);
  unsigned int min = minor (d1);
  dev_t d2 = makedev (maj, min);
  if (d1 != d2)
    {
      printf ("FAIL: %016" PRIx64 " != %016" PRIx64 " (maj %08x min %08x)\n",
	      (uint64_t)d2, (uint64_t)d1, maj, min);
      return 1;
    }
  else
    return 0;
}

/* Confirm that major (makedev (maj, min)) == maj and
   minor (makedev (maj, min)) == min.  */
static int
do_test_combine_split (unsigned int maj1, unsigned int min1)
{
  dev_t d = makedev (maj1, min1);
  unsigned int maj2 = major (d);
  unsigned int min2 = minor (d);
  if (maj1 != maj2 && min1 != min2)
    {
      printf ("FAIL: %08x != %08x, %08x != %08x (dev %016" PRIx64 ")\n",
	      maj2, maj1, min2, min1, (uint64_t)d);
      return 1;
    }
  else if (maj1 != maj2)
    {
      printf ("FAIL: %08x != %08x, %08x == %08x (dev %016" PRIx64 ")\n",
	      maj2, maj1, min2, min1, (uint64_t)d);
      return 1;
    }
  else if (min1 != min2)
    {
      printf ("FAIL: %08x == %08x, %08x != %08x (dev %016" PRIx64 ")\n",
	      maj2, maj1, min2, min1, (uint64_t)d);
      return 1;
    }
  else
    return 0;
}

static int
do_test (void)
{
  dev_t d;
  unsigned int maj, min;
  int status = 0;

  /* Test the traditional range (16-bit dev_t, 8-bit each maj/min)
     exhaustively.  */
  for (d = 0; d <= 0xFFFF; d++)
    status |= do_test_split_combine (d);

  for (maj = 0; maj <= 0xFF; maj++)
    for (min = 0; min <= 0xFF; min++)
      status |= do_test_combine_split (maj, min);

  /* Test glibc's expanded range (64-bit dev_t, 32-bit each maj/min).
     Exhaustive testing would take much too long, instead we shift a
     pair of 1-bits over each range.  */
  {
    unsigned int a, b;
    for (a = 0; a <= 63; a++)
      do_test_split_combine (((dev_t) 0x03) << a);

    for (a = 0; a < 31; a++)
      for (b = 0; b <= 31; b++)
	do_test_combine_split (0x03u << a, 0x03u << b);
  }

  return status;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
