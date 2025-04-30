/* Copyright (C) 2000-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@redhat.com>, 2000.

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

#include <limits.h>
#include <math.h>
#include <stdio.h>


static int
do_test (void)
{
  int result = 0;

  if (FP_ILOGB0 != INT_MIN && FP_ILOGB0 != -INT_MAX)
    {
      puts ("FP_ILOGB0 has no valid value");
      result = 1;
    }
  else
    puts ("FP_ILOGB0 value is OK");

  if (FP_ILOGBNAN != INT_MIN && FP_ILOGBNAN != INT_MAX)
    {
      puts ("FP_ILOBNAN has no valid value");
      result = 1;
    }
  else
    puts ("FP_ILOGBNAN value is OK");

  return result;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
