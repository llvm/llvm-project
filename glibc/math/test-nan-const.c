/* Test nan functions do not have const attribute.  Bug 23277.
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

#include <math.h>
#include <string.h>

static int
do_test (void)
{
  char buf[2] = { '2', 0 };
  float a = nanf (buf);
  buf[0] = '3';
  float b = nanf (buf);
  return memcmp (&a, &b, sizeof (float)) == 0;
}

#include <support/test-driver.c>
