/* Test for the C++ implementation of iscanonical.
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

#define _GNU_SOURCE 1
#include <math.h>
#include <stdio.h>

static int errors;

template <class T>
static void
check_type ()
{
  T val = 0;

  /* Check if iscanonical is available in C++ mode (bug 22235).  */
  if (iscanonical (val) == 0)
    errors++;
}

static int
do_test (void)
{
  check_type<float> ();
  check_type<double> ();
  check_type<long double> ();
#if __HAVE_DISTINCT_FLOAT128
  check_type<_Float128> ();
#endif
  return errors != 0;
}

#include <support/test-driver.c>
