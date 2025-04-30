/* Test libnldbl_nonshared.a wrappers call visible functions (bug 23735).
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

/* To use the wrappers, this file deliberately does not include
   <math.h>.  */

long double sqrtl (long double);
long double ceill (long double);
long double floorl (long double);
long double rintl (long double);
long double truncl (long double);
long double roundl (long double);

volatile long double x = 2.25L;

static int
do_test (void)
{
  return (sqrtl (x) != 1.5L
	  || ceill (x) != 3.0L
	  || floorl (x) != 2.0L
	  || rintl (x) != 2.0L
	  || truncl (x) != 2.0L
	  || roundl (x) != 2.0L);
}

#include <support/test-driver.c>
