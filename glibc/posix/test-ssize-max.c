/* Test SSIZE_MAX value and type.
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

#include <limits.h>
#include <sys/types.h>

/* Test SSIZE_MAX has type ssize_t.  */
ssize_t x;
extern __typeof (SSIZE_MAX) x;

/* Test the value of SSIZE_MAX.  */
_Static_assert (SSIZE_MAX == (sizeof (ssize_t) == sizeof (int)
			      ? INT_MAX
			      : LONG_MAX),
		"value of SSIZE_MAX");

static int
do_test (void)
{
  /* This is a compilation test.  */
  return 0;
}

#include <support/test-driver.c>
