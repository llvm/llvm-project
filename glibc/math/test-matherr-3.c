/* Test matherr not supported for new binaries.
   Copyright (C) 1997-2021 Free Software Foundation, Inc.
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
#include <stdio.h>
#include <stdlib.h>

#include <math-svid-compat.h>

_LIB_VERSION_TYPE _LIB_VERSION = _SVID_;

static int fail = 0;

int
matherr (struct exception *s)
{
  printf ("matherr is working, but should not be\n");
  fail = 1;
  return 1;
}

static int
do_test (void)
{
  acos (2.0);
  return fail;
}

#include <support/test-driver.c>
