/* Test compilation of tgmath macros.
   Copyright (C) 2005-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Andreas Jaeger <aj@suse.de>, 2005.

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
#include <complex.h>
#include <tgmath.h>
#include <stdio.h>

static int errors = 0;

static void
our_error (const char *c)
{
  puts (c);
  ++errors;
}

#define CHECK_RET_CONST_TYPE(func, rettype, name)			\
  if (sizeof (func) != sizeof (rettype))				\
    our_error ("Return size of " #name " is " #func" wrong");

#define CHECK_RET_CONST_FLOAT(func, name)		\
  CHECK_RET_CONST_TYPE (func, float, name)

#define CHECK_RET_CONST_DOUBLE(func, name)		\
  CHECK_RET_CONST_TYPE (func, double, name)

static int
do_test (void)
{
  int i;
  float f;
  double d;

  CHECK_RET_CONST_DOUBLE (sin (i), "sin (i)");
  CHECK_RET_CONST_DOUBLE (pow (i, i), "pow (i, i)");
  CHECK_RET_CONST_DOUBLE (pow (i, i), "pow (i, i)");
  CHECK_RET_CONST_DOUBLE (pow (i, f), "pow (i, f)");
  CHECK_RET_CONST_DOUBLE (pow (i, d), "pow (i, d)");
  CHECK_RET_CONST_DOUBLE (pow (f, i), "pow (f, i)");
  CHECK_RET_CONST_DOUBLE (pow (d, i), "pow (d, i)");
  CHECK_RET_CONST_DOUBLE (fma (i, i, i), "fma (i, i, i)");
  CHECK_RET_CONST_DOUBLE (fma (f, i, i), "fma (f, i, i)");
  CHECK_RET_CONST_DOUBLE (fma (i, f, i), "fma (i, f, i)");
  CHECK_RET_CONST_DOUBLE (fma (i, i, f), "fma (i, i, f)");
  CHECK_RET_CONST_DOUBLE (fma (d, i, i), "fma (d, i, i)");
  CHECK_RET_CONST_DOUBLE (fma (i, d, i), "fma (i, d, i)");
  CHECK_RET_CONST_DOUBLE (fma (i, i, d), "fma (i, i, d)");

  return errors != 0;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
