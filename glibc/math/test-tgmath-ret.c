/* Test compilation of tgmath macros.
   Copyright (C) 2003-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Andreas Jaeger <aj@suse.de>, 2003.

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
#include <stdint.h>
#include <stdio.h>

static float fx;
static double dx;
static long double lx;
static int rm = FP_INT_UPWARD;
static unsigned int width = 64;
static int errors = 0;

static void
our_error (const char *c)
{
  puts (c);
  ++errors;
}

/* First function where the return type is constant.  */

#define CHECK_RET_CONST_TYPE(func, rettype, arg, name, ...)		\
  if (sizeof (func (arg, ## __VA_ARGS__)) != sizeof (rettype))		\
    our_error ("Return size of " #func " is wrong with " #name " argument");

#define CHECK_RET_CONST_FLOAT(func, rettype, ...)			\
  CHECK_RET_CONST_TYPE (func, rettype, fx, float, ## __VA_ARGS__)
#define CHECK_RET_CONST_DOUBLE(func, rettype, ...)			\
  CHECK_RET_CONST_TYPE (func, rettype, dx, double, ## __VA_ARGS__)
#define CHECK_RET_CONST_LDOUBLE(func, rettype, ...)			\
  CHECK_RET_CONST_TYPE (func, rettype, lx, long double, ## __VA_ARGS__)

#define CHECK_RET_CONST(func, rettype, ...)			\
static void							\
check_return_ ##func (void)					\
{								\
  CHECK_RET_CONST_FLOAT (func, rettype, ## __VA_ARGS__)		\
  CHECK_RET_CONST_DOUBLE (func, rettype, ## __VA_ARGS__)	\
  CHECK_RET_CONST_LDOUBLE (func, rettype, ## __VA_ARGS__)	\
}

CHECK_RET_CONST(ilogb, int)
CHECK_RET_CONST(llogb, long)
CHECK_RET_CONST(lrint, long)
CHECK_RET_CONST(lround, long)
CHECK_RET_CONST(llrint, long long)
CHECK_RET_CONST(llround, long long)
CHECK_RET_CONST(fromfp, intmax_t, rm, width)
CHECK_RET_CONST(ufromfp, uintmax_t, rm, width)
CHECK_RET_CONST(fromfpx, intmax_t, rm, width)
CHECK_RET_CONST(ufromfpx, uintmax_t, rm, width)

static int
do_test (void)
{
  check_return_ilogb ();
  check_return_llogb ();
  check_return_lrint ();
  check_return_lround ();
  check_return_llrint ();
  check_return_llround ();
  check_return_fromfp ();
  check_return_ufromfp ();
  check_return_fromfpx ();
  check_return_ufromfpx ();

  printf ("%Zd\n", sizeof (carg (lx)));

  return errors != 0;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
