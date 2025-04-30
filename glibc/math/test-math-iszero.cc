/* Test for the C++ implementation of iszero.
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

#define _GNU_SOURCE 1
#include <math.h>
#include <stdio.h>

#include <limits>

/* Support for _Float128 in std::numeric_limits is limited.
   Include ieee754_float128.h and use the bitfields in the union
   ieee854_float128.ieee_nan to build corner-case inputs.  */
#if __HAVE_DISTINCT_FLOAT128
# include <ieee754_float128.h>
#endif

static bool errors;

static void
check (int actual, int expected, const char *actual_expr, int line)
{
  if (actual != expected)
    {
      errors = true;
      printf ("%s:%d: error: %s\n", __FILE__, line, actual_expr);
      printf ("%s:%d:   expected: %d\n", __FILE__, line, expected);
      printf ("%s:%d:   actual: %d\n", __FILE__, line, actual);
    }
}

#define CHECK(actual, expected) \
  check ((actual), (expected), #actual, __LINE__)

template <class T>
static void
check_type ()
{
  typedef std::numeric_limits<T> limits;
  CHECK (iszero (T{}), 1);
  CHECK (iszero (T{0}), 1);
  CHECK (iszero (T{-0.0}), 1);
  CHECK (iszero (T{1}), 0);
  CHECK (iszero (T{-1}), 0);
  CHECK (iszero (limits::min ()), 0);
  CHECK (iszero (-limits::min ()), 0);
  CHECK (iszero (limits::max ()), 0);
  CHECK (iszero (-limits::max ()), 0);
  if (limits::has_infinity)
    {
      CHECK (iszero (limits::infinity ()), 0);
      CHECK (iszero (-limits::infinity ()), 0);
    }
  CHECK (iszero (limits::epsilon ()), 0);
  CHECK (iszero (-limits::epsilon ()), 0);
  if (limits::has_quiet_NaN)
    CHECK (iszero (limits::quiet_NaN ()), 0);
  if (limits::has_signaling_NaN)
    CHECK (iszero (limits::signaling_NaN ()), 0);
  if (limits::has_signaling_NaN)
    CHECK (iszero (limits::signaling_NaN ()), 0);
  CHECK (iszero (limits::denorm_min ()),
         std::numeric_limits<T>::has_denorm == std::denorm_absent);
  CHECK (iszero (-limits::denorm_min ()),
         std::numeric_limits<T>::has_denorm == std::denorm_absent);
}

#if __HAVE_DISTINCT_FLOAT128
static void
check_float128 ()
{
  ieee854_float128 q;

  q.d = 0.0Q;
  CHECK (iszero (q.d), 1);
  q.d = -0.0Q;
  CHECK (iszero (q.d), 1);
  q.d = 1.0Q;
  CHECK (iszero (q.d), 0);
  q.d = -1.0Q;
  CHECK (iszero (q.d), 0);

  /* Normal min.  */
  q.ieee.negative = 0;
  q.ieee.exponent = 0x0001;
  q.ieee.mantissa0 = 0x0000;
  q.ieee.mantissa1 = 0x00000000;
  q.ieee.mantissa2 = 0x00000000;
  q.ieee.mantissa3 = 0x00000000;
  CHECK (iszero (q.d), 0);
  q.ieee.negative = 1;
  CHECK (iszero (q.d), 0);

  /* Normal max.  */
  q.ieee.negative = 0;
  q.ieee.exponent = 0x7FFE;
  q.ieee.mantissa0 = 0xFFFF;
  q.ieee.mantissa1 = 0xFFFFFFFF;
  q.ieee.mantissa2 = 0xFFFFFFFF;
  q.ieee.mantissa3 = 0xFFFFFFFF;
  CHECK (iszero (q.d), 0);
  q.ieee.negative = 1;
  CHECK (iszero (q.d), 0);

  /* Infinity.  */
  q.ieee.negative = 0;
  q.ieee.exponent = 0x7FFF;
  q.ieee.mantissa0 = 0x0000;
  q.ieee.mantissa1 = 0x00000000;
  q.ieee.mantissa2 = 0x00000000;
  q.ieee.mantissa3 = 0x00000000;
  CHECK (iszero (q.d), 0);

  /* Quiet NaN.  */
  q.ieee_nan.quiet_nan = 1;
  q.ieee_nan.mantissa0 = 0x0000;
  CHECK (iszero (q.d), 0);

  /* Signaling NaN.  */
  q.ieee_nan.quiet_nan = 0;
  q.ieee_nan.mantissa0 = 0x4000;
  CHECK (iszero (q.d), 0);

  /* Denormal min.  */
  q.ieee.negative = 0;
  q.ieee.exponent = 0x0000;
  q.ieee.mantissa0 = 0x0000;
  q.ieee.mantissa1 = 0x00000000;
  q.ieee.mantissa2 = 0x00000000;
  q.ieee.mantissa3 = 0x00000001;
  CHECK (iszero (q.d), 0);
  q.ieee.negative = 1;
  CHECK (iszero (q.d), 0);
}
#endif

static int
do_test (void)
{
  check_type<float> ();
  check_type<double> ();
  check_type<long double> ();
#if __HAVE_DISTINCT_FLOAT128
  check_float128 ();
#endif
  return errors;
}

#define TEST_FUNCTION do_test ()
#include "../test-skeleton.c"
