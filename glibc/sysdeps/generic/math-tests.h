/* Configuration for math tests.  Generic version.
   Copyright (C) 2013-2021 Free Software Foundation, Inc.
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

#include <bits/floatn.h>

/* Expand the appropriate macro for whether to enable tests for a
   given type.  */
#if __HAVE_DISTINCT_FLOAT128
# define MATH_TESTS_TG(PREFIX, ARGS, TYPE)				\
  (sizeof (TYPE) == sizeof (float) ? PREFIX ## float ARGS		\
   : sizeof (TYPE) == sizeof (double) ? PREFIX ## double ARGS		\
   : __builtin_types_compatible_p (TYPE, _Float128) ? PREFIX ## float128 ARGS \
   : PREFIX ## long_double ARGS)
# else
# define MATH_TESTS_TG(PREFIX, ARGS, TYPE)				\
  (sizeof (TYPE) == sizeof (float) ? PREFIX ## float ARGS		\
   : sizeof (TYPE) == sizeof (double) ? PREFIX ## double ARGS		\
   : PREFIX ## long_double ARGS)
#endif

/* Return nonzero value if to run tests involving sNaN values for X.  */
#define SNAN_TESTS(x) MATH_TESTS_TG (SNAN_TESTS_, , x)

#define ROUNDING_TESTS(TYPE, MODE)		\
  MATH_TESTS_TG (ROUNDING_TESTS_, (MODE), TYPE)

#define EXCEPTION_TESTS(TYPE) MATH_TESTS_TG (EXCEPTION_TESTS_, , TYPE)

#include <math-tests-exceptions.h>
#include <math-tests-rounding.h>
#include <math-tests-snan.h>
#include <math-tests-snan-cast.h>
#include <math-tests-snan-payload.h>
#include <math-tests-trap.h>
#include <math-tests-trap-force.h>
