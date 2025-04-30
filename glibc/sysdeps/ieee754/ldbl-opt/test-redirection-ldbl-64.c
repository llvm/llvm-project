/* Test if __LDBL_COMPAT redirections conflict with other types.
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
   <http://www.gnu.org/licenses/>.  */

/* Redirecting functions twice may lead to -Werror=pragmas errors.
   In a __LDBL_COMPAT environment, only long double functions should be
   redirected.  This test redirects math functions to a dummy function in
   order to validate if they have not been redirected before.  */

#include <math.h>
#include <complex.h>

#if defined __FINITE_MATH_ONLY__ && __FINITE_MATH_ONLY__ > 0
# error "This test should never request finite functions"
#endif

#define MATH_REDIRX(function, to) \
  extern typeof (function) function __asm__ ("" # to);
#define MATH_REDIR(function) MATH_REDIRX (function, __ ## function)

#if __HAVE_FLOAT32
# define MATH_F32(function) MATH_REDIR(function ## f32)
#else
# define MATH_F32(function)
#endif

#if __HAVE_FLOAT32X
# define MATH_F32X(function) MATH_REDIR(function ## f32x)
#else
# define MATH_F32X(function)
#endif

#if __HAVE_FLOAT64
# define MATH_F64(function) MATH_REDIR(function ## f64)
#else
# define MATH_F64(function)
#endif

#if __HAVE_FLOAT64X
# define MATH_F64X(function) MATH_REDIR(function ## f64x)
#else
# define MATH_F64X(function)
#endif

#define MATH_FUNCTION(function) \
  MATH_REDIR(function); \
  MATH_REDIR(function ## f); \
  MATH_F32(function); \
  MATH_F32X(function); \
  MATH_F64(function); \
  MATH_F64X(function);

MATH_FUNCTION (acos);
MATH_FUNCTION (asin);
MATH_FUNCTION (exp);
MATH_FUNCTION (floor);
MATH_FUNCTION (ldexp);
MATH_FUNCTION (log);
MATH_FUNCTION (sin);
MATH_FUNCTION (cabs);
MATH_FUNCTION (cacos);
MATH_FUNCTION (casin);
MATH_FUNCTION (clog);
MATH_FUNCTION (csin);

static int
do_test (void)
{
  /* This is a compilation test.  */
  return 0;
}

#include <support/test-driver.c>
