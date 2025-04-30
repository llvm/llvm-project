/* Test scalb.
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

#include "libm-test-driver.c"

static const struct test_ff_f_data scalb_test_data[] =
  {
#if !TEST_FLOATN
    /* Results in this case are unspecified by POSIX, so, for an
       otherwise fully-determined function, spurious "inexact"
       exceptions are OK.  */
    TEST_ff_f (scalb, 2.0, 0.5, qnan_value, INVALID_EXCEPTION|ERRNO_EDOM),
    TEST_ff_f (scalb, 3.0, -2.5, qnan_value, INVALID_EXCEPTION|ERRNO_EDOM),

    TEST_ff_f (scalb, 0, qnan_value, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (scalb, 0, -qnan_value, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (scalb, 1, qnan_value, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (scalb, 1, -qnan_value, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (scalb, 0, snan_value, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (scalb, 0, -snan_value, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (scalb, 1, snan_value, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (scalb, 1, -snan_value, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),

    TEST_ff_f (scalb, 1, 0, 1, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (scalb, -1, 0, -1, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),

    TEST_ff_f (scalb, 0, plus_infty, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION|ERRNO_EDOM),
    TEST_ff_f (scalb, minus_zero, plus_infty, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION|ERRNO_EDOM),

    TEST_ff_f (scalb, 0, 2, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (scalb, minus_zero, -4, minus_zero, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (scalb, 0, 0, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (scalb, minus_zero, 0, minus_zero, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (scalb, 0, -1, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (scalb, minus_zero, -10, minus_zero, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (scalb, 0, minus_infty, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (scalb, minus_zero, minus_infty, minus_zero, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),

    TEST_ff_f (scalb, plus_infty, -1, plus_infty, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (scalb, minus_infty, -10, minus_infty, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (scalb, plus_infty, 0, plus_infty, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (scalb, minus_infty, 0, minus_infty, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (scalb, plus_infty, 2, plus_infty, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (scalb, minus_infty, 100, minus_infty, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),

    TEST_ff_f (scalb, 0.1L, minus_infty, 0.0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (scalb, -0.1L, minus_infty, minus_zero, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),

    TEST_ff_f (scalb, 1, plus_infty, plus_infty, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (scalb, -1, plus_infty, minus_infty, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (scalb, plus_infty, plus_infty, plus_infty, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (scalb, minus_infty, plus_infty, minus_infty, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),

    TEST_ff_f (scalb, plus_infty, minus_infty, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION|ERRNO_EDOM),
    TEST_ff_f (scalb, minus_infty, minus_infty, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION|ERRNO_EDOM),

    TEST_ff_f (scalb, qnan_value, 1, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (scalb, -qnan_value, 1, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (scalb, snan_value, 1, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (scalb, -snan_value, 1, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (scalb, 1, qnan_value, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (scalb, 1, -qnan_value, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (scalb, 1, snan_value, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (scalb, 1, -snan_value, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (scalb, qnan_value, 0.5, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (scalb, -qnan_value, 0.5, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (scalb, snan_value, 0.5, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (scalb, -snan_value, 0.5, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (scalb, 0.5, qnan_value, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (scalb, 0.5, -qnan_value, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (scalb, 0.5, snan_value, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (scalb, 0.5, -snan_value, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (scalb, qnan_value, 0, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (scalb, -qnan_value, 0, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (scalb, snan_value, 0, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (scalb, -snan_value, 0, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (scalb, 0, qnan_value, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (scalb, 0, -qnan_value, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (scalb, 0, snan_value, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (scalb, 0, -snan_value, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (scalb, qnan_value, plus_infty, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (scalb, -qnan_value, plus_infty, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (scalb, snan_value, plus_infty, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (scalb, -snan_value, plus_infty, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (scalb, plus_infty, qnan_value, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (scalb, plus_infty, -qnan_value, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (scalb, plus_infty, snan_value, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (scalb, plus_infty, -snan_value, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (scalb, qnan_value, minus_infty, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (scalb, -qnan_value, minus_infty, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (scalb, snan_value, minus_infty, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (scalb, -snan_value, minus_infty, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (scalb, minus_infty, qnan_value, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (scalb, minus_infty, -qnan_value, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (scalb, minus_infty, snan_value, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (scalb, minus_infty, -snan_value, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (scalb, qnan_value, qnan_value, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (scalb, qnan_value, -qnan_value, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (scalb, -qnan_value, qnan_value, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (scalb, -qnan_value, -qnan_value, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (scalb, snan_value, qnan_value, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (scalb, snan_value, -qnan_value, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (scalb, -snan_value, qnan_value, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (scalb, -snan_value, -qnan_value, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (scalb, qnan_value, snan_value, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (scalb, qnan_value, -snan_value, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (scalb, -qnan_value, snan_value, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (scalb, -qnan_value, -snan_value, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (scalb, snan_value, snan_value, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (scalb, snan_value, -snan_value, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (scalb, -snan_value, snan_value, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (scalb, -snan_value, -snan_value, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),

    TEST_ff_f (scalb, max_value, max_value, plus_oflow, INEXACT_EXCEPTION|OVERFLOW_EXCEPTION|ERRNO_PLUS_OFLOW|XFAIL_ROUNDING_IBM128_LIBGCC),
    TEST_ff_f (scalb, max_value, -max_value, plus_uflow, INEXACT_EXCEPTION|UNDERFLOW_EXCEPTION|ERRNO_PLUS_UFLOW),
    TEST_ff_f (scalb, 1, max_value, plus_oflow, INEXACT_EXCEPTION|OVERFLOW_EXCEPTION|ERRNO_PLUS_OFLOW|XFAIL_ROUNDING_IBM128_LIBGCC),
    TEST_ff_f (scalb, 1, -max_value, plus_uflow, INEXACT_EXCEPTION|UNDERFLOW_EXCEPTION|ERRNO_PLUS_UFLOW),
    TEST_ff_f (scalb, min_value, max_value, plus_oflow, INEXACT_EXCEPTION|OVERFLOW_EXCEPTION|ERRNO_PLUS_OFLOW|XFAIL_ROUNDING_IBM128_LIBGCC),
    TEST_ff_f (scalb, min_value, -max_value, plus_uflow, INEXACT_EXCEPTION|UNDERFLOW_EXCEPTION|ERRNO_PLUS_UFLOW),
    TEST_ff_f (scalb, min_subnorm_value, max_value, plus_oflow, INEXACT_EXCEPTION|OVERFLOW_EXCEPTION|ERRNO_PLUS_OFLOW|XFAIL_ROUNDING_IBM128_LIBGCC),
    TEST_ff_f (scalb, min_subnorm_value, -max_value, plus_uflow, INEXACT_EXCEPTION|UNDERFLOW_EXCEPTION|ERRNO_PLUS_UFLOW),
    TEST_ff_f (scalb, -max_value, max_value, minus_oflow, INEXACT_EXCEPTION|OVERFLOW_EXCEPTION|ERRNO_MINUS_OFLOW|XFAIL_ROUNDING_IBM128_LIBGCC),
    TEST_ff_f (scalb, -max_value, -max_value, minus_uflow, INEXACT_EXCEPTION|UNDERFLOW_EXCEPTION|ERRNO_MINUS_UFLOW),
    TEST_ff_f (scalb, -1, max_value, minus_oflow, INEXACT_EXCEPTION|OVERFLOW_EXCEPTION|ERRNO_MINUS_OFLOW|XFAIL_ROUNDING_IBM128_LIBGCC),
    TEST_ff_f (scalb, -1, -max_value, minus_uflow, INEXACT_EXCEPTION|UNDERFLOW_EXCEPTION|ERRNO_MINUS_UFLOW),
    TEST_ff_f (scalb, -min_value, max_value, minus_oflow, INEXACT_EXCEPTION|OVERFLOW_EXCEPTION|ERRNO_MINUS_OFLOW|XFAIL_ROUNDING_IBM128_LIBGCC),
    TEST_ff_f (scalb, -min_value, -max_value, minus_uflow, INEXACT_EXCEPTION|UNDERFLOW_EXCEPTION|ERRNO_MINUS_UFLOW),
    TEST_ff_f (scalb, -min_subnorm_value, max_value, minus_oflow, INEXACT_EXCEPTION|OVERFLOW_EXCEPTION|ERRNO_MINUS_OFLOW|XFAIL_ROUNDING_IBM128_LIBGCC),
    TEST_ff_f (scalb, -min_subnorm_value, -max_value, minus_uflow, INEXACT_EXCEPTION|UNDERFLOW_EXCEPTION|ERRNO_MINUS_UFLOW),

    TEST_ff_f (scalb, 0.8L, 4, 12.8L, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (scalb, -0.854375L, 5, -27.34L, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
#endif
  };

static void
scalb_test (void)
{
#if !TEST_FLOATN
  ALL_RM_TEST (scalb, 1, scalb_test_data, RUN_TEST_LOOP_ff_f, END);
#endif
}

static void
do_test (void)
{
  scalb_test ();
}

/*
 * Local Variables:
 * mode:c
 * End:
 */
