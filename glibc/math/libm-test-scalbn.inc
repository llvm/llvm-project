/* Test scalbn.
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

static const struct test_fi_f_data scalbn_test_data[] =
  {
    TEST_fi_f (scalbn, 0, 0, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_fi_f (scalbn, minus_zero, 0, minus_zero, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),

    TEST_fi_f (scalbn, plus_infty, 1, plus_infty, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_fi_f (scalbn, minus_infty, 1, minus_infty, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_fi_f (scalbn, qnan_value, 1, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_fi_f (scalbn, -qnan_value, 1, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_fi_f (scalbn, snan_value, 1, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_fi_f (scalbn, -snan_value, 1, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),

    TEST_fi_f (scalbn, 0.8L, 4, 12.8L, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_fi_f (scalbn, -0.854375L, 5, -27.34L, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),

    TEST_fi_f (scalbn, 1, 0L, 1, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),

    TEST_fi_f (scalbn, min_value / 2, 0, min_value / 2, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_fi_f (scalbn, -min_value / 2, 0, -min_value / 2, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_fi_f (scalbn, min_value / 2, 1, min_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_fi_f (scalbn, -min_value / 2, 1, -min_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_fi_f (scalbn, min_value * 0x0.ffffp0, 0, min_value * 0x0.ffffp0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_fi_f (scalbn, -min_value * 0x0.ffffp0, 0, -min_value * 0x0.ffffp0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_fi_f (scalbn, min_subnorm_value, 0, min_subnorm_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_fi_f (scalbn, -min_subnorm_value, 0, -min_subnorm_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_fi_f (scalbn, min_subnorm_value, MANT_DIG - 1, min_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_fi_f (scalbn, -min_subnorm_value, MANT_DIG - 1, -min_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),

    TEST_fi_f (scalbn, min_value, -MANT_DIG + 1, min_subnorm_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_fi_f (scalbn, -min_value, -MANT_DIG + 1, -min_subnorm_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_fi_f (scalbn, min_value, -MANT_DIG, plus_uflow, INEXACT_EXCEPTION|UNDERFLOW_EXCEPTION|ERRNO_PLUS_UFLOW),
    TEST_fi_f (scalbn, -min_value, -MANT_DIG, minus_uflow, INEXACT_EXCEPTION|UNDERFLOW_EXCEPTION|ERRNO_MINUS_UFLOW),
    TEST_fi_f (scalbn, min_value, -MANT_DIG - 1, plus_uflow, INEXACT_EXCEPTION|UNDERFLOW_EXCEPTION|ERRNO_PLUS_UFLOW),
    TEST_fi_f (scalbn, -min_value, -MANT_DIG - 1, minus_uflow, INEXACT_EXCEPTION|UNDERFLOW_EXCEPTION|ERRNO_MINUS_UFLOW),
    TEST_fi_f (scalbn, min_value * 1.5, -MANT_DIG + 1, min_subnorm_value, INEXACT_EXCEPTION|UNDERFLOW_EXCEPTION, min_subnorm_value * 2, INEXACT_EXCEPTION|UNDERFLOW_EXCEPTION, min_subnorm_value, INEXACT_EXCEPTION|UNDERFLOW_EXCEPTION, min_subnorm_value * 2, INEXACT_EXCEPTION|UNDERFLOW_EXCEPTION),
    TEST_fi_f (scalbn, -min_value * 1.5, -MANT_DIG + 1, -min_subnorm_value * 2, INEXACT_EXCEPTION|UNDERFLOW_EXCEPTION, -min_subnorm_value * 2, INEXACT_EXCEPTION|UNDERFLOW_EXCEPTION, -min_subnorm_value, INEXACT_EXCEPTION|UNDERFLOW_EXCEPTION, -min_subnorm_value, INEXACT_EXCEPTION|UNDERFLOW_EXCEPTION),
    TEST_fi_f (scalbn, min_value * 1.5, -MANT_DIG, plus_zero, INEXACT_EXCEPTION|UNDERFLOW_EXCEPTION, min_subnorm_value, INEXACT_EXCEPTION|UNDERFLOW_EXCEPTION, plus_zero, INEXACT_EXCEPTION|UNDERFLOW_EXCEPTION, min_subnorm_value, INEXACT_EXCEPTION|UNDERFLOW_EXCEPTION),
    TEST_fi_f (scalbn, -min_value * 1.5, -MANT_DIG, -min_subnorm_value, INEXACT_EXCEPTION|UNDERFLOW_EXCEPTION, -min_subnorm_value, INEXACT_EXCEPTION|UNDERFLOW_EXCEPTION, minus_zero, INEXACT_EXCEPTION|UNDERFLOW_EXCEPTION, minus_zero, INEXACT_EXCEPTION|UNDERFLOW_EXCEPTION),
    TEST_fi_f (scalbn, min_value * 1.5, -MANT_DIG - 1, plus_uflow, INEXACT_EXCEPTION|UNDERFLOW_EXCEPTION|ERRNO_PLUS_UFLOW),
    TEST_fi_f (scalbn, -min_value * 1.5, -MANT_DIG - 1, minus_uflow, INEXACT_EXCEPTION|UNDERFLOW_EXCEPTION|ERRNO_MINUS_UFLOW),
    TEST_fi_f (scalbn, min_value * 1.25, -MANT_DIG + 1, min_subnorm_value, INEXACT_EXCEPTION|UNDERFLOW_EXCEPTION, min_subnorm_value, INEXACT_EXCEPTION|UNDERFLOW_EXCEPTION, min_subnorm_value, INEXACT_EXCEPTION|UNDERFLOW_EXCEPTION, min_subnorm_value * 2, INEXACT_EXCEPTION|UNDERFLOW_EXCEPTION),
    TEST_fi_f (scalbn, -min_value * 1.25, -MANT_DIG + 1, -min_subnorm_value * 2, INEXACT_EXCEPTION|UNDERFLOW_EXCEPTION, -min_subnorm_value, INEXACT_EXCEPTION|UNDERFLOW_EXCEPTION, -min_subnorm_value, INEXACT_EXCEPTION|UNDERFLOW_EXCEPTION, -min_subnorm_value, INEXACT_EXCEPTION|UNDERFLOW_EXCEPTION),
    TEST_fi_f (scalbn, min_value * 1.25, -MANT_DIG, plus_zero, INEXACT_EXCEPTION|UNDERFLOW_EXCEPTION, min_subnorm_value, INEXACT_EXCEPTION|UNDERFLOW_EXCEPTION, plus_zero, INEXACT_EXCEPTION|UNDERFLOW_EXCEPTION, min_subnorm_value, INEXACT_EXCEPTION|UNDERFLOW_EXCEPTION),
    TEST_fi_f (scalbn, -min_value * 1.25, -MANT_DIG, -min_subnorm_value, INEXACT_EXCEPTION|UNDERFLOW_EXCEPTION, -min_subnorm_value, INEXACT_EXCEPTION|UNDERFLOW_EXCEPTION, minus_zero, INEXACT_EXCEPTION|UNDERFLOW_EXCEPTION, minus_zero, INEXACT_EXCEPTION|UNDERFLOW_EXCEPTION),
    TEST_fi_f (scalbn, min_value * 1.25, -MANT_DIG - 1, plus_uflow, INEXACT_EXCEPTION|UNDERFLOW_EXCEPTION|ERRNO_PLUS_UFLOW),
    TEST_fi_f (scalbn, -min_value * 1.25, -MANT_DIG - 1, minus_uflow, INEXACT_EXCEPTION|UNDERFLOW_EXCEPTION|ERRNO_MINUS_UFLOW),

    TEST_fi_f (scalbn, 1, INT_MAX, plus_oflow, INEXACT_EXCEPTION|OVERFLOW_EXCEPTION|ERRNO_PLUS_OFLOW|XFAIL_ROUNDING_IBM128_LIBGCC),
    TEST_fi_f (scalbn, 1, INT_MIN, plus_uflow, INEXACT_EXCEPTION|UNDERFLOW_EXCEPTION|ERRNO_PLUS_UFLOW),
    TEST_fi_f (scalbn, max_value, INT_MAX, plus_oflow, INEXACT_EXCEPTION|OVERFLOW_EXCEPTION|ERRNO_PLUS_OFLOW|XFAIL_ROUNDING_IBM128_LIBGCC),
    TEST_fi_f (scalbn, max_value, INT_MIN, plus_uflow, INEXACT_EXCEPTION|UNDERFLOW_EXCEPTION|ERRNO_PLUS_UFLOW),
    TEST_fi_f (scalbn, min_value, INT_MAX, plus_oflow, INEXACT_EXCEPTION|OVERFLOW_EXCEPTION|ERRNO_PLUS_OFLOW|XFAIL_ROUNDING_IBM128_LIBGCC),
    TEST_fi_f (scalbn, min_value, INT_MIN, plus_uflow, INEXACT_EXCEPTION|UNDERFLOW_EXCEPTION|ERRNO_PLUS_UFLOW),
    TEST_fi_f (scalbn, min_value / 4, INT_MAX, plus_oflow, INEXACT_EXCEPTION|OVERFLOW_EXCEPTION|ERRNO_PLUS_OFLOW|XFAIL_ROUNDING_IBM128_LIBGCC),
    TEST_fi_f (scalbn, min_value / 4, INT_MIN, plus_uflow, INEXACT_EXCEPTION|UNDERFLOW_EXCEPTION|ERRNO_PLUS_UFLOW),
    TEST_fi_f (scalbn, min_subnorm_value, INT_MAX, plus_oflow, INEXACT_EXCEPTION|OVERFLOW_EXCEPTION|ERRNO_PLUS_OFLOW|XFAIL_ROUNDING_IBM128_LIBGCC),
    TEST_fi_f (scalbn, min_subnorm_value, INT_MIN, plus_uflow, INEXACT_EXCEPTION|UNDERFLOW_EXCEPTION|ERRNO_PLUS_UFLOW),

    TEST_fi_f (scalbn, -1, INT_MAX, minus_oflow, INEXACT_EXCEPTION|OVERFLOW_EXCEPTION|ERRNO_MINUS_OFLOW|XFAIL_ROUNDING_IBM128_LIBGCC),
    TEST_fi_f (scalbn, -1, INT_MIN, minus_uflow, INEXACT_EXCEPTION|UNDERFLOW_EXCEPTION|ERRNO_MINUS_UFLOW),
    TEST_fi_f (scalbn, -max_value, INT_MAX, minus_oflow, INEXACT_EXCEPTION|OVERFLOW_EXCEPTION|ERRNO_MINUS_OFLOW|XFAIL_ROUNDING_IBM128_LIBGCC),
    TEST_fi_f (scalbn, -max_value, INT_MIN, minus_uflow, INEXACT_EXCEPTION|UNDERFLOW_EXCEPTION|ERRNO_MINUS_UFLOW),
    TEST_fi_f (scalbn, -min_value, INT_MAX, minus_oflow, INEXACT_EXCEPTION|OVERFLOW_EXCEPTION|ERRNO_MINUS_OFLOW|XFAIL_ROUNDING_IBM128_LIBGCC),
    TEST_fi_f (scalbn, -min_value, INT_MIN, minus_uflow, INEXACT_EXCEPTION|UNDERFLOW_EXCEPTION|ERRNO_MINUS_UFLOW),
    TEST_fi_f (scalbn, -min_value / 4, INT_MAX, minus_oflow, INEXACT_EXCEPTION|OVERFLOW_EXCEPTION|ERRNO_MINUS_OFLOW|XFAIL_ROUNDING_IBM128_LIBGCC),
    TEST_fi_f (scalbn, -min_value / 4, INT_MIN, minus_uflow, INEXACT_EXCEPTION|UNDERFLOW_EXCEPTION|ERRNO_MINUS_UFLOW),
    TEST_fi_f (scalbn, -min_subnorm_value, INT_MAX, minus_oflow, INEXACT_EXCEPTION|OVERFLOW_EXCEPTION|ERRNO_MINUS_OFLOW|XFAIL_ROUNDING_IBM128_LIBGCC),
    TEST_fi_f (scalbn, -min_subnorm_value, INT_MIN, minus_uflow, INEXACT_EXCEPTION|UNDERFLOW_EXCEPTION|ERRNO_MINUS_UFLOW),
  };

static void
scalbn_test (void)
{
  ALL_RM_TEST (scalbn, 1, scalbn_test_data, RUN_TEST_LOOP_fi_f, END);
}

static void
ldexp_test (void)
{
  /* ldexp uses the same test data as scalbn.  */
  ALL_RM_TEST (ldexp, 1, scalbn_test_data, RUN_TEST_LOOP_fi_f, END);
}

static void
do_test (void)
{
  scalbn_test ();
  ldexp_test ();
}

/*
 * Local Variables:
 * mode:c
 * End:
 */
