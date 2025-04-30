/* Test remainder.
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

static const struct test_ff_f_data remainder_test_data[] =
  {
    TEST_ff_f (remainder, 1, 0, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION|ERRNO_EDOM),
    TEST_ff_f (remainder, 1, minus_zero, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION|ERRNO_EDOM),
    TEST_ff_f (remainder, -1.1L, 0, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION|ERRNO_EDOM),
    TEST_ff_f (remainder, -1.1L, minus_zero, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION|ERRNO_EDOM),
    TEST_ff_f (remainder, plus_zero, plus_zero, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION|ERRNO_EDOM),
    TEST_ff_f (remainder, plus_zero, minus_zero, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION|ERRNO_EDOM),
    TEST_ff_f (remainder, minus_zero, plus_zero, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION|ERRNO_EDOM),
    TEST_ff_f (remainder, minus_zero, minus_zero, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION|ERRNO_EDOM),
    TEST_ff_f (remainder, plus_infty, minus_zero, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION|ERRNO_EDOM),
    TEST_ff_f (remainder, plus_infty, 0, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION|ERRNO_EDOM),
    TEST_ff_f (remainder, plus_infty, 1, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION|ERRNO_EDOM),
    TEST_ff_f (remainder, plus_infty, 1.1L, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION|ERRNO_EDOM),
    TEST_ff_f (remainder, plus_infty, 2, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION|ERRNO_EDOM),
    TEST_ff_f (remainder, plus_infty, plus_infty, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION|ERRNO_EDOM),
    TEST_ff_f (remainder, plus_infty, minus_infty, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION|ERRNO_EDOM),
    TEST_ff_f (remainder, minus_infty, minus_zero, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION|ERRNO_EDOM),
    TEST_ff_f (remainder, minus_infty, 0, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION|ERRNO_EDOM),
    TEST_ff_f (remainder, minus_infty, 1, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION|ERRNO_EDOM),
    TEST_ff_f (remainder, minus_infty, 1.1L, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION|ERRNO_EDOM),
    TEST_ff_f (remainder, minus_infty, 2, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION|ERRNO_EDOM),
    TEST_ff_f (remainder, minus_infty, plus_infty, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION|ERRNO_EDOM),
    TEST_ff_f (remainder, minus_infty, minus_infty, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION|ERRNO_EDOM),
    TEST_ff_f (remainder, qnan_value, qnan_value, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (remainder, qnan_value, -qnan_value, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (remainder, -qnan_value, qnan_value, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (remainder, -qnan_value, -qnan_value, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (remainder, snan_value, qnan_value, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (remainder, snan_value, -qnan_value, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (remainder, -snan_value, qnan_value, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (remainder, -snan_value, -qnan_value, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (remainder, qnan_value, snan_value, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (remainder, qnan_value, -snan_value, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (remainder, -qnan_value, snan_value, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (remainder, -qnan_value, -snan_value, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (remainder, snan_value, snan_value, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (remainder, snan_value, -snan_value, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (remainder, -snan_value, snan_value, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (remainder, -snan_value, -snan_value, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (remainder, 0, qnan_value, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (remainder, 0, -qnan_value, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (remainder, 0, snan_value, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (remainder, 0, -snan_value, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (remainder, qnan_value, 0, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (remainder, -qnan_value, 0, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (remainder, snan_value, 0, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (remainder, -snan_value, 0, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (remainder, 1, qnan_value, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (remainder, 1, -qnan_value, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (remainder, 1, snan_value, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (remainder, 1, -snan_value, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (remainder, qnan_value, 1, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (remainder, -qnan_value, 1, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (remainder, snan_value, 1, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (remainder, -snan_value, 1, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (remainder, plus_infty, qnan_value, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (remainder, plus_infty, -qnan_value, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (remainder, plus_infty, snan_value, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (remainder, plus_infty, -snan_value, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (remainder, qnan_value, plus_infty, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (remainder, -qnan_value, plus_infty, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (remainder, snan_value, plus_infty, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (remainder, -snan_value, plus_infty, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (remainder, minus_infty, qnan_value, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (remainder, minus_infty, -qnan_value, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (remainder, minus_infty, snan_value, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (remainder, minus_infty, -snan_value, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (remainder, qnan_value, minus_infty, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (remainder, -qnan_value, minus_infty, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (remainder, snan_value, minus_infty, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (remainder, -snan_value, minus_infty, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),

    TEST_ff_f (remainder, 7.0, plus_infty, 7.0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (remainder, 7.0, minus_infty, 7.0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),

    TEST_ff_f (remainder, 1.625, 1.0, -0.375, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (remainder, -1.625, 1.0, 0.375, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (remainder, 1.625, -1.0, -0.375, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (remainder, -1.625, -1.0, 0.375, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (remainder, 5.0, 2.0, 1.0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (remainder, 3.0, 2.0, -1.0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
#if MANT_DIG >= 56
    TEST_ff_f (remainder, -0x1.80000000000002p1L, 2.0, 0x1.fffffffffffff8p-1L, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (remainder, 0x0.ffffffffffffffp0L, -0x1.00000000000001p0L, -0x1p-55L, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
#endif

    TEST_ff_f (remainder, 3419, 360, 179, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (remainder, -3419, 360, -179, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (remainder, 3419, -360, 179, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (remainder, -3419, -360, -179, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),

    TEST_ff_f (remainder, max_value, max_value, plus_zero, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (remainder, max_value, -max_value, plus_zero, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (remainder, max_value, min_value, plus_zero, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (remainder, max_value, -min_value, plus_zero, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (remainder, max_value, min_subnorm_value, plus_zero, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (remainder, max_value, -min_subnorm_value, plus_zero, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (remainder, -max_value, max_value, minus_zero, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (remainder, -max_value, -max_value, minus_zero, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (remainder, -max_value, min_value, minus_zero, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (remainder, -max_value, -min_value, minus_zero, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (remainder, -max_value, min_subnorm_value, minus_zero, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (remainder, -max_value, -min_subnorm_value, minus_zero, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (remainder, min_value, max_value, min_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (remainder, min_value, -max_value, min_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (remainder, min_value, min_value, plus_zero, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (remainder, min_value, -min_value, plus_zero, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (remainder, min_value, min_subnorm_value, plus_zero, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (remainder, min_value, -min_subnorm_value, plus_zero, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (remainder, -min_value, max_value, -min_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (remainder, -min_value, -max_value, -min_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (remainder, -min_value, min_value, minus_zero, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (remainder, -min_value, -min_value, minus_zero, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (remainder, -min_value, min_subnorm_value, minus_zero, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (remainder, -min_value, -min_subnorm_value, minus_zero, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (remainder, min_subnorm_value, max_value, min_subnorm_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (remainder, min_subnorm_value, -max_value, min_subnorm_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (remainder, min_subnorm_value, min_value, min_subnorm_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (remainder, min_subnorm_value, -min_value, min_subnorm_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (remainder, min_subnorm_value, min_subnorm_value, plus_zero, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (remainder, min_subnorm_value, -min_subnorm_value, plus_zero, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (remainder, -min_subnorm_value, max_value, -min_subnorm_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (remainder, -min_subnorm_value, -max_value, -min_subnorm_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (remainder, -min_subnorm_value, min_value, -min_subnorm_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (remainder, -min_subnorm_value, -min_value, -min_subnorm_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (remainder, -min_subnorm_value, min_subnorm_value, minus_zero, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (remainder, -min_subnorm_value, -min_subnorm_value, minus_zero, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),

    TEST_ff_f (remainder, 1, max_value, 1, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (remainder, 1, -max_value, 1, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (remainder, 1, max_value / 2, 1, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED|XFAIL_ROUNDING_IBM128_LIBGCC),
    TEST_ff_f (remainder, 1, -max_value / 2, 1, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED|XFAIL_ROUNDING_IBM128_LIBGCC),
    TEST_ff_f (remainder, 1, max_value / 4, 1, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (remainder, 1, -max_value / 4, 1, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (remainder, 1, max_value / 8, 1, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (remainder, 1, -max_value / 8, 1, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (remainder, -1, max_value, -1, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (remainder, -1, -max_value, -1, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (remainder, -1, max_value / 2, -1, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED|XFAIL_ROUNDING_IBM128_LIBGCC),
    TEST_ff_f (remainder, -1, -max_value / 2, -1, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED|XFAIL_ROUNDING_IBM128_LIBGCC),
    TEST_ff_f (remainder, -1, max_value / 4, -1, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (remainder, -1, -max_value / 4, -1, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (remainder, -1, max_value / 8, -1, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (remainder, -1, -max_value / 8, -1, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),

    TEST_ff_f (remainder, max_value, max_value / 2, plus_zero, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED|XFAIL_ROUNDING_IBM128_LIBGCC),
    TEST_ff_f (remainder, max_value, -max_value / 2, plus_zero, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED|XFAIL_ROUNDING_IBM128_LIBGCC),
    TEST_ff_f (remainder, -max_value, max_value / 2, minus_zero, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED|XFAIL_ROUNDING_IBM128_LIBGCC),
    TEST_ff_f (remainder, -max_value, -max_value / 2, minus_zero, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED|XFAIL_ROUNDING_IBM128_LIBGCC),

    TEST_ff_f (remainder, 2, 1, plus_zero, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (remainder, 2, -1, plus_zero, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (remainder, -2, 1, minus_zero, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (remainder, -2, -1, minus_zero, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
  };

static void
remainder_test (void)
{
  ALL_RM_TEST (remainder, 1, remainder_test_data, RUN_TEST_LOOP_ff_f, END);
}

static void
drem_test (void)
{
#if !TEST_FLOATN
  /* drem uses the same test data as remainder.  */
  ALL_RM_TEST (drem, 1, remainder_test_data, RUN_TEST_LOOP_ff_f, END);
#endif
}

static void
do_test (void)
{
  remainder_test ();
  drem_test ();
}

/*
 * Local Variables:
 * mode:c
 * End:
 */
