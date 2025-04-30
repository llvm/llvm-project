/* Test significand.
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

static const struct test_f_f_data significand_test_data[] =
  {
#if !TEST_FLOATN
    /* significand returns the mantissa of the exponential
       representation.  Tests for infinities, zero and NaNs reflect
       the implementation (including possibility of "inexact" and
       variations between architectures) rather than any
       specification.  */
    TEST_f_f (significand, qnan_value, qnan_value, INVALID_EXCEPTION_OK),
    TEST_f_f (significand, -qnan_value, qnan_value, INVALID_EXCEPTION_OK),
    TEST_f_f (significand, snan_value, qnan_value, INVALID_EXCEPTION),
    TEST_f_f (significand, -snan_value, qnan_value, INVALID_EXCEPTION),
    TEST_f_f (significand, plus_infty, plus_infty, INVALID_EXCEPTION_OK),
    TEST_f_f (significand, minus_infty, minus_infty, INVALID_EXCEPTION_OK),
    TEST_f_f (significand, 0, 0, INVALID_EXCEPTION_OK|DIVIDE_BY_ZERO_EXCEPTION_OK),
    TEST_f_f (significand, minus_zero, minus_zero, INVALID_EXCEPTION_OK|DIVIDE_BY_ZERO_EXCEPTION_OK),
    TEST_f_f (significand, min_value, 1.0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_f_f (significand, -min_value, -1.0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_f_f (significand, min_subnorm_value, 1.0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_f_f (significand, -min_subnorm_value, -1.0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_f_f (significand, 1.0, 1.0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_f_f (significand, -1.0, -1.0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_f_f (significand, 4.0, 1.0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_f_f (significand, -4.0, -1.0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_f_f (significand, 6.0, 1.5, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_f_f (significand, -6.0, -1.5, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_f_f (significand, 8.0, 1.0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_f_f (significand, -8.0, -1.0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
#endif
  };

static void
significand_test (void)
{
#if !TEST_FLOATN
  ALL_RM_TEST (significand, 1, significand_test_data, RUN_TEST_LOOP_f_f, END);
#endif
}

static void
do_test (void)
{
  significand_test ();
}

/*
 * Local Variables:
 * mode:c
 * End:
 */
