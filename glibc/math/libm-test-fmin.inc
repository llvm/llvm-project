/* Test fmin.
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

static const struct test_ff_f_data fmin_test_data[] =
  {
    TEST_ff_f (fmin, 0, 0, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmin, minus_zero, minus_zero, minus_zero, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmin, 0, minus_zero, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED|IGNORE_ZERO_INF_SIGN),
    TEST_ff_f (fmin, minus_zero, 0, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED|IGNORE_ZERO_INF_SIGN),
    TEST_ff_f (fmin, min_subnorm_value, -min_subnorm_value, -min_subnorm_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmin, -min_subnorm_value, min_subnorm_value, -min_subnorm_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmin, min_value, -min_value, -min_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmin, -min_value, min_value, -min_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmin, max_value, -max_value, -max_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmin, -max_value, max_value, -max_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmin, 9, 0, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmin, 0, 9, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmin, -9, 0, -9, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmin, 0, -9, -9, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),

    TEST_ff_f (fmin, plus_infty, 9, 9, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmin, 9, plus_infty, 9, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmin, plus_infty, -9, -9, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmin, -9, plus_infty, -9, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmin, minus_infty, 9, minus_infty, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmin, minus_infty, -9, minus_infty, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmin, 9, minus_infty, minus_infty, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmin, -9, minus_infty, minus_infty, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),

    TEST_ff_f (fmin, 0, qnan_value, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmin, 0, -qnan_value, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmin, minus_zero, qnan_value, minus_zero, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmin, minus_zero, -qnan_value, minus_zero, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmin, 9, qnan_value, 9, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmin, 9, -qnan_value, 9, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmin, -9, qnan_value, -9, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmin, -9, -qnan_value, -9, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmin, 0, snan_value, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (fmin, 0, -snan_value, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (fmin, minus_zero, snan_value, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (fmin, minus_zero, -snan_value, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (fmin, 9, snan_value, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (fmin, 9, -snan_value, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (fmin, -9, snan_value, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (fmin, -9, -snan_value, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (fmin, qnan_value, 0, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmin, -qnan_value, 0, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmin, qnan_value, minus_zero, minus_zero, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmin, -qnan_value, minus_zero, minus_zero, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmin, qnan_value, 9, 9, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmin, -qnan_value, 9, 9, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmin, qnan_value, -9, -9, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmin, -qnan_value, -9, -9, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmin, snan_value, 0, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (fmin, -snan_value, 0, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (fmin, snan_value, minus_zero, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (fmin, -snan_value, minus_zero, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (fmin, snan_value, 9, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (fmin, -snan_value, 9, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (fmin, snan_value, -9, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (fmin, -snan_value, -9, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (fmin, plus_infty, qnan_value, plus_infty, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmin, plus_infty, -qnan_value, plus_infty, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmin, minus_infty, qnan_value, minus_infty, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmin, minus_infty, -qnan_value, minus_infty, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmin, plus_infty, snan_value, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (fmin, plus_infty, -snan_value, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (fmin, minus_infty, snan_value, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (fmin, minus_infty, -snan_value, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (fmin, qnan_value, plus_infty, plus_infty, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmin, -qnan_value, plus_infty, plus_infty, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmin, qnan_value, minus_infty, minus_infty, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmin, -qnan_value, minus_infty, minus_infty, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmin, snan_value, plus_infty, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (fmin, -snan_value, plus_infty, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (fmin, snan_value, minus_infty, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (fmin, -snan_value, minus_infty, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (fmin, qnan_value, qnan_value, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmin, qnan_value, -qnan_value, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmin, -qnan_value, qnan_value, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmin, -qnan_value, -qnan_value, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmin, qnan_value, snan_value, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (fmin, qnan_value, -snan_value, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (fmin, -qnan_value, snan_value, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (fmin, -qnan_value, -snan_value, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (fmin, snan_value, qnan_value, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (fmin, snan_value, -qnan_value, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (fmin, -snan_value, qnan_value, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (fmin, -snan_value, -qnan_value, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (fmin, snan_value, snan_value, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (fmin, snan_value, -snan_value, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (fmin, -snan_value, snan_value, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (fmin, -snan_value, -snan_value, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
  };

static void
fmin_test (void)
{
  ALL_RM_TEST (fmin, 1, fmin_test_data, RUN_TEST_LOOP_ff_f, END);
}

static void
do_test (void)
{
  fmin_test ();
}

/*
 * Local Variables:
 * mode:c
 * End:
 */
