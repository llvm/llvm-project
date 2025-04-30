/* Test fmaxmag.
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

static const struct test_ff_f_data fmaxmag_test_data[] =
  {
    TEST_ff_f (fmaxmag, 0, 0, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmaxmag, minus_zero, minus_zero, minus_zero, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmaxmag, 0, minus_zero, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED|IGNORE_ZERO_INF_SIGN),
    TEST_ff_f (fmaxmag, minus_zero, 0, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED|IGNORE_ZERO_INF_SIGN),
    TEST_ff_f (fmaxmag, min_subnorm_value, min_subnorm_value, min_subnorm_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmaxmag, min_subnorm_value, -min_subnorm_value, min_subnorm_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmaxmag, -min_subnorm_value, min_subnorm_value, min_subnorm_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmaxmag, -min_subnorm_value, -min_subnorm_value, -min_subnorm_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmaxmag, min_value, min_value, min_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmaxmag, min_value, -min_value, min_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmaxmag, -min_value, min_value, min_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmaxmag, -min_value, -min_value, -min_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmaxmag, max_value, max_value, max_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmaxmag, max_value, -max_value, max_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmaxmag, -max_value, max_value, max_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmaxmag, -max_value, -max_value, -max_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmaxmag, max_value, min_value, max_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmaxmag, max_value, -min_value, max_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmaxmag, -max_value, min_value, -max_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmaxmag, -max_value, -min_value, -max_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmaxmag, min_value, max_value, max_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmaxmag, min_value, -max_value, -max_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmaxmag, -min_value, max_value, max_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmaxmag, -min_value, -max_value, -max_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmaxmag, 9, 0, 9, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmaxmag, 0, 9, 9, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmaxmag, -9, 0, -9, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmaxmag, 0, -9, -9, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),

    TEST_ff_f (fmaxmag, plus_infty, 9, plus_infty, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmaxmag, 0, plus_infty, plus_infty, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmaxmag, -9, plus_infty, plus_infty, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmaxmag, plus_infty, -9, plus_infty, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),

    TEST_ff_f (fmaxmag, minus_infty, 9, minus_infty, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmaxmag, minus_infty, -9, minus_infty, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmaxmag, 9, minus_infty, minus_infty, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmaxmag, -9, minus_infty, minus_infty, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),

    TEST_ff_f (fmaxmag, 0, qnan_value, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmaxmag, 0, -qnan_value, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmaxmag, minus_zero, qnan_value, minus_zero, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmaxmag, minus_zero, -qnan_value, minus_zero, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmaxmag, 9, qnan_value, 9, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmaxmag, 9, -qnan_value, 9, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmaxmag, -9, qnan_value, -9, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmaxmag, -9, -qnan_value, -9, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmaxmag, 0, snan_value, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (fmaxmag, 0, -snan_value, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (fmaxmag, minus_zero, snan_value, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (fmaxmag, minus_zero, -snan_value, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (fmaxmag, 9, snan_value, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (fmaxmag, 9, -snan_value, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (fmaxmag, -9, snan_value, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (fmaxmag, -9, -snan_value, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (fmaxmag, qnan_value, 0, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmaxmag, -qnan_value, 0, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmaxmag, qnan_value, minus_zero, minus_zero, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmaxmag, -qnan_value, minus_zero, minus_zero, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmaxmag, qnan_value, 9, 9, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmaxmag, -qnan_value, 9, 9, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmaxmag, qnan_value, -9, -9, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmaxmag, -qnan_value, -9, -9, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmaxmag, snan_value, 0, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (fmaxmag, -snan_value, 0, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (fmaxmag, snan_value, minus_zero, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (fmaxmag, -snan_value, minus_zero, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (fmaxmag, snan_value, 9, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (fmaxmag, -snan_value, 9, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (fmaxmag, snan_value, -9, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (fmaxmag, -snan_value, -9, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (fmaxmag, plus_infty, qnan_value, plus_infty, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmaxmag, plus_infty, -qnan_value, plus_infty, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmaxmag, minus_infty, qnan_value, minus_infty, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmaxmag, minus_infty, -qnan_value, minus_infty, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmaxmag, plus_infty, snan_value, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (fmaxmag, plus_infty, -snan_value, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (fmaxmag, minus_infty, snan_value, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (fmaxmag, minus_infty, -snan_value, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (fmaxmag, qnan_value, plus_infty, plus_infty, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmaxmag, -qnan_value, plus_infty, plus_infty, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmaxmag, qnan_value, minus_infty, minus_infty, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmaxmag, -qnan_value, minus_infty, minus_infty, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmaxmag, snan_value, plus_infty, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (fmaxmag, -snan_value, plus_infty, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (fmaxmag, snan_value, minus_infty, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (fmaxmag, -snan_value, minus_infty, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (fmaxmag, qnan_value, qnan_value, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmaxmag, qnan_value, -qnan_value, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmaxmag, -qnan_value, qnan_value, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmaxmag, -qnan_value, -qnan_value, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmaxmag, qnan_value, snan_value, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (fmaxmag, qnan_value, -snan_value, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (fmaxmag, -qnan_value, snan_value, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (fmaxmag, -qnan_value, -snan_value, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (fmaxmag, snan_value, qnan_value, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (fmaxmag, snan_value, -qnan_value, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (fmaxmag, -snan_value, qnan_value, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (fmaxmag, -snan_value, -qnan_value, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (fmaxmag, snan_value, snan_value, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (fmaxmag, snan_value, -snan_value, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (fmaxmag, -snan_value, snan_value, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (fmaxmag, -snan_value, -snan_value, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
  };

static void
fmaxmag_test (void)
{
  ALL_RM_TEST (fmaxmag, 1, fmaxmag_test_data, RUN_TEST_LOOP_ff_f, END);
}

static void
do_test (void)
{
  fmaxmag_test ();
}

/*
 * Local Variables:
 * mode:c
 * End:
 */
