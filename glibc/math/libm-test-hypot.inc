/* Test hypot.
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

static const struct test_ff_f_data hypot_test_data[] =
  {
    TEST_ff_f (hypot, plus_infty, 1, plus_infty, ERRNO_UNCHANGED),
    TEST_ff_f (hypot, plus_infty, -1, plus_infty, ERRNO_UNCHANGED),
    TEST_ff_f (hypot, plus_infty, 0, plus_infty, ERRNO_UNCHANGED),
    TEST_ff_f (hypot, plus_infty, minus_zero, plus_infty, ERRNO_UNCHANGED),
    TEST_ff_f (hypot, plus_infty, max_value, plus_infty, ERRNO_UNCHANGED|XFAIL_ROUNDING_IBM128_LIBGCC),
    TEST_ff_f (hypot, plus_infty, -max_value, plus_infty, ERRNO_UNCHANGED|XFAIL_ROUNDING_IBM128_LIBGCC),
    TEST_ff_f (hypot, plus_infty, min_value, plus_infty, ERRNO_UNCHANGED),
    TEST_ff_f (hypot, plus_infty, -min_value, plus_infty, ERRNO_UNCHANGED),
    TEST_ff_f (hypot, plus_infty, min_subnorm_value, plus_infty, ERRNO_UNCHANGED),
    TEST_ff_f (hypot, plus_infty, -min_subnorm_value, plus_infty, ERRNO_UNCHANGED),
    TEST_ff_f (hypot, 1, plus_infty, plus_infty, ERRNO_UNCHANGED),
    TEST_ff_f (hypot, -1, plus_infty, plus_infty, ERRNO_UNCHANGED),
    TEST_ff_f (hypot, 0, plus_infty, plus_infty, ERRNO_UNCHANGED),
    TEST_ff_f (hypot, minus_zero, plus_infty, plus_infty, ERRNO_UNCHANGED),
    TEST_ff_f (hypot, max_value, plus_infty, plus_infty, ERRNO_UNCHANGED|XFAIL_ROUNDING_IBM128_LIBGCC),
    TEST_ff_f (hypot, -max_value, plus_infty, plus_infty, ERRNO_UNCHANGED|XFAIL_ROUNDING_IBM128_LIBGCC),
    TEST_ff_f (hypot, min_value, plus_infty, plus_infty, ERRNO_UNCHANGED),
    TEST_ff_f (hypot, -min_value, plus_infty, plus_infty, ERRNO_UNCHANGED),
    TEST_ff_f (hypot, min_subnorm_value, plus_infty, plus_infty, ERRNO_UNCHANGED),
    TEST_ff_f (hypot, -min_subnorm_value, plus_infty, plus_infty, ERRNO_UNCHANGED),
    TEST_ff_f (hypot, minus_infty, 1, plus_infty, ERRNO_UNCHANGED),
    TEST_ff_f (hypot, minus_infty, -1, plus_infty, ERRNO_UNCHANGED),
    TEST_ff_f (hypot, minus_infty, 0, plus_infty, ERRNO_UNCHANGED),
    TEST_ff_f (hypot, minus_infty, minus_zero, plus_infty, ERRNO_UNCHANGED),
    TEST_ff_f (hypot, minus_infty, max_value, plus_infty, ERRNO_UNCHANGED|XFAIL_ROUNDING_IBM128_LIBGCC),
    TEST_ff_f (hypot, minus_infty, -max_value, plus_infty, ERRNO_UNCHANGED|XFAIL_ROUNDING_IBM128_LIBGCC),
    TEST_ff_f (hypot, minus_infty, min_value, plus_infty, ERRNO_UNCHANGED),
    TEST_ff_f (hypot, minus_infty, -min_value, plus_infty, ERRNO_UNCHANGED),
    TEST_ff_f (hypot, minus_infty, min_subnorm_value, plus_infty, ERRNO_UNCHANGED),
    TEST_ff_f (hypot, minus_infty, -min_subnorm_value, plus_infty, ERRNO_UNCHANGED),
    TEST_ff_f (hypot, 1, minus_infty, plus_infty, ERRNO_UNCHANGED),
    TEST_ff_f (hypot, -1, minus_infty, plus_infty, ERRNO_UNCHANGED),
    TEST_ff_f (hypot, 0, minus_infty, plus_infty, ERRNO_UNCHANGED),
    TEST_ff_f (hypot, minus_zero, minus_infty, plus_infty, ERRNO_UNCHANGED),
    TEST_ff_f (hypot, max_value, minus_infty, plus_infty, ERRNO_UNCHANGED|XFAIL_ROUNDING_IBM128_LIBGCC),
    TEST_ff_f (hypot, -max_value, minus_infty, plus_infty, ERRNO_UNCHANGED|XFAIL_ROUNDING_IBM128_LIBGCC),
    TEST_ff_f (hypot, min_value, minus_infty, plus_infty, ERRNO_UNCHANGED),
    TEST_ff_f (hypot, -min_value, minus_infty, plus_infty, ERRNO_UNCHANGED),
    TEST_ff_f (hypot, min_subnorm_value, minus_infty, plus_infty, ERRNO_UNCHANGED),
    TEST_ff_f (hypot, -min_subnorm_value, minus_infty, plus_infty, ERRNO_UNCHANGED),
    TEST_ff_f (hypot, plus_infty, plus_infty, plus_infty, ERRNO_UNCHANGED),
    TEST_ff_f (hypot, plus_infty, minus_infty, plus_infty, ERRNO_UNCHANGED),
    TEST_ff_f (hypot, minus_infty, plus_infty, plus_infty, ERRNO_UNCHANGED),
    TEST_ff_f (hypot, minus_infty, minus_infty, plus_infty, ERRNO_UNCHANGED),

    TEST_ff_f (hypot, plus_infty, qnan_value, plus_infty, ERRNO_UNCHANGED),
    TEST_ff_f (hypot, plus_infty, -qnan_value, plus_infty, ERRNO_UNCHANGED),
    TEST_ff_f (hypot, minus_infty, qnan_value, plus_infty, ERRNO_UNCHANGED),
    TEST_ff_f (hypot, minus_infty, -qnan_value, plus_infty, ERRNO_UNCHANGED),
    TEST_ff_f (hypot, qnan_value, plus_infty, plus_infty, ERRNO_UNCHANGED),
    TEST_ff_f (hypot, -qnan_value, plus_infty, plus_infty, ERRNO_UNCHANGED),
    TEST_ff_f (hypot, qnan_value, minus_infty, plus_infty, ERRNO_UNCHANGED),
    TEST_ff_f (hypot, -qnan_value, minus_infty, plus_infty, ERRNO_UNCHANGED),
    TEST_ff_f (hypot, plus_infty, snan_value, qnan_value, INVALID_EXCEPTION),
    TEST_ff_f (hypot, plus_infty, -snan_value, qnan_value, INVALID_EXCEPTION),
    TEST_ff_f (hypot, minus_infty, snan_value, qnan_value, INVALID_EXCEPTION),
    TEST_ff_f (hypot, minus_infty, -snan_value, qnan_value, INVALID_EXCEPTION),
    TEST_ff_f (hypot, snan_value, plus_infty, qnan_value, INVALID_EXCEPTION),
    TEST_ff_f (hypot, -snan_value, plus_infty, qnan_value, INVALID_EXCEPTION),
    TEST_ff_f (hypot, snan_value, minus_infty, qnan_value, INVALID_EXCEPTION),
    TEST_ff_f (hypot, -snan_value, minus_infty, qnan_value, INVALID_EXCEPTION),

    TEST_ff_f (hypot, 0, qnan_value, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (hypot, 0, -qnan_value, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (hypot, minus_zero, qnan_value, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (hypot, minus_zero, -qnan_value, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (hypot, max_value, qnan_value, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (hypot, max_value, -qnan_value, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (hypot, -max_value, qnan_value, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (hypot, -max_value, -qnan_value, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (hypot, min_value, qnan_value, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (hypot, min_value, -qnan_value, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (hypot, -min_value, qnan_value, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (hypot, -min_value, -qnan_value, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (hypot, min_subnorm_value, qnan_value, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (hypot, min_subnorm_value, -qnan_value, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (hypot, -min_subnorm_value, qnan_value, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (hypot, -min_subnorm_value, -qnan_value, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (hypot, 0, snan_value, qnan_value, INVALID_EXCEPTION),
    TEST_ff_f (hypot, 0, -snan_value, qnan_value, INVALID_EXCEPTION),
    TEST_ff_f (hypot, minus_zero, snan_value, qnan_value, INVALID_EXCEPTION),
    TEST_ff_f (hypot, minus_zero, -snan_value, qnan_value, INVALID_EXCEPTION),
    TEST_ff_f (hypot, max_value, snan_value, qnan_value, INVALID_EXCEPTION),
    TEST_ff_f (hypot, max_value, -snan_value, qnan_value, INVALID_EXCEPTION),
    TEST_ff_f (hypot, -max_value, snan_value, qnan_value, INVALID_EXCEPTION),
    TEST_ff_f (hypot, -max_value, -snan_value, qnan_value, INVALID_EXCEPTION),
    TEST_ff_f (hypot, min_value, snan_value, qnan_value, INVALID_EXCEPTION),
    TEST_ff_f (hypot, min_value, -snan_value, qnan_value, INVALID_EXCEPTION),
    TEST_ff_f (hypot, -min_value, snan_value, qnan_value, INVALID_EXCEPTION),
    TEST_ff_f (hypot, -min_value, -snan_value, qnan_value, INVALID_EXCEPTION),
    TEST_ff_f (hypot, min_subnorm_value, snan_value, qnan_value, INVALID_EXCEPTION),
    TEST_ff_f (hypot, min_subnorm_value, -snan_value, qnan_value, INVALID_EXCEPTION),
    TEST_ff_f (hypot, -min_subnorm_value, snan_value, qnan_value, INVALID_EXCEPTION),
    TEST_ff_f (hypot, -min_subnorm_value, -snan_value, qnan_value, INVALID_EXCEPTION),
    TEST_ff_f (hypot, qnan_value, 0, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (hypot, -qnan_value, 0, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (hypot, qnan_value, minus_zero, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (hypot, -qnan_value, minus_zero, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (hypot, qnan_value, max_value, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (hypot, -qnan_value, max_value, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (hypot, qnan_value, -max_value, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (hypot, -qnan_value, -max_value, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (hypot, qnan_value, min_value, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (hypot, -qnan_value, min_value, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (hypot, qnan_value, -min_value, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (hypot, -qnan_value, -min_value, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (hypot, qnan_value, min_subnorm_value, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (hypot, -qnan_value, min_subnorm_value, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (hypot, qnan_value, -min_subnorm_value, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (hypot, -qnan_value, -min_subnorm_value, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (hypot, snan_value, 0, qnan_value, INVALID_EXCEPTION),
    TEST_ff_f (hypot, -snan_value, 0, qnan_value, INVALID_EXCEPTION),
    TEST_ff_f (hypot, snan_value, minus_zero, qnan_value, INVALID_EXCEPTION),
    TEST_ff_f (hypot, -snan_value, minus_zero, qnan_value, INVALID_EXCEPTION),
    TEST_ff_f (hypot, snan_value, max_value, qnan_value, INVALID_EXCEPTION),
    TEST_ff_f (hypot, -snan_value, max_value, qnan_value, INVALID_EXCEPTION),
    TEST_ff_f (hypot, snan_value, -max_value, qnan_value, INVALID_EXCEPTION),
    TEST_ff_f (hypot, -snan_value, -max_value, qnan_value, INVALID_EXCEPTION),
    TEST_ff_f (hypot, snan_value, min_value, qnan_value, INVALID_EXCEPTION),
    TEST_ff_f (hypot, -snan_value, min_value, qnan_value, INVALID_EXCEPTION),
    TEST_ff_f (hypot, snan_value, -min_value, qnan_value, INVALID_EXCEPTION),
    TEST_ff_f (hypot, -snan_value, -min_value, qnan_value, INVALID_EXCEPTION),
    TEST_ff_f (hypot, snan_value, min_subnorm_value, qnan_value, INVALID_EXCEPTION),
    TEST_ff_f (hypot, -snan_value, min_subnorm_value, qnan_value, INVALID_EXCEPTION),
    TEST_ff_f (hypot, snan_value, -min_subnorm_value, qnan_value, INVALID_EXCEPTION),
    TEST_ff_f (hypot, -snan_value, -min_subnorm_value, qnan_value, INVALID_EXCEPTION),

    TEST_ff_f (hypot, qnan_value, qnan_value, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (hypot, qnan_value, -qnan_value, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (hypot, -qnan_value, qnan_value, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (hypot, -qnan_value, -qnan_value, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (hypot, qnan_value, snan_value, qnan_value, INVALID_EXCEPTION),
    TEST_ff_f (hypot, qnan_value, -snan_value, qnan_value, INVALID_EXCEPTION),
    TEST_ff_f (hypot, -qnan_value, snan_value, qnan_value, INVALID_EXCEPTION),
    TEST_ff_f (hypot, -qnan_value, -snan_value, qnan_value, INVALID_EXCEPTION),
    TEST_ff_f (hypot, snan_value, qnan_value, qnan_value, INVALID_EXCEPTION),
    TEST_ff_f (hypot, snan_value, -qnan_value, qnan_value, INVALID_EXCEPTION),
    TEST_ff_f (hypot, -snan_value, qnan_value, qnan_value, INVALID_EXCEPTION),
    TEST_ff_f (hypot, -snan_value, -qnan_value, qnan_value, INVALID_EXCEPTION),
    TEST_ff_f (hypot, snan_value, snan_value, qnan_value, INVALID_EXCEPTION),
    TEST_ff_f (hypot, snan_value, -snan_value, qnan_value, INVALID_EXCEPTION),
    TEST_ff_f (hypot, -snan_value, snan_value, qnan_value, INVALID_EXCEPTION),
    TEST_ff_f (hypot, -snan_value, -snan_value, qnan_value, INVALID_EXCEPTION),

    AUTO_TESTS_ff_f (hypot),
  };

static void
hypot_test (void)
{
  ALL_RM_TEST (hypot, 0, hypot_test_data, RUN_TEST_LOOP_ff_f, END);
}

static void
do_test (void)
{
  hypot_test ();
}

/*
 * Local Variables:
 * mode:c
 * End:
 */
