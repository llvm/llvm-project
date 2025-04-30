/* Test yn.
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

static const struct test_if_f_data yn_test_data[] =
  {
    /* yn is the Bessel function of the second kind of order n */
    /* yn (0, x) == y0 (x)  */
    TEST_if_f (yn, 0, -1.0, qnan_value, INVALID_EXCEPTION|ERRNO_EDOM),
    TEST_if_f (yn, 0, -min_value, qnan_value, INVALID_EXCEPTION|ERRNO_EDOM),
    TEST_if_f (yn, 0, -min_subnorm_value, qnan_value, INVALID_EXCEPTION|ERRNO_EDOM),
    TEST_if_f (yn, 0, -max_value, qnan_value, INVALID_EXCEPTION|ERRNO_EDOM),
    TEST_if_f (yn, 0, minus_infty, qnan_value, INVALID_EXCEPTION|ERRNO_EDOM),
    TEST_if_f (yn, 0, 0.0, minus_infty, DIVIDE_BY_ZERO_EXCEPTION|ERRNO_ERANGE),
    TEST_if_f (yn, 0, -0.0, minus_infty, DIVIDE_BY_ZERO_EXCEPTION|ERRNO_ERANGE),
    TEST_if_f (yn, 0, qnan_value, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_if_f (yn, 0, -qnan_value, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_if_f (yn, 0, snan_value, qnan_value, INVALID_EXCEPTION),
    TEST_if_f (yn, 0, -snan_value, qnan_value, INVALID_EXCEPTION),
    TEST_if_f (yn, 0, plus_infty, 0, ERRNO_UNCHANGED|IGNORE_ZERO_INF_SIGN),

    /* yn (1, x) == y1 (x)  */
    TEST_if_f (yn, 1, -1.0, qnan_value, INVALID_EXCEPTION|ERRNO_EDOM),
    TEST_if_f (yn, 1, -min_value, qnan_value, INVALID_EXCEPTION|ERRNO_EDOM),
    TEST_if_f (yn, 1, -min_subnorm_value, qnan_value, INVALID_EXCEPTION|ERRNO_EDOM),
    TEST_if_f (yn, 1, -max_value, qnan_value, INVALID_EXCEPTION|ERRNO_EDOM),
    TEST_if_f (yn, 1, minus_infty, qnan_value, INVALID_EXCEPTION|ERRNO_EDOM),
    TEST_if_f (yn, 1, 0.0, minus_infty, DIVIDE_BY_ZERO_EXCEPTION|ERRNO_ERANGE),
    TEST_if_f (yn, 1, -0.0, minus_infty, DIVIDE_BY_ZERO_EXCEPTION|ERRNO_ERANGE),
    TEST_if_f (yn, 1, plus_infty, 0, ERRNO_UNCHANGED|IGNORE_ZERO_INF_SIGN),
    TEST_if_f (yn, 1, qnan_value, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_if_f (yn, 1, -qnan_value, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_if_f (yn, 1, snan_value, qnan_value, INVALID_EXCEPTION),
    TEST_if_f (yn, 1, -snan_value, qnan_value, INVALID_EXCEPTION),

    /* yn (-1, x) == -y1 (x).  */
    TEST_if_f (yn, -1, -1.0, qnan_value, INVALID_EXCEPTION|ERRNO_EDOM),
    TEST_if_f (yn, -1, -min_value, qnan_value, INVALID_EXCEPTION|ERRNO_EDOM),
    TEST_if_f (yn, -1, -min_subnorm_value, qnan_value, INVALID_EXCEPTION|ERRNO_EDOM),
    TEST_if_f (yn, -1, -max_value, qnan_value, INVALID_EXCEPTION|ERRNO_EDOM),
    TEST_if_f (yn, -1, minus_infty, qnan_value, INVALID_EXCEPTION|ERRNO_EDOM),
    TEST_if_f (yn, -1, 0.0, plus_infty, DIVIDE_BY_ZERO_EXCEPTION|ERRNO_ERANGE),
    TEST_if_f (yn, -1, -0.0, plus_infty, DIVIDE_BY_ZERO_EXCEPTION|ERRNO_ERANGE),
    TEST_if_f (yn, -1, plus_infty, minus_zero, ERRNO_UNCHANGED|IGNORE_ZERO_INF_SIGN),
    TEST_if_f (yn, -1, qnan_value, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_if_f (yn, -1, -qnan_value, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_if_f (yn, -1, snan_value, qnan_value, INVALID_EXCEPTION),
    TEST_if_f (yn, -1, -snan_value, qnan_value, INVALID_EXCEPTION),

    /* yn (2, x).  */
    TEST_if_f (yn, 2, -1.0, qnan_value, INVALID_EXCEPTION|ERRNO_EDOM),
    TEST_if_f (yn, 2, -min_value, qnan_value, INVALID_EXCEPTION|ERRNO_EDOM),
    TEST_if_f (yn, 2, -min_subnorm_value, qnan_value, INVALID_EXCEPTION|ERRNO_EDOM),
    TEST_if_f (yn, 2, -max_value, qnan_value, INVALID_EXCEPTION|ERRNO_EDOM),
    TEST_if_f (yn, 2, minus_infty, qnan_value, INVALID_EXCEPTION|ERRNO_EDOM),
    TEST_if_f (yn, 2, 0.0, minus_infty, DIVIDE_BY_ZERO_EXCEPTION|ERRNO_ERANGE),
    TEST_if_f (yn, 2, -0.0, minus_infty, DIVIDE_BY_ZERO_EXCEPTION|ERRNO_ERANGE),
    TEST_if_f (yn, 2, plus_infty, 0, ERRNO_UNCHANGED|IGNORE_ZERO_INF_SIGN),
    TEST_if_f (yn, 2, qnan_value, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_if_f (yn, 2, -qnan_value, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_if_f (yn, 2, snan_value, qnan_value, INVALID_EXCEPTION),
    TEST_if_f (yn, 2, -snan_value, qnan_value, INVALID_EXCEPTION),

    /* yn (-2, x) == yn (2, x).  */
    TEST_if_f (yn, -2, -1.0, qnan_value, INVALID_EXCEPTION|ERRNO_EDOM),
    TEST_if_f (yn, -2, -min_value, qnan_value, INVALID_EXCEPTION|ERRNO_EDOM),
    TEST_if_f (yn, -2, -min_subnorm_value, qnan_value, INVALID_EXCEPTION|ERRNO_EDOM),
    TEST_if_f (yn, -2, -max_value, qnan_value, INVALID_EXCEPTION|ERRNO_EDOM),
    TEST_if_f (yn, -2, minus_infty, qnan_value, INVALID_EXCEPTION|ERRNO_EDOM),
    TEST_if_f (yn, -2, 0.0, minus_infty, DIVIDE_BY_ZERO_EXCEPTION|ERRNO_ERANGE),
    TEST_if_f (yn, -2, -0.0, minus_infty, DIVIDE_BY_ZERO_EXCEPTION|ERRNO_ERANGE),
    TEST_if_f (yn, -2, plus_infty, 0, ERRNO_UNCHANGED|IGNORE_ZERO_INF_SIGN),
    TEST_if_f (yn, -2, qnan_value, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_if_f (yn, -2, -qnan_value, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_if_f (yn, -2, snan_value, qnan_value, INVALID_EXCEPTION),
    TEST_if_f (yn, -2, -snan_value, qnan_value, INVALID_EXCEPTION),

    /* yn (3, x)  */
    TEST_if_f (yn, 3, -1.0, qnan_value, INVALID_EXCEPTION|ERRNO_EDOM),
    TEST_if_f (yn, 3, -min_value, qnan_value, INVALID_EXCEPTION|ERRNO_EDOM),
    TEST_if_f (yn, 3, -min_subnorm_value, qnan_value, INVALID_EXCEPTION|ERRNO_EDOM),
    TEST_if_f (yn, 3, -max_value, qnan_value, INVALID_EXCEPTION|ERRNO_EDOM),
    TEST_if_f (yn, 3, minus_infty, qnan_value, INVALID_EXCEPTION|ERRNO_EDOM),
    TEST_if_f (yn, 3, 0.0, minus_infty, DIVIDE_BY_ZERO_EXCEPTION|ERRNO_ERANGE),
    TEST_if_f (yn, 3, -0.0, minus_infty, DIVIDE_BY_ZERO_EXCEPTION|ERRNO_ERANGE),
    TEST_if_f (yn, 3, plus_infty, 0, ERRNO_UNCHANGED|IGNORE_ZERO_INF_SIGN),
    TEST_if_f (yn, 3, qnan_value, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_if_f (yn, 3, -qnan_value, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_if_f (yn, 3, snan_value, qnan_value, INVALID_EXCEPTION),
    TEST_if_f (yn, 3, -snan_value, qnan_value, INVALID_EXCEPTION),

    /* yn (10, x)  */
    TEST_if_f (yn, 10, plus_infty, 0, ERRNO_UNCHANGED|IGNORE_ZERO_INF_SIGN),
    TEST_if_f (yn, 10, qnan_value, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_if_f (yn, 10, -qnan_value, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_if_f (yn, 10, snan_value, qnan_value, INVALID_EXCEPTION),
    TEST_if_f (yn, 10, -snan_value, qnan_value, INVALID_EXCEPTION),

    AUTO_TESTS_if_f (yn),
  };

static void
yn_test (void)
{
  ALL_RM_TEST (yn, 0, yn_test_data, RUN_TEST_LOOP_if_f, END);
}

static void
do_test (void)
{
  yn_test ();
}

/*
 * Local Variables:
 * mode:c
 * End:
 */
