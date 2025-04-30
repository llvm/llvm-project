/* Test jn.
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

static const struct test_if_f_data jn_test_data[] =
  {
    /* jn is the Bessel function of the first kind of order n.  */
    /* jn (0, x) == j0 (x)  */
    TEST_if_f (jn, 0, qnan_value, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_if_f (jn, 0, -qnan_value, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_if_f (jn, 0, snan_value, qnan_value, INVALID_EXCEPTION),
    TEST_if_f (jn, 0, -snan_value, qnan_value, INVALID_EXCEPTION),
    TEST_if_f (jn, 0, plus_infty, 0, ERRNO_UNCHANGED|IGNORE_ZERO_INF_SIGN),
    TEST_if_f (jn, 0, minus_infty, 0, ERRNO_UNCHANGED|IGNORE_ZERO_INF_SIGN),

    /* jn (1, x) == j1 (x)  */
    TEST_if_f (jn, 1, qnan_value, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_if_f (jn, 1, -qnan_value, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_if_f (jn, 1, snan_value, qnan_value, INVALID_EXCEPTION),
    TEST_if_f (jn, 1, -snan_value, qnan_value, INVALID_EXCEPTION),
    TEST_if_f (jn, 1, plus_infty, 0, ERRNO_UNCHANGED|IGNORE_ZERO_INF_SIGN),
    TEST_if_f (jn, 1, minus_infty, minus_zero, ERRNO_UNCHANGED|IGNORE_ZERO_INF_SIGN),

    /* jn (3, x)  */
    TEST_if_f (jn, 3, qnan_value, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_if_f (jn, 3, -qnan_value, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_if_f (jn, 3, snan_value, qnan_value, INVALID_EXCEPTION),
    TEST_if_f (jn, 3, -snan_value, qnan_value, INVALID_EXCEPTION),
    TEST_if_f (jn, 3, plus_infty, 0, ERRNO_UNCHANGED|IGNORE_ZERO_INF_SIGN),
    TEST_if_f (jn, 3, minus_infty, minus_zero, ERRNO_UNCHANGED|IGNORE_ZERO_INF_SIGN),

    /*  jn (10, x)  */
    TEST_if_f (jn, 10, qnan_value, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_if_f (jn, 10, -qnan_value, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_if_f (jn, 10, snan_value, qnan_value, INVALID_EXCEPTION),
    TEST_if_f (jn, 10, -snan_value, qnan_value, INVALID_EXCEPTION),
    TEST_if_f (jn, 10, plus_infty, 0, ERRNO_UNCHANGED|IGNORE_ZERO_INF_SIGN),
    TEST_if_f (jn, 10, minus_infty, 0, ERRNO_UNCHANGED|IGNORE_ZERO_INF_SIGN),

    AUTO_TESTS_if_f (jn),
  };

static void
jn_test (void)
{
  ALL_RM_TEST (jn, 0, jn_test_data, RUN_TEST_LOOP_if_f, END);
}

static void
do_test (void)
{
  jn_test ();
}

/*
 * Local Variables:
 * mode:c
 * End:
 */
