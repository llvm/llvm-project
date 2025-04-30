/* Test cexp.
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

static const struct test_c_c_data cexp_test_data[] =
  {
    TEST_c_c (cexp, plus_infty, plus_zero, plus_infty, 0.0),
    TEST_c_c (cexp, plus_infty, minus_zero, plus_infty, minus_zero),

    TEST_c_c (cexp, minus_infty, plus_zero, 0.0, 0.0),
    TEST_c_c (cexp, minus_infty, minus_zero, 0.0, minus_zero),

    TEST_c_c (cexp, 0.0, plus_infty, qnan_value, qnan_value, INVALID_EXCEPTION),
    TEST_c_c (cexp, minus_zero, plus_infty, qnan_value, qnan_value, INVALID_EXCEPTION),

    TEST_c_c (cexp, 0.0, minus_infty, qnan_value, qnan_value, INVALID_EXCEPTION),
    TEST_c_c (cexp, minus_zero, minus_infty, qnan_value, qnan_value, INVALID_EXCEPTION),

    TEST_c_c (cexp, 100.0, plus_infty, qnan_value, qnan_value, INVALID_EXCEPTION),
    TEST_c_c (cexp, -100.0, plus_infty, qnan_value, qnan_value, INVALID_EXCEPTION),

    TEST_c_c (cexp, 100.0, minus_infty, qnan_value, qnan_value, INVALID_EXCEPTION),
    TEST_c_c (cexp, -100.0, minus_infty, qnan_value, qnan_value, INVALID_EXCEPTION),

    TEST_c_c (cexp, minus_infty, 2.0, minus_zero, 0.0),
    TEST_c_c (cexp, minus_infty, 4.0, minus_zero, minus_zero),
    TEST_c_c (cexp, plus_infty, 2.0, minus_infty, plus_infty),
    TEST_c_c (cexp, plus_infty, 4.0, minus_infty, minus_infty),

    TEST_c_c (cexp, plus_infty, plus_infty, plus_infty, qnan_value, INVALID_EXCEPTION|IGNORE_ZERO_INF_SIGN),
    TEST_c_c (cexp, plus_infty, minus_infty, plus_infty, qnan_value, INVALID_EXCEPTION|IGNORE_ZERO_INF_SIGN),

    TEST_c_c (cexp, minus_infty, plus_infty, 0.0, 0.0, IGNORE_ZERO_INF_SIGN),
    TEST_c_c (cexp, minus_infty, minus_infty, 0.0, minus_zero, IGNORE_ZERO_INF_SIGN),

    TEST_c_c (cexp, minus_infty, qnan_value, 0, 0, IGNORE_ZERO_INF_SIGN),

    TEST_c_c (cexp, plus_infty, qnan_value, plus_infty, qnan_value),

    TEST_c_c (cexp, qnan_value, 0.0, qnan_value, 0.0),
    TEST_c_c (cexp, qnan_value, minus_zero, qnan_value, minus_zero),
    TEST_c_c (cexp, qnan_value, 1.0, qnan_value, qnan_value, INVALID_EXCEPTION_OK),

    TEST_c_c (cexp, qnan_value, plus_infty, qnan_value, qnan_value, INVALID_EXCEPTION_OK),
    TEST_c_c (cexp, 0, qnan_value, qnan_value, qnan_value, INVALID_EXCEPTION_OK),
    TEST_c_c (cexp, 1, qnan_value, qnan_value, qnan_value, INVALID_EXCEPTION_OK),
    TEST_c_c (cexp, qnan_value, qnan_value, qnan_value, qnan_value),

    AUTO_TESTS_c_c (cexp),
  };

static void
cexp_test (void)
{
  ALL_RM_TEST (cexp, 0, cexp_test_data, RUN_TEST_LOOP_c_c, END_COMPLEX);
}

static void
do_test (void)
{
  cexp_test ();
}

/*
 * Local Variables:
 * mode:c
 * End:
 */
