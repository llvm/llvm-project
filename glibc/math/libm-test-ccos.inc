/* Test ccos.
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

static const struct test_c_c_data ccos_test_data[] =
  {
    TEST_c_c (ccos, plus_infty, 0.0, qnan_value, 0.0, INVALID_EXCEPTION|IGNORE_ZERO_INF_SIGN),
    TEST_c_c (ccos, plus_infty, minus_zero, qnan_value, 0.0, INVALID_EXCEPTION|IGNORE_ZERO_INF_SIGN),
    TEST_c_c (ccos, minus_infty, 0.0, qnan_value, 0.0, INVALID_EXCEPTION|IGNORE_ZERO_INF_SIGN),
    TEST_c_c (ccos, minus_infty, minus_zero, qnan_value, 0.0, INVALID_EXCEPTION|IGNORE_ZERO_INF_SIGN),

    TEST_c_c (ccos, 0.0, plus_infty, plus_infty, minus_zero),
    TEST_c_c (ccos, 0.0, minus_infty, plus_infty, 0.0),
    TEST_c_c (ccos, minus_zero, plus_infty, plus_infty, 0.0),
    TEST_c_c (ccos, minus_zero, minus_infty, plus_infty, minus_zero),

    TEST_c_c (ccos, plus_infty, plus_infty, plus_infty, qnan_value, INVALID_EXCEPTION),
    TEST_c_c (ccos, minus_infty, plus_infty, plus_infty, qnan_value, INVALID_EXCEPTION),
    TEST_c_c (ccos, plus_infty, minus_infty, plus_infty, qnan_value, INVALID_EXCEPTION),
    TEST_c_c (ccos, minus_infty, minus_infty, plus_infty, qnan_value, INVALID_EXCEPTION),

    TEST_c_c (ccos, 4.625, plus_infty, minus_infty, plus_infty),
    TEST_c_c (ccos, 4.625, minus_infty, minus_infty, minus_infty),
    TEST_c_c (ccos, -4.625, plus_infty, minus_infty, minus_infty),
    TEST_c_c (ccos, -4.625, minus_infty, minus_infty, plus_infty),

    TEST_c_c (ccos, plus_infty, 6.75, qnan_value, qnan_value, INVALID_EXCEPTION),
    TEST_c_c (ccos, plus_infty, -6.75, qnan_value, qnan_value, INVALID_EXCEPTION),
    TEST_c_c (ccos, minus_infty, 6.75, qnan_value, qnan_value, INVALID_EXCEPTION),
    TEST_c_c (ccos, minus_infty, -6.75, qnan_value, qnan_value, INVALID_EXCEPTION),

    TEST_c_c (ccos, qnan_value, 0.0, qnan_value, 0.0, IGNORE_ZERO_INF_SIGN),
    TEST_c_c (ccos, qnan_value, minus_zero, qnan_value, 0.0, IGNORE_ZERO_INF_SIGN),

    TEST_c_c (ccos, qnan_value, plus_infty, plus_infty, qnan_value),
    TEST_c_c (ccos, qnan_value, minus_infty, plus_infty, qnan_value),

    TEST_c_c (ccos, qnan_value, 9.0, qnan_value, qnan_value, INVALID_EXCEPTION_OK),
    TEST_c_c (ccos, qnan_value, -9.0, qnan_value, qnan_value, INVALID_EXCEPTION_OK),

    TEST_c_c (ccos, 0.0, qnan_value, qnan_value, 0.0, IGNORE_ZERO_INF_SIGN),
    TEST_c_c (ccos, minus_zero, qnan_value, qnan_value, 0.0, IGNORE_ZERO_INF_SIGN),

    TEST_c_c (ccos, 10.0, qnan_value, qnan_value, qnan_value, INVALID_EXCEPTION_OK),
    TEST_c_c (ccos, -10.0, qnan_value, qnan_value, qnan_value, INVALID_EXCEPTION_OK),

    TEST_c_c (ccos, plus_infty, qnan_value, qnan_value, qnan_value, INVALID_EXCEPTION_OK),
    TEST_c_c (ccos, minus_infty, qnan_value, qnan_value, qnan_value, INVALID_EXCEPTION_OK),

    TEST_c_c (ccos, qnan_value, qnan_value, qnan_value, qnan_value),

    AUTO_TESTS_c_c (ccos),
  };

static void
ccos_test (void)
{
  ALL_RM_TEST (ccos, 0, ccos_test_data, RUN_TEST_LOOP_c_c, END_COMPLEX);
}

static void
do_test (void)
{
  ccos_test ();
}

/*
 * Local Variables:
 * mode:c
 * End:
 */
