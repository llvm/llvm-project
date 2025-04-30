/* Test catanh.
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

static const struct test_c_c_data catanh_test_data[] =
  {
    TEST_c_c (catanh, 1.0L, plus_zero, plus_infty, plus_zero, DIVIDE_BY_ZERO_EXCEPTION),
    TEST_c_c (catanh, 1.0L, minus_zero, plus_infty, minus_zero, DIVIDE_BY_ZERO_EXCEPTION),
    TEST_c_c (catanh, -1.0L, plus_zero, minus_infty, plus_zero, DIVIDE_BY_ZERO_EXCEPTION),
    TEST_c_c (catanh, -1.0L, minus_zero, minus_infty, minus_zero, DIVIDE_BY_ZERO_EXCEPTION),

    TEST_c_c (catanh, plus_infty, plus_infty, 0.0, lit_pi_2_d),
    TEST_c_c (catanh, plus_infty, minus_infty, 0.0, -lit_pi_2_d),
    TEST_c_c (catanh, minus_infty, plus_infty, minus_zero, lit_pi_2_d),
    TEST_c_c (catanh, minus_infty, minus_infty, minus_zero, -lit_pi_2_d),

    TEST_c_c (catanh, -10.0, plus_infty, minus_zero, lit_pi_2_d),
    TEST_c_c (catanh, -10.0, minus_infty, minus_zero, -lit_pi_2_d),
    TEST_c_c (catanh, minus_zero, plus_infty, minus_zero, lit_pi_2_d),
    TEST_c_c (catanh, minus_zero, minus_infty, minus_zero, -lit_pi_2_d),
    TEST_c_c (catanh, 0, plus_infty, 0.0, lit_pi_2_d),
    TEST_c_c (catanh, 0, minus_infty, 0.0, -lit_pi_2_d),
    TEST_c_c (catanh, 0.1L, plus_infty, 0.0, lit_pi_2_d),
    TEST_c_c (catanh, 0.1L, minus_infty, 0.0, -lit_pi_2_d),

    TEST_c_c (catanh, minus_infty, 0, minus_zero, lit_pi_2_d),
    TEST_c_c (catanh, minus_infty, minus_zero, minus_zero, -lit_pi_2_d),
    TEST_c_c (catanh, minus_infty, 100, minus_zero, lit_pi_2_d),
    TEST_c_c (catanh, minus_infty, -100, minus_zero, -lit_pi_2_d),

    TEST_c_c (catanh, plus_infty, 0, 0.0, lit_pi_2_d),
    TEST_c_c (catanh, plus_infty, minus_zero, 0.0, -lit_pi_2_d),
    TEST_c_c (catanh, plus_infty, 0.5, 0.0, lit_pi_2_d),
    TEST_c_c (catanh, plus_infty, -0.5, 0.0, -lit_pi_2_d),

    TEST_c_c (catanh, 0, qnan_value, 0.0, qnan_value),
    TEST_c_c (catanh, minus_zero, qnan_value, minus_zero, qnan_value),

    TEST_c_c (catanh, plus_infty, qnan_value, 0.0, qnan_value),
    TEST_c_c (catanh, minus_infty, qnan_value, minus_zero, qnan_value),

    TEST_c_c (catanh, qnan_value, 0, qnan_value, qnan_value),
    TEST_c_c (catanh, qnan_value, minus_zero, qnan_value, qnan_value),

    TEST_c_c (catanh, qnan_value, plus_infty, 0.0, lit_pi_2_d, IGNORE_ZERO_INF_SIGN),
    TEST_c_c (catanh, qnan_value, minus_infty, 0.0, -lit_pi_2_d, IGNORE_ZERO_INF_SIGN),

    TEST_c_c (catanh, 10.5, qnan_value, qnan_value, qnan_value, INVALID_EXCEPTION_OK),
    TEST_c_c (catanh, -10.5, qnan_value, qnan_value, qnan_value, INVALID_EXCEPTION_OK),

    TEST_c_c (catanh, qnan_value, 0.75, qnan_value, qnan_value, INVALID_EXCEPTION_OK),
    TEST_c_c (catanh, qnan_value, -0.75, qnan_value, qnan_value, INVALID_EXCEPTION_OK),

    TEST_c_c (catanh, qnan_value, qnan_value, qnan_value, qnan_value),

    AUTO_TESTS_c_c (catanh),
  };

static void
catanh_test (void)
{
  ALL_RM_TEST (catanh, 0, catanh_test_data, RUN_TEST_LOOP_c_c, END_COMPLEX);
}

static void
do_test (void)
{
  catanh_test ();
}

/*
 * Local Variables:
 * mode:c
 * End:
 */
