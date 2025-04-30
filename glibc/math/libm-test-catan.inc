/* Test catan.
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

static const struct test_c_c_data catan_test_data[] =
  {
    TEST_c_c (catan, plus_zero, 1.0L, plus_zero, plus_infty, DIVIDE_BY_ZERO_EXCEPTION),
    TEST_c_c (catan, minus_zero, 1.0L, minus_zero, plus_infty, DIVIDE_BY_ZERO_EXCEPTION),
    TEST_c_c (catan, plus_zero, -1.0L, plus_zero, minus_infty, DIVIDE_BY_ZERO_EXCEPTION),
    TEST_c_c (catan, minus_zero, -1.0L, minus_zero, minus_infty, DIVIDE_BY_ZERO_EXCEPTION),

    TEST_c_c (catan, plus_infty, plus_infty, lit_pi_2_d, 0),
    TEST_c_c (catan, plus_infty, minus_infty, lit_pi_2_d, minus_zero),
    TEST_c_c (catan, minus_infty, plus_infty, -lit_pi_2_d, 0),
    TEST_c_c (catan, minus_infty, minus_infty, -lit_pi_2_d, minus_zero),


    TEST_c_c (catan, plus_infty, -10.0, lit_pi_2_d, minus_zero),
    TEST_c_c (catan, minus_infty, -10.0, -lit_pi_2_d, minus_zero),
    TEST_c_c (catan, plus_infty, minus_zero, lit_pi_2_d, minus_zero),
    TEST_c_c (catan, minus_infty, minus_zero, -lit_pi_2_d, minus_zero),
    TEST_c_c (catan, plus_infty, 0.0, lit_pi_2_d, 0),
    TEST_c_c (catan, minus_infty, 0.0, -lit_pi_2_d, 0),
    TEST_c_c (catan, plus_infty, 0.1L, lit_pi_2_d, 0),
    TEST_c_c (catan, minus_infty, 0.1L, -lit_pi_2_d, 0),

    TEST_c_c (catan, 0.0, minus_infty, lit_pi_2_d, minus_zero),
    TEST_c_c (catan, minus_zero, minus_infty, -lit_pi_2_d, minus_zero),
    TEST_c_c (catan, 100.0, minus_infty, lit_pi_2_d, minus_zero),
    TEST_c_c (catan, -100.0, minus_infty, -lit_pi_2_d, minus_zero),

    TEST_c_c (catan, 0.0, plus_infty, lit_pi_2_d, 0),
    TEST_c_c (catan, minus_zero, plus_infty, -lit_pi_2_d, 0),
    TEST_c_c (catan, 0.5, plus_infty, lit_pi_2_d, 0),
    TEST_c_c (catan, -0.5, plus_infty, -lit_pi_2_d, 0),

    TEST_c_c (catan, qnan_value, 0.0, qnan_value, 0),
    TEST_c_c (catan, qnan_value, minus_zero, qnan_value, minus_zero),

    TEST_c_c (catan, qnan_value, plus_infty, qnan_value, 0),
    TEST_c_c (catan, qnan_value, minus_infty, qnan_value, minus_zero),

    TEST_c_c (catan, 0.0, qnan_value, qnan_value, qnan_value),
    TEST_c_c (catan, minus_zero, qnan_value, qnan_value, qnan_value),

    TEST_c_c (catan, plus_infty, qnan_value, lit_pi_2_d, 0, IGNORE_ZERO_INF_SIGN),
    TEST_c_c (catan, minus_infty, qnan_value, -lit_pi_2_d, 0, IGNORE_ZERO_INF_SIGN),

    TEST_c_c (catan, qnan_value, 10.5, qnan_value, qnan_value, INVALID_EXCEPTION_OK),
    TEST_c_c (catan, qnan_value, -10.5, qnan_value, qnan_value, INVALID_EXCEPTION_OK),

    TEST_c_c (catan, 0.75, qnan_value, qnan_value, qnan_value, INVALID_EXCEPTION_OK),
    TEST_c_c (catan, -0.75, qnan_value, qnan_value, qnan_value, INVALID_EXCEPTION_OK),

    TEST_c_c (catan, qnan_value, qnan_value, qnan_value, qnan_value),

    AUTO_TESTS_c_c (catan),
  };

static void
catan_test (void)
{
  ALL_RM_TEST (catan, 0, catan_test_data, RUN_TEST_LOOP_c_c, END_COMPLEX);
}

static void
do_test (void)
{
  catan_test ();
}

/*
 * Local Variables:
 * mode:c
 * End:
 */
