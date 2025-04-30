/* Test casin.
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

static const struct test_c_c_data casin_test_data[] =
  {
    TEST_c_c (casin, plus_infty, plus_infty, lit_pi_4_d, plus_infty),
    TEST_c_c (casin, plus_infty, minus_infty, lit_pi_4_d, minus_infty),
    TEST_c_c (casin, minus_infty, plus_infty, -lit_pi_4_d, plus_infty),
    TEST_c_c (casin, minus_infty, minus_infty, -lit_pi_4_d, minus_infty),

    TEST_c_c (casin, -10.0, plus_infty, minus_zero, plus_infty),
    TEST_c_c (casin, -10.0, minus_infty, minus_zero, minus_infty),
    TEST_c_c (casin, 0, plus_infty, 0.0, plus_infty),
    TEST_c_c (casin, 0, minus_infty, 0.0, minus_infty),
    TEST_c_c (casin, minus_zero, plus_infty, minus_zero, plus_infty),
    TEST_c_c (casin, minus_zero, minus_infty, minus_zero, minus_infty),
    TEST_c_c (casin, 0.1L, plus_infty, 0.0, plus_infty),
    TEST_c_c (casin, 0.1L, minus_infty, 0.0, minus_infty),

    TEST_c_c (casin, minus_infty, 0, -lit_pi_2_d, plus_infty),
    TEST_c_c (casin, minus_infty, minus_zero, -lit_pi_2_d, minus_infty),
    TEST_c_c (casin, minus_infty, 100, -lit_pi_2_d, plus_infty),
    TEST_c_c (casin, minus_infty, -100, -lit_pi_2_d, minus_infty),

    TEST_c_c (casin, plus_infty, 0, lit_pi_2_d, plus_infty),
    TEST_c_c (casin, plus_infty, minus_zero, lit_pi_2_d, minus_infty),
    TEST_c_c (casin, plus_infty, 0.5, lit_pi_2_d, plus_infty),
    TEST_c_c (casin, plus_infty, -0.5, lit_pi_2_d, minus_infty),

    TEST_c_c (casin, qnan_value, plus_infty, qnan_value, plus_infty),
    TEST_c_c (casin, qnan_value, minus_infty, qnan_value, minus_infty),

    TEST_c_c (casin, 0.0, qnan_value, 0.0, qnan_value),
    TEST_c_c (casin, minus_zero, qnan_value, minus_zero, qnan_value),

    TEST_c_c (casin, plus_infty, qnan_value, qnan_value, plus_infty, IGNORE_ZERO_INF_SIGN),
    TEST_c_c (casin, minus_infty, qnan_value, qnan_value, plus_infty, IGNORE_ZERO_INF_SIGN),

    TEST_c_c (casin, qnan_value, 10.5, qnan_value, qnan_value, INVALID_EXCEPTION_OK),
    TEST_c_c (casin, qnan_value, -10.5, qnan_value, qnan_value, INVALID_EXCEPTION_OK),

    TEST_c_c (casin, 0.75, qnan_value, qnan_value, qnan_value, INVALID_EXCEPTION_OK),
    TEST_c_c (casin, -0.75, qnan_value, qnan_value, qnan_value, INVALID_EXCEPTION_OK),

    TEST_c_c (casin, qnan_value, qnan_value, qnan_value, qnan_value),

    AUTO_TESTS_c_c (casin),
  };

static void
casin_test (void)
{
  ALL_RM_TEST (casin, 0, casin_test_data, RUN_TEST_LOOP_c_c, END_COMPLEX);
}

static void
do_test (void)
{
  casin_test ();
}

/*
 * Local Variables:
 * mode:c
 * End:
 */
