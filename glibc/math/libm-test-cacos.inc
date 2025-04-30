/* Test cacos.
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

static const struct test_c_c_data cacos_test_data[] =
  {
    TEST_c_c (cacos, minus_infty, plus_infty, lit_pi_3_m_4_d, minus_infty),
    TEST_c_c (cacos, minus_infty, minus_infty, lit_pi_3_m_4_d, plus_infty),

    TEST_c_c (cacos, plus_infty, plus_infty, lit_pi_4_d, minus_infty),
    TEST_c_c (cacos, plus_infty, minus_infty, lit_pi_4_d, plus_infty),

    TEST_c_c (cacos, -10.0, plus_infty, lit_pi_2_d, minus_infty),
    TEST_c_c (cacos, -10.0, minus_infty, lit_pi_2_d, plus_infty),
    TEST_c_c (cacos, 0, plus_infty, lit_pi_2_d, minus_infty),
    TEST_c_c (cacos, 0, minus_infty, lit_pi_2_d, plus_infty),
    TEST_c_c (cacos, 0.1L, plus_infty, lit_pi_2_d, minus_infty),
    TEST_c_c (cacos, 0.1L, minus_infty, lit_pi_2_d, plus_infty),

    TEST_c_c (cacos, minus_infty, 0, lit_pi, minus_infty),
    TEST_c_c (cacos, minus_infty, minus_zero, lit_pi, plus_infty),
    TEST_c_c (cacos, minus_infty, 100, lit_pi, minus_infty),
    TEST_c_c (cacos, minus_infty, -100, lit_pi, plus_infty),

    TEST_c_c (cacos, plus_infty, 0, 0.0, minus_infty),
    TEST_c_c (cacos, plus_infty, minus_zero, 0.0, plus_infty),
    TEST_c_c (cacos, plus_infty, 0.5, 0.0, minus_infty),
    TEST_c_c (cacos, plus_infty, -0.5, 0.0, plus_infty),

    TEST_c_c (cacos, plus_infty, qnan_value, qnan_value, plus_infty, IGNORE_ZERO_INF_SIGN),
    TEST_c_c (cacos, minus_infty, qnan_value, qnan_value, plus_infty, IGNORE_ZERO_INF_SIGN),

    TEST_c_c (cacos, 0, qnan_value, lit_pi_2_d, qnan_value),
    TEST_c_c (cacos, minus_zero, qnan_value, lit_pi_2_d, qnan_value),

    TEST_c_c (cacos, qnan_value, plus_infty, qnan_value, minus_infty),
    TEST_c_c (cacos, qnan_value, minus_infty, qnan_value, plus_infty),

    TEST_c_c (cacos, 10.5, qnan_value, qnan_value, qnan_value, INVALID_EXCEPTION_OK),
    TEST_c_c (cacos, -10.5, qnan_value, qnan_value, qnan_value, INVALID_EXCEPTION_OK),

    TEST_c_c (cacos, qnan_value, 0.75, qnan_value, qnan_value, INVALID_EXCEPTION_OK),
    TEST_c_c (cacos, qnan_value, -0.75, qnan_value, qnan_value, INVALID_EXCEPTION_OK),

    TEST_c_c (cacos, qnan_value, qnan_value, qnan_value, qnan_value),

    AUTO_TESTS_c_c (cacos),
  };

static void
cacos_test (void)
{
  ALL_RM_TEST (cacos, 0, cacos_test_data, RUN_TEST_LOOP_c_c, END_COMPLEX);
}

static void
do_test (void)
{
  cacos_test ();
}

/*
 * Local Variables:
 * mode:c
 * End:
 */
