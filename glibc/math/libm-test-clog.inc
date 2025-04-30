/* Test clog.
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

static const struct test_c_c_data clog_test_data[] =
  {
    TEST_c_c (clog, minus_zero, 0, minus_infty, lit_pi, DIVIDE_BY_ZERO_EXCEPTION),
    TEST_c_c (clog, minus_zero, minus_zero, minus_infty, -lit_pi, DIVIDE_BY_ZERO_EXCEPTION),

    TEST_c_c (clog, 0, 0, minus_infty, 0.0, DIVIDE_BY_ZERO_EXCEPTION),
    TEST_c_c (clog, 0, minus_zero, minus_infty, minus_zero, DIVIDE_BY_ZERO_EXCEPTION),

    TEST_c_c (clog, minus_infty, plus_infty, plus_infty, lit_pi_3_m_4_d),
    TEST_c_c (clog, minus_infty, minus_infty, plus_infty, -lit_pi_3_m_4_d),

    TEST_c_c (clog, plus_infty, plus_infty, plus_infty, lit_pi_4_d),
    TEST_c_c (clog, plus_infty, minus_infty, plus_infty, -lit_pi_4_d),

    TEST_c_c (clog, 0, plus_infty, plus_infty, lit_pi_2_d),
    TEST_c_c (clog, 3, plus_infty, plus_infty, lit_pi_2_d),
    TEST_c_c (clog, minus_zero, plus_infty, plus_infty, lit_pi_2_d),
    TEST_c_c (clog, -3, plus_infty, plus_infty, lit_pi_2_d),
    TEST_c_c (clog, 0, minus_infty, plus_infty, -lit_pi_2_d),
    TEST_c_c (clog, 3, minus_infty, plus_infty, -lit_pi_2_d),
    TEST_c_c (clog, minus_zero, minus_infty, plus_infty, -lit_pi_2_d),
    TEST_c_c (clog, -3, minus_infty, plus_infty, -lit_pi_2_d),

    TEST_c_c (clog, minus_infty, 0, plus_infty, lit_pi),
    TEST_c_c (clog, minus_infty, 1, plus_infty, lit_pi),
    TEST_c_c (clog, minus_infty, minus_zero, plus_infty, -lit_pi),
    TEST_c_c (clog, minus_infty, -1, plus_infty, -lit_pi),

    TEST_c_c (clog, plus_infty, 0, plus_infty, 0.0),
    TEST_c_c (clog, plus_infty, 1, plus_infty, 0.0),
    TEST_c_c (clog, plus_infty, minus_zero, plus_infty, minus_zero),
    TEST_c_c (clog, plus_infty, -1, plus_infty, minus_zero),

    TEST_c_c (clog, plus_infty, qnan_value, plus_infty, qnan_value),
    TEST_c_c (clog, minus_infty, qnan_value, plus_infty, qnan_value),

    TEST_c_c (clog, qnan_value, plus_infty, plus_infty, qnan_value),
    TEST_c_c (clog, qnan_value, minus_infty, plus_infty, qnan_value),

    TEST_c_c (clog, 0, qnan_value, qnan_value, qnan_value, INVALID_EXCEPTION_OK),
    TEST_c_c (clog, 3, qnan_value, qnan_value, qnan_value, INVALID_EXCEPTION_OK),
    TEST_c_c (clog, minus_zero, qnan_value, qnan_value, qnan_value, INVALID_EXCEPTION_OK),
    TEST_c_c (clog, -3, qnan_value, qnan_value, qnan_value, INVALID_EXCEPTION_OK),

    TEST_c_c (clog, qnan_value, 0, qnan_value, qnan_value, INVALID_EXCEPTION_OK),
    TEST_c_c (clog, qnan_value, 5, qnan_value, qnan_value, INVALID_EXCEPTION_OK),
    TEST_c_c (clog, qnan_value, minus_zero, qnan_value, qnan_value, INVALID_EXCEPTION_OK),
    TEST_c_c (clog, qnan_value, -5, qnan_value, qnan_value, INVALID_EXCEPTION_OK),

    TEST_c_c (clog, qnan_value, qnan_value, qnan_value, qnan_value),

    AUTO_TESTS_c_c (clog),
  };

static void
clog_test (void)
{
  ALL_RM_TEST (clog, 0, clog_test_data, RUN_TEST_LOOP_c_c, END_COMPLEX);
}

static void
do_test (void)
{
  clog_test ();
}

/*
 * Local Variables:
 * mode:c
 * End:
 */
