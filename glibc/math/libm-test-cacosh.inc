/* Test cacosh.
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

static const struct test_c_c_data cacosh_test_data[] =
  {
    TEST_c_c (cacosh, minus_infty, plus_infty, plus_infty, lit_pi_3_m_4_d),
    TEST_c_c (cacosh, minus_infty, minus_infty, plus_infty, -lit_pi_3_m_4_d),

    TEST_c_c (cacosh, plus_infty, plus_infty, plus_infty, lit_pi_4_d),
    TEST_c_c (cacosh, plus_infty, minus_infty, plus_infty, -lit_pi_4_d),

    TEST_c_c (cacosh, -10.0, plus_infty, plus_infty, lit_pi_2_d),
    TEST_c_c (cacosh, -10.0, minus_infty, plus_infty, -lit_pi_2_d),
    TEST_c_c (cacosh, 0, plus_infty, plus_infty, lit_pi_2_d),
    TEST_c_c (cacosh, 0, minus_infty, plus_infty, -lit_pi_2_d),
    TEST_c_c (cacosh, 0.1L, plus_infty, plus_infty, lit_pi_2_d),
    TEST_c_c (cacosh, 0.1L, minus_infty, plus_infty, -lit_pi_2_d),

    TEST_c_c (cacosh, minus_infty, 0, plus_infty, lit_pi),
    TEST_c_c (cacosh, minus_infty, minus_zero, plus_infty, -lit_pi),
    TEST_c_c (cacosh, minus_infty, 100, plus_infty, lit_pi),
    TEST_c_c (cacosh, minus_infty, -100, plus_infty, -lit_pi),

    TEST_c_c (cacosh, plus_infty, 0, plus_infty, 0.0),
    TEST_c_c (cacosh, plus_infty, minus_zero, plus_infty, minus_zero),
    TEST_c_c (cacosh, plus_infty, 0.5, plus_infty, 0.0),
    TEST_c_c (cacosh, plus_infty, -0.5, plus_infty, minus_zero),

    TEST_c_c (cacosh, plus_infty, qnan_value, plus_infty, qnan_value),
    TEST_c_c (cacosh, minus_infty, qnan_value, plus_infty, qnan_value),

    /* The sign of the imaginary part of the result is unspecified
       (see
       <http://www.open-std.org/jtc1/sc22/wg14/www/docs/n2173.htm>),
       but the libm-test infrastructure does not support describing
       that.  */
    TEST_c_c (cacosh, 0, qnan_value, qnan_value, lit_pi_2_d),
    TEST_c_c (cacosh, minus_zero, qnan_value, qnan_value, lit_pi_2_d),

    TEST_c_c (cacosh, qnan_value, plus_infty, plus_infty, qnan_value),
    TEST_c_c (cacosh, qnan_value, minus_infty, plus_infty, qnan_value),

    TEST_c_c (cacosh, 10.5, qnan_value, qnan_value, qnan_value, INVALID_EXCEPTION_OK),
    TEST_c_c (cacosh, -10.5, qnan_value, qnan_value, qnan_value, INVALID_EXCEPTION_OK),

    TEST_c_c (cacosh, qnan_value, 0.75, qnan_value, qnan_value, INVALID_EXCEPTION_OK),
    TEST_c_c (cacosh, qnan_value, -0.75, qnan_value, qnan_value, INVALID_EXCEPTION_OK),

    TEST_c_c (cacosh, qnan_value, qnan_value, qnan_value, qnan_value),

    AUTO_TESTS_c_c (cacosh),
  };

static void
cacosh_test (void)
{
  ALL_RM_TEST (cacosh, 0, cacosh_test_data, RUN_TEST_LOOP_c_c, END_COMPLEX);
}

static void
do_test (void)
{
  cacosh_test ();
}

/*
 * Local Variables:
 * mode:c
 * End:
 */
