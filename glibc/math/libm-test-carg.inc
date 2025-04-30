/* Test carg.
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

static const struct test_c_f_data carg_test_data[] =
  {
    /* carg (x + iy) is specified as atan2 (y, x) */

    /* carg (inf + i y) == +0 for finite y > 0.  */
    TEST_c_f (carg, plus_infty, 2.0, 0),

    /* carg (inf + i y) == -0 for finite y < 0.  */
    TEST_c_f (carg, plus_infty, -2.0, minus_zero),

    /* carg(x + i inf) == pi/2 for finite x.  */
    TEST_c_f (carg, 10.0, plus_infty, lit_pi_2_d),

    /* carg(x - i inf) == -pi/2 for finite x.  */
    TEST_c_f (carg, 10.0, minus_infty, -lit_pi_2_d),

    /* carg (-inf + i y) == +pi for finite y > 0.  */
    TEST_c_f (carg, minus_infty, 10.0, lit_pi),

    /* carg (-inf + i y) == -pi for finite y < 0.  */
    TEST_c_f (carg, minus_infty, -10.0, -lit_pi),

    TEST_c_f (carg, plus_infty, plus_infty, lit_pi_4_d),

    TEST_c_f (carg, plus_infty, minus_infty, -lit_pi_4_d),

    TEST_c_f (carg, minus_infty, plus_infty, lit_pi_3_m_4_d),

    TEST_c_f (carg, minus_infty, minus_infty, -lit_pi_3_m_4_d),

    TEST_c_f (carg, qnan_value, qnan_value, qnan_value),

    AUTO_TESTS_c_f (carg),
  };

static void
carg_test (void)
{
  ALL_RM_TEST (carg, 0, carg_test_data, RUN_TEST_LOOP_c_f, END);
}

static void
do_test (void)
{
  carg_test ();
}

/*
 * Local Variables:
 * mode:c
 * End:
 */
