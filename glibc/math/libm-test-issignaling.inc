/* Test issignaling.
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

static const struct test_f_i_data issignaling_test_data[] =
  {
    TEST_f_b (issignaling, 0, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_f_b (issignaling, minus_zero, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_f_b (issignaling, 10, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_f_b (issignaling, min_subnorm_value, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_f_b (issignaling, -min_subnorm_value, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_f_b (issignaling, min_value, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_f_b (issignaling, -min_value, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_f_b (issignaling, max_value, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_f_b (issignaling, -max_value, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_f_b (issignaling, plus_infty, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_f_b (issignaling, minus_infty, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_f_b (issignaling, qnan_value, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_f_b (issignaling, -qnan_value, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_f_b (issignaling, snan_value, 1, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_f_b (issignaling, -snan_value, 1, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_f_b (issignaling, snan_value_MACRO, 1, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_f_b (issignaling, -snan_value_MACRO, 1, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
  };

#if TEST_COND_intel96
static const struct test_j_i_data_u issignaling_test_data_u[] =
  {
    TEST_j_b (issignaling, pseudo_zero, 1, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_j_b (issignaling, pseudo_inf, 1, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_j_b (issignaling, pseudo_qnan, 1, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_j_b (issignaling, pseudo_snan, 1, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_j_b (issignaling, pseudo_unnormal, 1, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
  };
#endif

static void
issignaling_test (void)
{
  ALL_RM_TEST (issignaling, 1, issignaling_test_data, RUN_TEST_LOOP_f_b_tg, END);
#if TEST_COND_intel96
  ALL_RM_TEST (issignaling, 1, issignaling_test_data_u, RUN_TEST_LOOP_j_b_tg_u, END);
#endif
}

static void
do_test (void)
{
  issignaling_test ();
}

/*
 * Local Variables:
 * mode:c
 * End:
 */
