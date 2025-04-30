/* Test fpclassify.
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

static const struct test_f_i_data fpclassify_test_data[] =
  {
    TEST_f_i (fpclassify, qnan_value, FP_NAN, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_f_i (fpclassify, -qnan_value, FP_NAN, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_f_i (fpclassify, snan_value, FP_NAN, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_f_i (fpclassify, -snan_value, FP_NAN, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_f_i (fpclassify, plus_infty, FP_INFINITE, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_f_i (fpclassify, minus_infty, FP_INFINITE, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_f_i (fpclassify, plus_zero, FP_ZERO, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_f_i (fpclassify, minus_zero, FP_ZERO, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_f_i (fpclassify, 1000, FP_NORMAL, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_f_i (fpclassify, max_value, FP_NORMAL, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_f_i (fpclassify, -max_value, FP_NORMAL, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_f_i (fpclassify, min_value, FP_NORMAL, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_f_i (fpclassify, -min_value, FP_NORMAL, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_f_i (fpclassify, min_subnorm_value, FP_SUBNORMAL, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_f_i (fpclassify, -min_subnorm_value, FP_SUBNORMAL, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
  };

#if TEST_COND_intel96
static const struct test_j_i_data_u fpclassify_test_data_u[] =
  {
    TEST_j_i (fpclassify, pseudo_zero, FP_NAN, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_j_i (fpclassify, pseudo_inf, FP_NAN, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_j_i (fpclassify, pseudo_qnan, FP_NAN, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_j_i (fpclassify, pseudo_snan, FP_NAN, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_j_i (fpclassify, pseudo_unnormal, FP_NAN, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
  };
#endif

static void
fpclassify_test (void)
{
  ALL_RM_TEST (fpclassify, 1, fpclassify_test_data, RUN_TEST_LOOP_f_i_tg, END);
#if TEST_COND_intel96
  ALL_RM_TEST (fpclassify, 1, fpclassify_test_data_u, RUN_TEST_LOOP_j_i_tg_u, END);
#endif
}

static void
do_test (void)
{
  fpclassify_test ();
}

/*
 * Local Variables:
 * mode:c
 * End:
 */
