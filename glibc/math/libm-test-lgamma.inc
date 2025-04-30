/* Test lgamma.
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

static const struct test_f_f1_data lgamma_test_data[] =
  {
    TEST_f_f1 (lgamma, plus_infty, plus_infty, 1, ERRNO_UNCHANGED),
    TEST_f_f1 (lgamma, 0, plus_infty, 1, DIVIDE_BY_ZERO_EXCEPTION|ERRNO_ERANGE),
    TEST_f_f1 (lgamma, minus_zero, plus_infty, -1, DIVIDE_BY_ZERO_EXCEPTION|ERRNO_ERANGE),
    TEST_f_f1 (lgamma, qnan_value, qnan_value, IGNORE, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_f_f1 (lgamma, -qnan_value, qnan_value, IGNORE, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_f_f1 (lgamma, snan_value, qnan_value, IGNORE, INVALID_EXCEPTION),
    TEST_f_f1 (lgamma, -snan_value, qnan_value, IGNORE, INVALID_EXCEPTION),

    /* lgamma (x) == +inf plus divide by zero exception for integer x <= 0.  */
    TEST_f_f1 (lgamma, -3, plus_infty, IGNORE, DIVIDE_BY_ZERO_EXCEPTION|ERRNO_ERANGE),
    TEST_f_f1 (lgamma, minus_infty, plus_infty, IGNORE, ERRNO_UNCHANGED),
    TEST_f_f1 (lgamma, -max_value, plus_infty, IGNORE, DIVIDE_BY_ZERO_EXCEPTION|ERRNO_ERANGE),

    AUTO_TESTS_f_f1 (lgamma),
  };

static void
lgamma_test (void)
{
  ALL_RM_TEST (lgamma, 0, lgamma_test_data, RUN_TEST_LOOP_f_f1, END, signgam);
}

static void
gamma_test (void)
{
#if !TEST_FLOATN
  /* gamma uses the same test data as lgamma.  */
  ALL_RM_TEST (gamma, 0, lgamma_test_data, RUN_TEST_LOOP_f_f1, END, signgam);
#endif
}

static void
do_test (void)
{
  lgamma_test ();
  gamma_test ();
}

/*
 * Local Variables:
 * mode:c
 * End:
 */
