/* Test j1.
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

static const struct test_f_f_data j1_test_data[] =
  {
    /* j1 is the Bessel function of the first kind of order 1 */
    TEST_f_f (j1, qnan_value, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_f_f (j1, -qnan_value, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_f_f (j1, snan_value, qnan_value, INVALID_EXCEPTION),
    TEST_f_f (j1, -snan_value, qnan_value, INVALID_EXCEPTION),
    TEST_f_f (j1, plus_infty, 0, ERRNO_UNCHANGED|IGNORE_ZERO_INF_SIGN),
    TEST_f_f (j1, minus_infty, minus_zero, ERRNO_UNCHANGED|IGNORE_ZERO_INF_SIGN),

    AUTO_TESTS_f_f (j1),
  };

static void
j1_test (void)
{
  ALL_RM_TEST (j1, 0, j1_test_data, RUN_TEST_LOOP_f_f, END);
}

static void
do_test (void)
{
  j1_test ();
}

/*
 * Local Variables:
 * mode:c
 * End:
 */
