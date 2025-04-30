/* Test log10.
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

static const struct test_f_f_data log10_test_data[] =
  {
    TEST_f_f (log10, 0, minus_infty, DIVIDE_BY_ZERO_EXCEPTION|ERRNO_ERANGE),
    TEST_f_f (log10, minus_zero, minus_infty, DIVIDE_BY_ZERO_EXCEPTION|ERRNO_ERANGE),

    /* log10 (x) == qNaN plus invalid exception if x < 0.  */
    TEST_f_f (log10, -1, qnan_value, INVALID_EXCEPTION|ERRNO_EDOM),
    TEST_f_f (log10, -min_subnorm_value, qnan_value, INVALID_EXCEPTION|ERRNO_EDOM),
    TEST_f_f (log10, -min_value, qnan_value, INVALID_EXCEPTION|ERRNO_EDOM),
    TEST_f_f (log10, -max_value, qnan_value, INVALID_EXCEPTION|ERRNO_EDOM),
    TEST_f_f (log10, minus_infty, qnan_value, INVALID_EXCEPTION|ERRNO_EDOM),

    TEST_f_f (log10, plus_infty, plus_infty, ERRNO_UNCHANGED),
    TEST_f_f (log10, qnan_value, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_f_f (log10, -qnan_value, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_f_f (log10, snan_value, qnan_value, INVALID_EXCEPTION),
    TEST_f_f (log10, -snan_value, qnan_value, INVALID_EXCEPTION),

    AUTO_TESTS_f_f (log10),
  };

static void
log10_test (void)
{
  ALL_RM_TEST (log10, 0, log10_test_data, RUN_TEST_LOOP_f_f, END);
}

static void
do_test (void)
{
  log10_test ();
}

/*
 * Local Variables:
 * mode:c
 * End:
 */
