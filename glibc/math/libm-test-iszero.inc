/* Test iszero.
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

static const struct test_f_i_data iszero_test_data[] =
  {
    TEST_f_b (iszero, 0, 1, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_f_b (iszero, minus_zero, 1, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_f_b (iszero, 10, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_f_b (iszero, min_subnorm_value, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_f_b (iszero, -min_subnorm_value, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_f_b (iszero, min_value, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_f_b (iszero, -min_value, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_f_b (iszero, max_value, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_f_b (iszero, -max_value, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_f_b (iszero, plus_infty, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_f_b (iszero, minus_infty, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_f_b (iszero, qnan_value, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_f_b (iszero, -qnan_value, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_f_b (iszero, snan_value, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_f_b (iszero, -snan_value, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
  };

static void
iszero_test (void)
{
  ALL_RM_TEST (iszero, 1, iszero_test_data, RUN_TEST_LOOP_f_b_tg, END);
}

static void
do_test (void)
{
  iszero_test ();
}

/*
 * Local Variables:
 * mode:c
 * End:
 */
