/* Test isgreater.
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

static const struct test_ff_i_data isgreater_test_data[] =
  {
    TEST_ff_i (isgreater, minus_zero, minus_zero, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_i (isgreater, minus_zero, plus_zero, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_i (isgreater, minus_zero, (FLOAT) 1, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_i (isgreater, minus_zero, qnan_value, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_i (isgreater, minus_zero, -qnan_value, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_i (isgreater, minus_zero, snan_value, 0, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_i (isgreater, minus_zero, -snan_value, 0, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_i (isgreater, plus_zero, minus_zero, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_i (isgreater, plus_zero, plus_zero, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_i (isgreater, plus_zero, (FLOAT) 1, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_i (isgreater, plus_zero, qnan_value, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_i (isgreater, plus_zero, -qnan_value, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_i (isgreater, plus_zero, snan_value, 0, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_i (isgreater, plus_zero, -snan_value, 0, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_i (isgreater, (FLOAT) 1, minus_zero, 1, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_i (isgreater, (FLOAT) 1, plus_zero, 1, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_i (isgreater, (FLOAT) 1, (FLOAT) 1, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_i (isgreater, (FLOAT) 1, qnan_value, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_i (isgreater, (FLOAT) 1, -qnan_value, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_i (isgreater, (FLOAT) 1, snan_value, 0, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_i (isgreater, (FLOAT) 1, -snan_value, 0, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_i (isgreater, qnan_value, minus_zero, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_i (isgreater, -qnan_value, minus_zero, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_i (isgreater, qnan_value, plus_zero, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_i (isgreater, -qnan_value, plus_zero, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_i (isgreater, qnan_value, (FLOAT) 1, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_i (isgreater, -qnan_value, (FLOAT) 1, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_i (isgreater, snan_value, minus_zero, 0, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_i (isgreater, -snan_value, minus_zero, 0, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_i (isgreater, snan_value, plus_zero, 0, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_i (isgreater, -snan_value, plus_zero, 0, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_i (isgreater, snan_value, (FLOAT) 1, 0, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_i (isgreater, -snan_value, (FLOAT) 1, 0, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_i (isgreater, qnan_value, qnan_value, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_i (isgreater, qnan_value, -qnan_value, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_i (isgreater, -qnan_value, qnan_value, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_i (isgreater, -qnan_value, -qnan_value, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_i (isgreater, snan_value, qnan_value, 0, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_i (isgreater, snan_value, -qnan_value, 0, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_i (isgreater, -snan_value, qnan_value, 0, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_i (isgreater, -snan_value, -qnan_value, 0, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_i (isgreater, qnan_value, snan_value, 0, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_i (isgreater, qnan_value, -snan_value, 0, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_i (isgreater, -qnan_value, snan_value, 0, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_i (isgreater, -qnan_value, -snan_value, 0, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_i (isgreater, snan_value, snan_value, 0, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_i (isgreater, snan_value, -snan_value, 0, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_i (isgreater, -snan_value, snan_value, 0, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_i (isgreater, -snan_value, -snan_value, 0, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
  };

static void
isgreater_test (void)
{
  ALL_RM_TEST (isgreater, 1, isgreater_test_data, RUN_TEST_LOOP_ff_i_tg, END);
}

static void
do_test (void)
{
  isgreater_test ();
}

/*
 * Local Variables:
 * mode:c
 * End:
 */
