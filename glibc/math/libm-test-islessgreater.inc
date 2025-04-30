/* Test islessgreater.
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

static const struct test_ff_i_data islessgreater_test_data[] =
  {
    TEST_ff_i (islessgreater, minus_zero, minus_zero, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_i (islessgreater, minus_zero, plus_zero, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_i (islessgreater, minus_zero, (FLOAT) 1, 1, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_i (islessgreater, minus_zero, qnan_value, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_i (islessgreater, minus_zero, -qnan_value, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_i (islessgreater, minus_zero, snan_value, 0, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_i (islessgreater, minus_zero, -snan_value, 0, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_i (islessgreater, plus_zero, minus_zero, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_i (islessgreater, plus_zero, plus_zero, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_i (islessgreater, plus_zero, (FLOAT) 1, 1, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_i (islessgreater, plus_zero, qnan_value, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_i (islessgreater, plus_zero, -qnan_value, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_i (islessgreater, plus_zero, snan_value, 0, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_i (islessgreater, plus_zero, -snan_value, 0, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_i (islessgreater, (FLOAT) 1, minus_zero, 1, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_i (islessgreater, (FLOAT) 1, plus_zero, 1, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_i (islessgreater, (FLOAT) 1, (FLOAT) 1, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_i (islessgreater, (FLOAT) 1, qnan_value, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_i (islessgreater, (FLOAT) 1, -qnan_value, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_i (islessgreater, (FLOAT) 1, snan_value, 0, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_i (islessgreater, (FLOAT) 1, -snan_value, 0, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_i (islessgreater, qnan_value, minus_zero, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_i (islessgreater, -qnan_value, minus_zero, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_i (islessgreater, qnan_value, plus_zero, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_i (islessgreater, -qnan_value, plus_zero, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_i (islessgreater, qnan_value, (FLOAT) 1, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_i (islessgreater, -qnan_value, (FLOAT) 1, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_i (islessgreater, snan_value, minus_zero, 0, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_i (islessgreater, -snan_value, minus_zero, 0, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_i (islessgreater, snan_value, plus_zero, 0, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_i (islessgreater, -snan_value, plus_zero, 0, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_i (islessgreater, snan_value, (FLOAT) 1, 0, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_i (islessgreater, -snan_value, (FLOAT) 1, 0, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_i (islessgreater, qnan_value, qnan_value, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_i (islessgreater, qnan_value, -qnan_value, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_i (islessgreater, -qnan_value, qnan_value, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_i (islessgreater, -qnan_value, -qnan_value, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_i (islessgreater, snan_value, qnan_value, 0, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_i (islessgreater, snan_value, -qnan_value, 0, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_i (islessgreater, -snan_value, qnan_value, 0, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_i (islessgreater, -snan_value, -qnan_value, 0, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_i (islessgreater, qnan_value, snan_value, 0, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_i (islessgreater, qnan_value, -snan_value, 0, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_i (islessgreater, -qnan_value, snan_value, 0, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_i (islessgreater, -qnan_value, -snan_value, 0, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_i (islessgreater, snan_value, snan_value, 0, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_i (islessgreater, snan_value, -snan_value, 0, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_i (islessgreater, -snan_value, snan_value, 0, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_i (islessgreater, -snan_value, -snan_value, 0, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
  };

static void
islessgreater_test (void)
{
  ALL_RM_TEST (islessgreater, 1, islessgreater_test_data, RUN_TEST_LOOP_ff_i_tg, END);
}

static void
do_test (void)
{
  islessgreater_test ();
}

/*
 * Local Variables:
 * mode:c
 * End:
 */
