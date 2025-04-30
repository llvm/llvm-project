/* Test sincos.
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

static const struct test_fFF_11_data sincos_test_data[] =
  {
    TEST_fFF_11 (sincos, plus_infty, qnan_value, qnan_value, INVALID_EXCEPTION|ERRNO_EDOM),
    TEST_fFF_11 (sincos, minus_infty, qnan_value, qnan_value, INVALID_EXCEPTION|ERRNO_EDOM),
    TEST_fFF_11 (sincos, qnan_value, qnan_value, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_fFF_11 (sincos, -qnan_value, qnan_value, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_fFF_11 (sincos, snan_value, qnan_value, qnan_value, INVALID_EXCEPTION),
    TEST_fFF_11 (sincos, -snan_value, qnan_value, qnan_value, INVALID_EXCEPTION),

    AUTO_TESTS_fFF_11 (sincos),
  };

static void
sincos_test (void)
{
  FLOAT sin_res, cos_res;

  ALL_RM_TEST (sincos, 0, sincos_test_data, RUN_TEST_LOOP_fFF_11, END, sin_res, cos_res);
}

static void
do_test (void)
{
  sincos_test ();
}

/*
 * Local Variables:
 * mode:c
 * End:
 */
