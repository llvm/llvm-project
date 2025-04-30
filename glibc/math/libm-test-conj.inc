/* Test conj.
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

static const struct test_c_c_data conj_test_data[] =
  {
    TEST_c_c (conj, 0.0, 0.0, 0.0, minus_zero, NO_INEXACT_EXCEPTION),
    TEST_c_c (conj, 0.0, minus_zero, 0.0, 0.0, NO_INEXACT_EXCEPTION),
    TEST_c_c (conj, qnan_value, qnan_value, qnan_value, qnan_value, NO_INEXACT_EXCEPTION),
    TEST_c_c (conj, qnan_value, snan_value, qnan_value, snan_value, NO_INEXACT_EXCEPTION),
    TEST_c_c (conj, snan_value, qnan_value, snan_value, qnan_value, NO_INEXACT_EXCEPTION),
    TEST_c_c (conj, snan_value, snan_value, snan_value, snan_value, NO_INEXACT_EXCEPTION),
    TEST_c_c (conj, plus_infty, minus_infty, plus_infty, plus_infty, NO_INEXACT_EXCEPTION),
    TEST_c_c (conj, plus_infty, plus_infty, plus_infty, minus_infty, NO_INEXACT_EXCEPTION),
    TEST_c_c (conj, 1.0, 2.0, 1.0, -2.0, NO_INEXACT_EXCEPTION),
    TEST_c_c (conj, 3.0, -4.0, 3.0, 4.0, NO_INEXACT_EXCEPTION),
  };

static void
conj_test (void)
{
  ALL_RM_TEST (conj, 1, conj_test_data, RUN_TEST_LOOP_c_c, END_COMPLEX);
}

static void
do_test (void)
{
  conj_test ();
}

/*
 * Local Variables:
 * mode:c
 * End:
 */
