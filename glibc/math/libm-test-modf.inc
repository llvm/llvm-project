/* Test modf.
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

static const struct test_fF_f1_data modf_test_data[] =
  {
    TEST_fF_f1 (modf, plus_infty, 0, plus_infty, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_fF_f1 (modf, minus_infty, minus_zero, minus_infty, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_fF_f1 (modf, qnan_value, qnan_value, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_fF_f1 (modf, -qnan_value, qnan_value, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_fF_f1 (modf, snan_value, qnan_value, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_fF_f1 (modf, -snan_value, qnan_value, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_fF_f1 (modf, 0, 0, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_fF_f1 (modf, minus_zero, minus_zero, minus_zero, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_fF_f1 (modf, min_value, min_value, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_fF_f1 (modf, -min_value, -min_value, minus_zero, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_fF_f1 (modf, min_subnorm_value, min_subnorm_value, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_fF_f1 (modf, -min_subnorm_value, -min_subnorm_value, minus_zero, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_fF_f1 (modf, max_value, 0, max_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED|XFAIL_ROUNDING_IBM128_LIBGCC),
    TEST_fF_f1 (modf, -max_value, minus_zero, -max_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED|XFAIL_ROUNDING_IBM128_LIBGCC),
    TEST_fF_f1 (modf, 1.5, 0.5, 1, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_fF_f1 (modf, 2.5, 0.5, 2, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_fF_f1 (modf, -2.5, -0.5, -2, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_fF_f1 (modf, 20, 0, 20, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_fF_f1 (modf, 21, 0, 21, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_fF_f1 (modf, -20, minus_zero, -20, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_fF_f1 (modf, -21, minus_zero, -21, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_fF_f1 (modf, 89.5, 0.5, 89, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
  };

static void
modf_test (void)
{
  FLOAT x;

  ALL_RM_TEST (modf, 1, modf_test_data, RUN_TEST_LOOP_fF_f1, END, x);
}

static void
do_test (void)
{
  modf_test ();
}

/*
 * Local Variables:
 * mode:c
 * End:
 */
