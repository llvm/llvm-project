/* Test fabs.
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

static const struct test_f_f_data fabs_test_data[] =
  {
    TEST_f_f (fabs, 0, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_f_f (fabs, minus_zero, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),

    TEST_f_f (fabs, plus_infty, plus_infty, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_f_f (fabs, minus_infty, plus_infty, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_f_f (fabs, qnan_value, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED|TEST_NAN_SIGN),
    TEST_f_f (fabs, -qnan_value, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED|TEST_NAN_SIGN),
    TEST_f_f (fabs, snan_value, snan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED|TEST_NAN_SIGN),
    TEST_f_f (fabs, -snan_value, snan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED|TEST_NAN_SIGN),

    TEST_f_f (fabs, min_subnorm_value, min_subnorm_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_f_f (fabs, min_value, min_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_f_f (fabs, max_value, max_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_f_f (fabs, -min_subnorm_value, min_subnorm_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_f_f (fabs, -min_value, min_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_f_f (fabs, -max_value, max_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_f_f (fabs, 38.0, 38.0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_f_f (fabs, -lit_e, lit_e, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
  };

static void
fabs_test (void)
{
  ALL_RM_TEST (fabs, 1, fabs_test_data, RUN_TEST_LOOP_f_f, END);
}

static void
do_test (void)
{
  fabs_test ();
}

/*
 * Local Variables:
 * mode:c
 * End:
 */
