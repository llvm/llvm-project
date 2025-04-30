/* Test copysign.
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

static const struct test_ff_f_data copysign_test_data[] =
  {
    TEST_ff_f (copysign, 0, 0, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (copysign, 0, minus_zero, minus_zero, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (copysign, minus_zero, 0, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (copysign, minus_zero, minus_zero, minus_zero, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),

    TEST_ff_f (copysign, 0, 4, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (copysign, 0, -4, minus_zero, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (copysign, minus_zero, 4, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (copysign, minus_zero, -4, minus_zero, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),

    TEST_ff_f (copysign, 2, 0, 2, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (copysign, 2, minus_zero, -2, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (copysign, -2, 0, 2, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (copysign, -2, minus_zero, -2, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),

    TEST_ff_f (copysign, plus_infty, 0, plus_infty, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (copysign, plus_infty, minus_zero, minus_infty, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (copysign, minus_infty, 0, plus_infty, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (copysign, minus_infty, minus_zero, minus_infty, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),

    TEST_ff_f (copysign, 0, plus_infty, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (copysign, 0, minus_infty, minus_zero, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (copysign, minus_zero, plus_infty, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (copysign, minus_zero, minus_infty, minus_zero, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),

    TEST_ff_f (copysign, 0, qnan_value, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (copysign, 0, -qnan_value, minus_zero, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (copysign, minus_zero, qnan_value, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (copysign, minus_zero, -qnan_value, minus_zero, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),

    TEST_ff_f (copysign, qnan_value, 0, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED|TEST_NAN_SIGN),
    TEST_ff_f (copysign, qnan_value, minus_zero, -qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED|TEST_NAN_SIGN),
    TEST_ff_f (copysign, -qnan_value, 0, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED|TEST_NAN_SIGN),
    TEST_ff_f (copysign, -qnan_value, minus_zero, -qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED|TEST_NAN_SIGN),

    TEST_ff_f (copysign, qnan_value, qnan_value, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED|TEST_NAN_SIGN),
    TEST_ff_f (copysign, qnan_value, -qnan_value, -qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED|TEST_NAN_SIGN),
    TEST_ff_f (copysign, -qnan_value, qnan_value, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED|TEST_NAN_SIGN),
    TEST_ff_f (copysign, -qnan_value, -qnan_value, -qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED|TEST_NAN_SIGN),

    TEST_ff_f (copysign, 0, snan_value, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (copysign, 0, -snan_value, minus_zero, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (copysign, minus_zero, snan_value, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (copysign, minus_zero, -snan_value, minus_zero, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),

    TEST_ff_f (copysign, snan_value, 0, snan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED|TEST_NAN_SIGN),
    TEST_ff_f (copysign, snan_value, minus_zero, -snan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED|TEST_NAN_SIGN),
    TEST_ff_f (copysign, -snan_value, 0, snan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED|TEST_NAN_SIGN),
    TEST_ff_f (copysign, -snan_value, minus_zero, -snan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED|TEST_NAN_SIGN),

    TEST_ff_f (copysign, qnan_value, snan_value, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED|TEST_NAN_SIGN),
    TEST_ff_f (copysign, qnan_value, -snan_value, -qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED|TEST_NAN_SIGN),
    TEST_ff_f (copysign, -qnan_value, snan_value, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED|TEST_NAN_SIGN),
    TEST_ff_f (copysign, -qnan_value, -snan_value, -qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED|TEST_NAN_SIGN),

    TEST_ff_f (copysign, snan_value, qnan_value, snan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED|TEST_NAN_SIGN),
    TEST_ff_f (copysign, snan_value, -qnan_value, -snan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED|TEST_NAN_SIGN),
    TEST_ff_f (copysign, -snan_value, qnan_value, snan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED|TEST_NAN_SIGN),
    TEST_ff_f (copysign, -snan_value, -qnan_value, -snan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED|TEST_NAN_SIGN),

    TEST_ff_f (copysign, snan_value, snan_value, snan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED|TEST_NAN_SIGN),
    TEST_ff_f (copysign, snan_value, -snan_value, -snan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED|TEST_NAN_SIGN),
    TEST_ff_f (copysign, -snan_value, snan_value, snan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED|TEST_NAN_SIGN),
    TEST_ff_f (copysign, -snan_value, -snan_value, -snan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED|TEST_NAN_SIGN),

    TEST_ff_f (copysign, min_value, min_subnorm_value, min_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (copysign, min_value, -min_subnorm_value, -min_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (copysign, -min_value, min_subnorm_value, min_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (copysign, -min_value, -min_subnorm_value, -min_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),

    TEST_ff_f (copysign, min_subnorm_value, max_value, min_subnorm_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (copysign, min_subnorm_value, -max_value, -min_subnorm_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (copysign, -min_subnorm_value, max_value, min_subnorm_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (copysign, -min_subnorm_value, -max_value, -min_subnorm_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),

    TEST_ff_f (copysign, max_value, min_value, max_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (copysign, max_value, -min_value, -max_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (copysign, -max_value, min_value, max_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (copysign, -max_value, -min_value, -max_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
  };

static void
copysign_test (void)
{
  ALL_RM_TEST (copysign, 1, copysign_test_data, RUN_TEST_LOOP_ff_f, END);
}

static void
do_test (void)
{
  copysign_test ();
}

/*
 * Local Variables:
 * mode:c
 * End:
 */
