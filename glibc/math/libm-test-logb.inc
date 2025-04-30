/* Test logb.
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

static const struct test_f_f_data logb_test_data[] =
  {
    TEST_f_f (logb, plus_infty, plus_infty, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_f_f (logb, minus_infty, plus_infty, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),

    /* Bug 6793: errno setting may be missing.  */
    TEST_f_f (logb, 0, minus_infty, NO_INEXACT_EXCEPTION|DIVIDE_BY_ZERO_EXCEPTION),

    TEST_f_f (logb, minus_zero, minus_infty, NO_INEXACT_EXCEPTION|DIVIDE_BY_ZERO_EXCEPTION),
    TEST_f_f (logb, qnan_value, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_f_f (logb, -qnan_value, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_f_f (logb, snan_value, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_f_f (logb, -snan_value, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),

    TEST_f_f (logb, 1, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_f_f (logb, lit_e, 1, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_f_f (logb, 1024, 10, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_f_f (logb, -2000, 10, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),

    TEST_f_f (logb, min_subnorm_value, MIN_EXP-MANT_DIG, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_f_f (logb, -min_subnorm_value, MIN_EXP-MANT_DIG, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_f_f (logb, min_value, MIN_EXP-1, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_f_f (logb, -min_value, MIN_EXP-1, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_f_f (logb, max_value, MAX_EXP-1, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_f_f (logb, -max_value, MAX_EXP-1, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),

    TEST_f_f (logb, 0x0.1p-127, -131, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_f_f (logb, 0x0.01p-127, -135, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_f_f (logb, 0x0.011p-127, -135, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
#if !TEST_COND_binary32
    TEST_f_f (logb, 0x0.8p-1022, -1023, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_f_f (logb, 0x0.1p-1022, -1026, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_f_f (logb, 0x0.00111p-1022, -1034, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_f_f (logb, 0x0.00001p-1022, -1042, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_f_f (logb, 0x0.000011p-1022, -1042, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_f_f (logb, 0x0.0000000000001p-1022, -1074, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
#endif
#if MIN_EXP - MANT_DIG <= -16400
    TEST_f_f (logb, 0x1p-16400L, -16400, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_f_f (logb, 0x.00000000001p-16382L, -16426, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
#endif

#if MANT_DIG >= 57
    TEST_f_f (logb, 0x1.ffffffffffffffp1L, 1, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_f_f (logb, -0x1.ffffffffffffffp1L, 1, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
#endif
  };

static void
logb_test (void)
{
  ALL_RM_TEST (logb, 1, logb_test_data, RUN_TEST_LOOP_f_f, END);
}

static void
do_test (void)
{
  logb_test ();
}

/*
 * Local Variables:
 * mode:c
 * End:
 */
