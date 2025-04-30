/* Test llogb.
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

static const struct test_f_l_data llogb_test_data[] =
  {
    TEST_f_l (llogb, 1, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_f_l (llogb, lit_e, 1, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_f_l (llogb, 1024, 10, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_f_l (llogb, -2000, 10, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_f_l (llogb, 0.5, -1, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_f_l (llogb, -0.5, -1, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),

#if MANT_DIG >= 57
    TEST_f_l (llogb, 0x1.ffffffffffffffp1L, 1, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_f_l (llogb, -0x1.ffffffffffffffp1L, 1, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
#endif

    TEST_f_l (llogb, min_subnorm_value, MIN_EXP-MANT_DIG, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_f_l (llogb, -min_subnorm_value, MIN_EXP-MANT_DIG, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_f_l (llogb, min_value, MIN_EXP-1, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_f_l (llogb, -min_value, MIN_EXP-1, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_f_l (llogb, max_value, MAX_EXP-1, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_f_l (llogb, -max_value, MAX_EXP-1, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),

    /* llogb (0.0) == FP_LLOGB0 plus invalid exception  */
    TEST_f_l (llogb, 0.0, FP_LLOGB0, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION|ERRNO_EDOM),
    TEST_f_l (llogb, -0.0, FP_LLOGB0, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION|ERRNO_EDOM),
    /* llogb (qNaN) == FP_LLOGBNAN plus invalid exception  */
    TEST_f_l (llogb, qnan_value, FP_LLOGBNAN, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION|ERRNO_EDOM),
    TEST_f_l (llogb, -qnan_value, FP_LLOGBNAN, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION|ERRNO_EDOM),
    TEST_f_l (llogb, snan_value, FP_LLOGBNAN, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION|ERRNO_EDOM),
    TEST_f_l (llogb, -snan_value, FP_LLOGBNAN, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION|ERRNO_EDOM),
    /* llogb (inf) == LONG_MAX plus invalid exception  */
    TEST_f_l (llogb, plus_infty, LONG_MAX, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION|ERRNO_EDOM),
    /* llogb (-inf) == LONG_MAX plus invalid exception  */
    TEST_f_l (llogb, minus_infty, LONG_MAX, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION|ERRNO_EDOM),
  };

static void
llogb_test (void)
{
  ALL_RM_TEST (llogb, 1, llogb_test_data, RUN_TEST_LOOP_f_l, END);
}

static void
do_test (void)
{
  llogb_test ();
}

/*
 * Local Variables:
 * mode:c
 * End:
 */
