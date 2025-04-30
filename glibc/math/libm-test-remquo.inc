/* Test remquo.
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

static const struct test_ffI_f1_data remquo_test_data[] =
  {
    /* Bug 6802: errno setting may be missing.  */
    TEST_ffI_f1 (remquo, 1, 0, qnan_value, IGNORE, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ffI_f1 (remquo, 1, minus_zero, qnan_value, IGNORE, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ffI_f1 (remquo, plus_zero, plus_zero, qnan_value, IGNORE, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ffI_f1 (remquo, plus_zero, minus_zero, qnan_value, IGNORE, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ffI_f1 (remquo, minus_zero, plus_zero, qnan_value, IGNORE, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ffI_f1 (remquo, minus_zero, minus_zero, qnan_value, IGNORE, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ffI_f1 (remquo, plus_infty, minus_zero, qnan_value, IGNORE, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ffI_f1 (remquo, plus_infty, 0, qnan_value, IGNORE, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ffI_f1 (remquo, plus_infty, 1, qnan_value, IGNORE, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ffI_f1 (remquo, plus_infty, plus_infty, qnan_value, IGNORE, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ffI_f1 (remquo, plus_infty, minus_infty, qnan_value, IGNORE, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ffI_f1 (remquo, minus_infty, minus_zero, qnan_value, IGNORE, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ffI_f1 (remquo, minus_infty, 0, qnan_value, IGNORE, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ffI_f1 (remquo, minus_infty, 1, qnan_value, IGNORE, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ffI_f1 (remquo, minus_infty, plus_infty, qnan_value, IGNORE, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ffI_f1 (remquo, minus_infty, minus_infty, qnan_value, IGNORE, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ffI_f1 (remquo, qnan_value, qnan_value, qnan_value, IGNORE, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ffI_f1 (remquo, qnan_value, -qnan_value, qnan_value, IGNORE, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ffI_f1 (remquo, -qnan_value, qnan_value, qnan_value, IGNORE, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ffI_f1 (remquo, -qnan_value, -qnan_value, qnan_value, IGNORE, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ffI_f1 (remquo, snan_value, qnan_value, qnan_value, IGNORE, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ffI_f1 (remquo, snan_value, -qnan_value, qnan_value, IGNORE, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ffI_f1 (remquo, -snan_value, qnan_value, qnan_value, IGNORE, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ffI_f1 (remquo, -snan_value, -qnan_value, qnan_value, IGNORE, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ffI_f1 (remquo, qnan_value, snan_value, qnan_value, IGNORE, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ffI_f1 (remquo, qnan_value, -snan_value, qnan_value, IGNORE, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ffI_f1 (remquo, -qnan_value, snan_value, qnan_value, IGNORE, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ffI_f1 (remquo, -qnan_value, -snan_value, qnan_value, IGNORE, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ffI_f1 (remquo, snan_value, snan_value, qnan_value, IGNORE, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ffI_f1 (remquo, snan_value, -snan_value, qnan_value, IGNORE, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ffI_f1 (remquo, -snan_value, snan_value, qnan_value, IGNORE, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ffI_f1 (remquo, -snan_value, -snan_value, qnan_value, IGNORE, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ffI_f1 (remquo, 0, qnan_value, qnan_value, IGNORE, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ffI_f1 (remquo, 0, -qnan_value, qnan_value, IGNORE, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ffI_f1 (remquo, 0, snan_value, qnan_value, IGNORE, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ffI_f1 (remquo, 0, -snan_value, qnan_value, IGNORE, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ffI_f1 (remquo, qnan_value, 0, qnan_value, IGNORE, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ffI_f1 (remquo, -qnan_value, 0, qnan_value, IGNORE, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ffI_f1 (remquo, snan_value, 0, qnan_value, IGNORE, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ffI_f1 (remquo, -snan_value, 0, qnan_value, IGNORE, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ffI_f1 (remquo, 1, qnan_value, qnan_value, IGNORE, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ffI_f1 (remquo, 1, -qnan_value, qnan_value, IGNORE, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ffI_f1 (remquo, 1, snan_value, qnan_value, IGNORE, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ffI_f1 (remquo, 1, -snan_value, qnan_value, IGNORE, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ffI_f1 (remquo, qnan_value, 1, qnan_value, IGNORE, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ffI_f1 (remquo, -qnan_value, 1, qnan_value, IGNORE, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ffI_f1 (remquo, snan_value, 1, qnan_value, IGNORE, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ffI_f1 (remquo, -snan_value, 1, qnan_value, IGNORE, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ffI_f1 (remquo, plus_infty, qnan_value, qnan_value, IGNORE, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ffI_f1 (remquo, plus_infty, -qnan_value, qnan_value, IGNORE, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ffI_f1 (remquo, plus_infty, snan_value, qnan_value, IGNORE, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ffI_f1 (remquo, plus_infty, -snan_value, qnan_value, IGNORE, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ffI_f1 (remquo, qnan_value, plus_infty, qnan_value, IGNORE, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ffI_f1 (remquo, -qnan_value, plus_infty, qnan_value, IGNORE, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ffI_f1 (remquo, snan_value, plus_infty, qnan_value, IGNORE, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ffI_f1 (remquo, -snan_value, plus_infty, qnan_value, IGNORE, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ffI_f1 (remquo, minus_infty, qnan_value, qnan_value, IGNORE, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ffI_f1 (remquo, minus_infty, -qnan_value, qnan_value, IGNORE, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ffI_f1 (remquo, minus_infty, snan_value, qnan_value, IGNORE, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ffI_f1 (remquo, minus_infty, -snan_value, qnan_value, IGNORE, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ffI_f1 (remquo, qnan_value, minus_infty, qnan_value, IGNORE, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ffI_f1 (remquo, -qnan_value, minus_infty, qnan_value, IGNORE, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ffI_f1 (remquo, snan_value, minus_infty, qnan_value, IGNORE, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ffI_f1 (remquo, -snan_value, minus_infty, qnan_value, IGNORE, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),

    TEST_ffI_f1 (remquo, 7.0, plus_infty, 7.0, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ffI_f1 (remquo, 7.0, minus_infty, 7.0, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),

    TEST_ffI_f1 (remquo, 1.625, 1.0, -0.375, 2, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ffI_f1 (remquo, -1.625, 1.0, 0.375, -2, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ffI_f1 (remquo, 1.625, -1.0, -0.375, -2, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ffI_f1 (remquo, -1.625, -1.0, 0.375, 2, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),

    TEST_ffI_f1 (remquo, 5, 2, 1, 2, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ffI_f1 (remquo, 3, 2, -1, 2, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
#if MANT_DIG >= 56
    TEST_ffI_f1 (remquo, -0x1.80000000000002p1L, 2.0, 0x1.fffffffffffff8p-1L, -2, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ffI_f1 (remquo, 0x0.ffffffffffffffp0L, -0x1.00000000000001p0L, -0x1p-55L, -1, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
#endif

    TEST_ffI_f1 (remquo, 3419, 360, 179, 1, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ffI_f1 (remquo, -3419, 360, -179, -1, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ffI_f1 (remquo, 3419, -360, 179, -1, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ffI_f1 (remquo, -3419, -360, -179, 1, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),

    TEST_ffI_f1 (remquo, max_value, max_value, plus_zero, 1, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ffI_f1 (remquo, max_value, -max_value, plus_zero, -1, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ffI_f1 (remquo, max_value, min_value, plus_zero, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ffI_f1 (remquo, max_value, -min_value, plus_zero, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ffI_f1 (remquo, max_value, min_subnorm_value, plus_zero, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ffI_f1 (remquo, max_value, -min_subnorm_value, plus_zero, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ffI_f1 (remquo, -max_value, max_value, minus_zero, -1, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ffI_f1 (remquo, -max_value, -max_value, minus_zero, 1, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ffI_f1 (remquo, -max_value, min_value, minus_zero, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ffI_f1 (remquo, -max_value, -min_value, minus_zero, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ffI_f1 (remquo, -max_value, min_subnorm_value, minus_zero, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ffI_f1 (remquo, -max_value, -min_subnorm_value, minus_zero, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ffI_f1 (remquo, min_value, max_value, min_value, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ffI_f1 (remquo, min_value, -max_value, min_value, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ffI_f1 (remquo, min_value, min_value, plus_zero, 1, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ffI_f1 (remquo, min_value, -min_value, plus_zero, -1, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ffI_f1 (remquo, min_value, min_subnorm_value, plus_zero, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ffI_f1 (remquo, min_value, -min_subnorm_value, plus_zero, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ffI_f1 (remquo, -min_value, max_value, -min_value, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ffI_f1 (remquo, -min_value, -max_value, -min_value, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ffI_f1 (remquo, -min_value, min_value, minus_zero, -1, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ffI_f1 (remquo, -min_value, -min_value, minus_zero, 1, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ffI_f1 (remquo, -min_value, min_subnorm_value, minus_zero, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ffI_f1 (remquo, -min_value, -min_subnorm_value, minus_zero, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ffI_f1 (remquo, min_subnorm_value, max_value, min_subnorm_value, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ffI_f1 (remquo, min_subnorm_value, -max_value, min_subnorm_value, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ffI_f1 (remquo, min_subnorm_value, min_value, min_subnorm_value, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ffI_f1 (remquo, min_subnorm_value, -min_value, min_subnorm_value, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ffI_f1 (remquo, min_subnorm_value, min_subnorm_value, plus_zero, 1, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ffI_f1 (remquo, min_subnorm_value, -min_subnorm_value, plus_zero, -1, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ffI_f1 (remquo, -min_subnorm_value, max_value, -min_subnorm_value, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ffI_f1 (remquo, -min_subnorm_value, -max_value, -min_subnorm_value, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ffI_f1 (remquo, -min_subnorm_value, min_value, -min_subnorm_value, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ffI_f1 (remquo, -min_subnorm_value, -min_value, -min_subnorm_value, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ffI_f1 (remquo, -min_subnorm_value, min_subnorm_value, minus_zero, -1, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ffI_f1 (remquo, -min_subnorm_value, -min_subnorm_value, minus_zero, 1, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),

    TEST_ffI_f1 (remquo, 1, max_value, 1, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ffI_f1 (remquo, 1, -max_value, 1, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ffI_f1 (remquo, 1, max_value / 2, 1, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED|XFAIL_ROUNDING_IBM128_LIBGCC),
    TEST_ffI_f1 (remquo, 1, -max_value / 2, 1, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED|XFAIL_ROUNDING_IBM128_LIBGCC),
    TEST_ffI_f1 (remquo, 1, max_value / 4, 1, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED|XFAIL_ROUNDING_IBM128_LIBGCC),
    TEST_ffI_f1 (remquo, 1, -max_value / 4, 1, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED|XFAIL_ROUNDING_IBM128_LIBGCC),
    TEST_ffI_f1 (remquo, 1, max_value / 8, 1, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED|XFAIL_ROUNDING_IBM128_LIBGCC),
    TEST_ffI_f1 (remquo, 1, -max_value / 8, 1, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED|XFAIL_ROUNDING_IBM128_LIBGCC),
    TEST_ffI_f1 (remquo, -1, max_value, -1, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ffI_f1 (remquo, -1, -max_value, -1, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ffI_f1 (remquo, -1, max_value / 2, -1, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED|XFAIL_ROUNDING_IBM128_LIBGCC),
    TEST_ffI_f1 (remquo, -1, -max_value / 2, -1, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED|XFAIL_ROUNDING_IBM128_LIBGCC),
    TEST_ffI_f1 (remquo, -1, max_value / 4, -1, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED|XFAIL_ROUNDING_IBM128_LIBGCC),
    TEST_ffI_f1 (remquo, -1, -max_value / 4, -1, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED|XFAIL_ROUNDING_IBM128_LIBGCC),
    TEST_ffI_f1 (remquo, -1, max_value / 8, -1, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED|XFAIL_ROUNDING_IBM128_LIBGCC),
    TEST_ffI_f1 (remquo, -1, -max_value / 8, -1, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED|XFAIL_ROUNDING_IBM128_LIBGCC),

    TEST_ffI_f1 (remquo, max_value, max_value / 2, plus_zero, 2, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED|XFAIL_ROUNDING_IBM128_LIBGCC),
    TEST_ffI_f1 (remquo, max_value, -max_value / 2, plus_zero, -2, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED|XFAIL_ROUNDING_IBM128_LIBGCC),
    TEST_ffI_f1 (remquo, -max_value, max_value / 2, minus_zero, -2, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED|XFAIL_ROUNDING_IBM128_LIBGCC),
    TEST_ffI_f1 (remquo, -max_value, -max_value / 2, minus_zero, 2, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED|XFAIL_ROUNDING_IBM128_LIBGCC),

    TEST_ffI_f1 (remquo, 2, 1, plus_zero, 2, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ffI_f1 (remquo, 2, -1, plus_zero, -2, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ffI_f1 (remquo, -2, 1, minus_zero, -2, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ffI_f1 (remquo, -2, -1, minus_zero, 2, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
  };

static void
remquo_test (void)
{
  int x;

  ALL_RM_TEST (remquo, 1, remquo_test_data, RUN_TEST_LOOP_ffI_f1_mod8, END, x);
}

static void
do_test (void)
{
  remquo_test ();
}

/*
 * Local Variables:
 * mode:c
 * End:
 */
