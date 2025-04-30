/* Test canonicalize.
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

static const struct test_Ff_b1_data canonicalize_test_data[] =
  {
    TEST_Ffp_b1 (canonicalize, plus_infty, 0, plus_infty, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_Ffp_b1 (canonicalize, minus_infty, 0, minus_infty, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_Ffp_b1 (canonicalize, plus_zero, 0, plus_zero, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_Ffp_b1 (canonicalize, minus_zero, 0, minus_zero, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_Ffp_b1 (canonicalize, 1000, 0, 1000, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_Ffp_b1 (canonicalize, max_value, 0, max_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_Ffp_b1 (canonicalize, -max_value, 0, -max_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_Ffp_b1 (canonicalize, min_value, 0, min_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_Ffp_b1 (canonicalize, -min_value, 0, -min_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_Ffp_b1 (canonicalize, min_subnorm_value, 0, min_subnorm_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_Ffp_b1 (canonicalize, -min_subnorm_value, 0, -min_subnorm_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_Ffp_b1 (canonicalize, qnan_value, 0, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED|TEST_NAN_SIGN|TEST_NAN_PAYLOAD),
    TEST_Ffp_b1 (canonicalize, -qnan_value, 0, -qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED|TEST_NAN_SIGN|TEST_NAN_PAYLOAD),
    TEST_Ffp_b1 (canonicalize, snan_value, 0, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION|ERRNO_UNCHANGED|TEST_NAN_SIGN),
    TEST_Ffp_b1 (canonicalize, -snan_value, 0, -qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION|ERRNO_UNCHANGED|TEST_NAN_SIGN),
#if HIGH_ORDER_BIT_IS_SET_FOR_SNAN
    TEST_Ffp_b1 (canonicalize, snan_value_pl ("0x0"), 0, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION|ERRNO_UNCHANGED|TEST_NAN_SIGN),
    TEST_Ffp_b1 (canonicalize, -snan_value_pl ("0x0"), 0, -qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION|ERRNO_UNCHANGED|TEST_NAN_SIGN),
#else
    TEST_Ffp_b1 (canonicalize, qnan_value_pl ("0x0"), 0, qnan_value_pl ("0x0"), NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED|TEST_NAN_SIGN|TEST_NAN_PAYLOAD),
    TEST_Ffp_b1 (canonicalize, -qnan_value_pl ("0x0"), 0, -qnan_value_pl ("0x0"), NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED|TEST_NAN_SIGN|TEST_NAN_PAYLOAD),
#endif
    TEST_Ffp_b1 (canonicalize, qnan_value_pl ("0x1"), 0, qnan_value_pl ("0x1"), NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED|TEST_NAN_SIGN|TEST_NAN_PAYLOAD),
    TEST_Ffp_b1 (canonicalize, -qnan_value_pl ("0x1"), 0, -qnan_value_pl ("0x1"), NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED|TEST_NAN_SIGN|TEST_NAN_PAYLOAD),
    TEST_Ffp_b1 (canonicalize, snan_value_pl ("0x1"), 0, qnan_value_pl ("0x1"), NO_INEXACT_EXCEPTION|INVALID_EXCEPTION|ERRNO_UNCHANGED|TEST_NAN_SIGN|TEST_NAN_PAYLOAD_CANONICALIZE),
    TEST_Ffp_b1 (canonicalize, -snan_value_pl ("0x1"), 0, -qnan_value_pl ("0x1"), NO_INEXACT_EXCEPTION|INVALID_EXCEPTION|ERRNO_UNCHANGED|TEST_NAN_SIGN|TEST_NAN_PAYLOAD_CANONICALIZE),
    TEST_Ffp_b1 (canonicalize, qnan_value_pl ("0x2"), 0, qnan_value_pl ("0x2"), NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED|TEST_NAN_SIGN|TEST_NAN_PAYLOAD),
    TEST_Ffp_b1 (canonicalize, -qnan_value_pl ("0x2"), 0, -qnan_value_pl ("0x2"), NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED|TEST_NAN_SIGN|TEST_NAN_PAYLOAD),
    TEST_Ffp_b1 (canonicalize, snan_value_pl ("0x2"), 0, qnan_value_pl ("0x2"), NO_INEXACT_EXCEPTION|INVALID_EXCEPTION|ERRNO_UNCHANGED|TEST_NAN_SIGN|TEST_NAN_PAYLOAD_CANONICALIZE),
    TEST_Ffp_b1 (canonicalize, -snan_value_pl ("0x2"), 0, -qnan_value_pl ("0x2"), NO_INEXACT_EXCEPTION|INVALID_EXCEPTION|ERRNO_UNCHANGED|TEST_NAN_SIGN|TEST_NAN_PAYLOAD_CANONICALIZE),
    TEST_Ffp_b1 (canonicalize, qnan_value_pl ("0x3fffff"), 0, qnan_value_pl ("0x3fffff"), NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED|TEST_NAN_SIGN|TEST_NAN_PAYLOAD),
    TEST_Ffp_b1 (canonicalize, -qnan_value_pl ("0x3fffff"), 0, -qnan_value_pl ("0x3fffff"), NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED|TEST_NAN_SIGN|TEST_NAN_PAYLOAD),
    TEST_Ffp_b1 (canonicalize, snan_value_pl ("0x3fffff"), 0, qnan_value_pl ("0x3fffff"), NO_INEXACT_EXCEPTION|INVALID_EXCEPTION|ERRNO_UNCHANGED|TEST_NAN_SIGN|TEST_NAN_PAYLOAD_CANONICALIZE),
    TEST_Ffp_b1 (canonicalize, -snan_value_pl ("0x3fffff"), 0, -qnan_value_pl ("0x3fffff"), NO_INEXACT_EXCEPTION|INVALID_EXCEPTION|ERRNO_UNCHANGED|TEST_NAN_SIGN|TEST_NAN_PAYLOAD_CANONICALIZE),
#if PAYLOAD_DIG >= 51
    TEST_Ffp_b1 (canonicalize, qnan_value_pl ("0x7ffffffffffff"), 0, qnan_value_pl ("0x7ffffffffffff"), NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED|TEST_NAN_SIGN|TEST_NAN_PAYLOAD),
    TEST_Ffp_b1 (canonicalize, -qnan_value_pl ("0x7ffffffffffff"), 0, -qnan_value_pl ("0x7ffffffffffff"), NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED|TEST_NAN_SIGN|TEST_NAN_PAYLOAD),
    TEST_Ffp_b1 (canonicalize, snan_value_pl ("0x7ffffffffffff"), 0, qnan_value_pl ("0x7ffffffffffff"), NO_INEXACT_EXCEPTION|INVALID_EXCEPTION|ERRNO_UNCHANGED|TEST_NAN_SIGN|TEST_NAN_PAYLOAD_CANONICALIZE),
    TEST_Ffp_b1 (canonicalize, -snan_value_pl ("0x7ffffffffffff"), 0, -qnan_value_pl ("0x7ffffffffffff"), NO_INEXACT_EXCEPTION|INVALID_EXCEPTION|ERRNO_UNCHANGED|TEST_NAN_SIGN|TEST_NAN_PAYLOAD_CANONICALIZE),
#endif
#if PAYLOAD_DIG >= 62
    TEST_Ffp_b1 (canonicalize, qnan_value_pl ("0x3fffffffffffffff"), 0, qnan_value_pl ("0x3fffffffffffffff"), NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED|TEST_NAN_SIGN|TEST_NAN_PAYLOAD),
    TEST_Ffp_b1 (canonicalize, -qnan_value_pl ("0x3fffffffffffffff"), 0, -qnan_value_pl ("0x3fffffffffffffff"), NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED|TEST_NAN_SIGN|TEST_NAN_PAYLOAD),
    TEST_Ffp_b1 (canonicalize, snan_value_pl ("0x3fffffffffffffff"), 0, qnan_value_pl ("0x3fffffffffffffff"), NO_INEXACT_EXCEPTION|INVALID_EXCEPTION|ERRNO_UNCHANGED|TEST_NAN_SIGN|TEST_NAN_PAYLOAD_CANONICALIZE),
    TEST_Ffp_b1 (canonicalize, -snan_value_pl ("0x3fffffffffffffff"), 0, -qnan_value_pl ("0x3fffffffffffffff"), NO_INEXACT_EXCEPTION|INVALID_EXCEPTION|ERRNO_UNCHANGED|TEST_NAN_SIGN|TEST_NAN_PAYLOAD_CANONICALIZE),
#endif
#if PAYLOAD_DIG >= 111
    TEST_Ffp_b1 (canonicalize, qnan_value_pl ("0x7fffffffffffffffffffffffffff"), 0, qnan_value_pl ("0x7fffffffffffffffffffffffffff"), NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED|TEST_NAN_SIGN|TEST_NAN_PAYLOAD),
    TEST_Ffp_b1 (canonicalize, -qnan_value_pl ("0x7fffffffffffffffffffffffffff"), 0, -qnan_value_pl ("0x7fffffffffffffffffffffffffff"), NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED|TEST_NAN_SIGN|TEST_NAN_PAYLOAD),
    TEST_Ffp_b1 (canonicalize, snan_value_pl ("0x7fffffffffffffffffffffffffff"), 0, qnan_value_pl ("0x7fffffffffffffffffffffffffff"), NO_INEXACT_EXCEPTION|INVALID_EXCEPTION|ERRNO_UNCHANGED|TEST_NAN_SIGN|TEST_NAN_PAYLOAD_CANONICALIZE),
    TEST_Ffp_b1 (canonicalize, -snan_value_pl ("0x7fffffffffffffffffffffffffff"), 0, -qnan_value_pl ("0x7fffffffffffffffffffffffffff"), NO_INEXACT_EXCEPTION|INVALID_EXCEPTION|ERRNO_UNCHANGED|TEST_NAN_SIGN|TEST_NAN_PAYLOAD_CANONICALIZE),
#endif
  };

static void
canonicalize_test (void)
{
  FLOAT x;

  ALL_RM_TEST (canonicalize, 1, canonicalize_test_data, RUN_TEST_LOOP_Ffp_b1, END, x);
}

static void
do_test (void)
{
  canonicalize_test ();
}

/*
 * Local Variables:
 * mode:c
 * End:
 */
