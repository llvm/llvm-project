/* Test setpayloadsig.
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

static const struct test_Ff_b1_data setpayloadsig_test_data[] =
  {
#if HIGH_ORDER_BIT_IS_SET_FOR_SNAN
    TEST_Ff_b1 (setpayloadsig, plus_zero, 0, snan_value_pl ("0x0"), NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED|TEST_NAN_SIGN|TEST_NAN_PAYLOAD),
#else
    TEST_Ff_b1 (setpayloadsig, plus_zero, 1, plus_zero, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
#endif
    TEST_Ff_b1 (setpayloadsig, 0x1p0, 0, snan_value_pl ("0x1"), NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED|TEST_NAN_SIGN|TEST_NAN_PAYLOAD),
    TEST_Ff_b1 (setpayloadsig, 0x2p0, 0, snan_value_pl ("0x2"), NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED|TEST_NAN_SIGN|TEST_NAN_PAYLOAD),
    TEST_Ff_b1 (setpayloadsig, 0x3fffffp0, 0, snan_value_pl ("0x3fffff"), NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED|TEST_NAN_SIGN|TEST_NAN_PAYLOAD),
#if PAYLOAD_DIG >= 51
    TEST_Ff_b1 (setpayloadsig, 0x7ffffffffffffp0, 0, snan_value_pl ("0x7ffffffffffff"), NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED|TEST_NAN_SIGN|TEST_NAN_PAYLOAD),
#else
    TEST_Ff_b1 (setpayloadsig, 0x7ffffffffffffp0, 1, plus_zero, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
#endif
#if PAYLOAD_DIG >= 62
    TEST_Ff_b1 (setpayloadsig, 0x3fffffffffffffffp0, 0, snan_value_pl ("0x3fffffffffffffff"), NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED|TEST_NAN_SIGN|TEST_NAN_PAYLOAD),
#else
    TEST_Ff_b1 (setpayloadsig, 0x3fffffffffffffffp0, 1, plus_zero, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
#endif
#if PAYLOAD_DIG >= 111
    TEST_Ff_b1 (setpayloadsig, 0x7fffffffffffffffffffffffffffp0, 0, snan_value_pl ("0x7fffffffffffffffffffffffffff"), NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED|TEST_NAN_SIGN|TEST_NAN_PAYLOAD),
#else
    TEST_Ff_b1 (setpayloadsig, 0x7fffffffffffffffffffffffffffp0, 1, plus_zero, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
#endif
    TEST_Ff_b1 (setpayloadsig, minus_zero, 1, plus_zero, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_Ff_b1 (setpayloadsig, -1.0, 1, plus_zero, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_Ff_b1 (setpayloadsig, -2.0, 1, plus_zero, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_Ff_b1 (setpayloadsig, 0.5, 1, plus_zero, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_Ff_b1 (setpayloadsig, -0.5, 1, plus_zero, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_Ff_b1 (setpayloadsig, max_value, 1, plus_zero, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_Ff_b1 (setpayloadsig, -max_value, 1, plus_zero, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_Ff_b1 (setpayloadsig, min_value, 1, plus_zero, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_Ff_b1 (setpayloadsig, -min_value, 1, plus_zero, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_Ff_b1 (setpayloadsig, min_subnorm_value, 1, plus_zero, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_Ff_b1 (setpayloadsig, -min_subnorm_value, 1, plus_zero, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_Ff_b1 (setpayloadsig, plus_infty, 1, plus_zero, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_Ff_b1 (setpayloadsig, minus_infty, 1, plus_zero, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_Ff_b1 (setpayloadsig, qnan_value, 1, plus_zero, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_Ff_b1 (setpayloadsig, -qnan_value, 1, plus_zero, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_Ff_b1 (setpayloadsig, snan_value, 1, plus_zero, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_Ff_b1 (setpayloadsig, -snan_value, 1, plus_zero, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_Ff_b1 (setpayloadsig, 0xffffffp-1, 1, plus_zero, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
#if MANT_DIG >= 53
    TEST_Ff_b1 (setpayloadsig, 0x1fffffffffffffp-1, 1, plus_zero, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
#endif
#if MANT_DIG >= 64
    TEST_Ff_b1 (setpayloadsig, 0xffffffffffffffffp-1, 1, plus_zero, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
#endif
#if MANT_DIG >= 106
    TEST_Ff_b1 (setpayloadsig, 0x3ffffffffffffffffffffffffffp-1, 1, plus_zero, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
#endif
#if MANT_DIG >= 113
    TEST_Ff_b1 (setpayloadsig, 0x1ffffffffffffffffffffffffffffp-1, 1, plus_zero, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
#endif
#if PAYLOAD_DIG >= 23
    TEST_Ff_b1 (setpayloadsig, 0x1p22, 0, snan_value_pl ("0x400000"), NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED|TEST_NAN_SIGN|TEST_NAN_PAYLOAD),
#else
    TEST_Ff_b1 (setpayloadsig, 0x1p22, 1, plus_zero, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
#endif
#if PAYLOAD_DIG >= 52
    TEST_Ff_b1 (setpayloadsig, 0x1p51, 0, snan_value_pl ("0x8000000000000"), NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED|TEST_NAN_SIGN|TEST_NAN_PAYLOAD),
#else
    TEST_Ff_b1 (setpayloadsig, 0x1p51, 1, plus_zero, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
#endif
#if PAYLOAD_DIG >= 63
    TEST_Ff_b1 (setpayloadsig, 0x1p62, 0, snan_value_pl ("0x4000000000000000"), NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED|TEST_NAN_SIGN|TEST_NAN_PAYLOAD),
#else
    TEST_Ff_b1 (setpayloadsig, 0x1p62, 1, plus_zero, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
#endif
    TEST_Ff_b1 (setpayloadsig, 0x1p111, 1, plus_zero, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
  };

static void
setpayloadsig_test (void)
{
  FLOAT x;

  ALL_RM_TEST (setpayloadsig, 1, setpayloadsig_test_data, RUN_TEST_LOOP_Ff_b1, END, x);
}

static void
do_test (void)
{
  setpayloadsig_test ();
}

/*
 * Local Variables:
 * mode:c
 * End:
 */
