/* Test setpayload.
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

static const struct test_Ff_b1_data setpayload_test_data[] =
  {
#if HIGH_ORDER_BIT_IS_SET_FOR_SNAN
    TEST_Ff_b1 (setpayload, plus_zero, 1, plus_zero, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
#else
    TEST_Ff_b1 (setpayload, plus_zero, 0, qnan_value_pl ("0x0"), NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED|TEST_NAN_SIGN|TEST_NAN_PAYLOAD),
#endif
    TEST_Ff_b1 (setpayload, 0x1p0, 0, qnan_value_pl ("0x1"), NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED|TEST_NAN_SIGN|TEST_NAN_PAYLOAD|XFAIL_FLOAT128_PAYLOAD),
    TEST_Ff_b1 (setpayload, 0x2p0, 0, qnan_value_pl ("0x2"), NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED|TEST_NAN_SIGN|TEST_NAN_PAYLOAD|XFAIL_FLOAT128_PAYLOAD),
    TEST_Ff_b1 (setpayload, 0x3fffffp0, 0, qnan_value_pl ("0x3fffff"), NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED|TEST_NAN_SIGN|TEST_NAN_PAYLOAD|XFAIL_FLOAT128_PAYLOAD),
#if PAYLOAD_DIG >= 51
    TEST_Ff_b1 (setpayload, 0x7ffffffffffffp0, 0, qnan_value_pl ("0x7ffffffffffff"), NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED|TEST_NAN_SIGN|TEST_NAN_PAYLOAD|XFAIL_FLOAT128_PAYLOAD),
#else
    TEST_Ff_b1 (setpayload, 0x7ffffffffffffp0, 1, plus_zero, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
#endif
#if PAYLOAD_DIG >= 62
    TEST_Ff_b1 (setpayload, 0x3fffffffffffffffp0, 0, qnan_value_pl ("0x3fffffffffffffff"), NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED|TEST_NAN_SIGN|TEST_NAN_PAYLOAD|XFAIL_FLOAT128_PAYLOAD),
#else
    TEST_Ff_b1 (setpayload, 0x3fffffffffffffffp0, 1, plus_zero, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
#endif
#if PAYLOAD_DIG >= 111
    TEST_Ff_b1 (setpayload, 0x7fffffffffffffffffffffffffffp0, 0, qnan_value_pl ("0x7fffffffffffffffffffffffffff"), NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED|TEST_NAN_SIGN|TEST_NAN_PAYLOAD|XFAIL_FLOAT128_PAYLOAD),
#else
    TEST_Ff_b1 (setpayload, 0x7fffffffffffffffffffffffffffp0, 1, plus_zero, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
#endif
    TEST_Ff_b1 (setpayload, minus_zero, 1, plus_zero, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_Ff_b1 (setpayload, -1.0, 1, plus_zero, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_Ff_b1 (setpayload, -2.0, 1, plus_zero, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_Ff_b1 (setpayload, 0.5, 1, plus_zero, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_Ff_b1 (setpayload, -0.5, 1, plus_zero, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_Ff_b1 (setpayload, max_value, 1, plus_zero, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_Ff_b1 (setpayload, -max_value, 1, plus_zero, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_Ff_b1 (setpayload, min_value, 1, plus_zero, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_Ff_b1 (setpayload, -min_value, 1, plus_zero, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_Ff_b1 (setpayload, min_subnorm_value, 1, plus_zero, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_Ff_b1 (setpayload, -min_subnorm_value, 1, plus_zero, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_Ff_b1 (setpayload, plus_infty, 1, plus_zero, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_Ff_b1 (setpayload, minus_infty, 1, plus_zero, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_Ff_b1 (setpayload, qnan_value, 1, plus_zero, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_Ff_b1 (setpayload, -qnan_value, 1, plus_zero, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_Ff_b1 (setpayload, snan_value, 1, plus_zero, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_Ff_b1 (setpayload, -snan_value, 1, plus_zero, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_Ff_b1 (setpayload, 0xffffffp-1, 1, plus_zero, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
#if MANT_DIG >= 53
    TEST_Ff_b1 (setpayload, 0x1fffffffffffffp-1, 1, plus_zero, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
#endif
#if MANT_DIG >= 64
    TEST_Ff_b1 (setpayload, 0xffffffffffffffffp-1, 1, plus_zero, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
#endif
#if MANT_DIG >= 106
    TEST_Ff_b1 (setpayload, 0x3ffffffffffffffffffffffffffp-1, 1, plus_zero, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
#endif
#if MANT_DIG >= 113
    TEST_Ff_b1 (setpayload, 0x1ffffffffffffffffffffffffffffp-1, 1, plus_zero, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
#endif
#if PAYLOAD_DIG >= 23
    TEST_Ff_b1 (setpayload, 0x1p22, 0, qnan_value_pl ("0x400000"), NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED|TEST_NAN_SIGN|TEST_NAN_PAYLOAD|XFAIL_FLOAT128_PAYLOAD),
#else
    TEST_Ff_b1 (setpayload, 0x1p22, 1, plus_zero, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
#endif
#if PAYLOAD_DIG >= 52
    TEST_Ff_b1 (setpayload, 0x1p51, 0, qnan_value_pl ("0x8000000000000"), NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED|TEST_NAN_SIGN|TEST_NAN_PAYLOAD|XFAIL_FLOAT128_PAYLOAD),
#else
    TEST_Ff_b1 (setpayload, 0x1p51, 1, plus_zero, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
#endif
#if PAYLOAD_DIG >= 63
    TEST_Ff_b1 (setpayload, 0x1p62, 0, qnan_value_pl ("0x4000000000000000"), NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED|TEST_NAN_SIGN|TEST_NAN_PAYLOAD|XFAIL_FLOAT128_PAYLOAD),
#else
    TEST_Ff_b1 (setpayload, 0x1p62, 1, plus_zero, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
#endif
    TEST_Ff_b1 (setpayload, 0x1p111, 1, plus_zero, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
  };

static void
setpayload_test (void)
{
  FLOAT x;

  ALL_RM_TEST (setpayload, 1, setpayload_test_data, RUN_TEST_LOOP_Ff_b1, END, x);
}

static void
do_test (void)
{
  setpayload_test ();
}

/*
 * Local Variables:
 * mode:c
 * End:
 */
