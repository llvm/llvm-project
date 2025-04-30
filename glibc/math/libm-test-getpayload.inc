/* Test getpayload.
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

static const struct test_f_f_data getpayload_test_data[] =
  {
    TEST_fp_f (getpayload, plus_infty, -1.0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_fp_f (getpayload, minus_infty, -1.0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_fp_f (getpayload, plus_zero, -1.0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_fp_f (getpayload, minus_zero, -1.0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_fp_f (getpayload, 1000, -1.0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_fp_f (getpayload, max_value, -1.0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_fp_f (getpayload, -max_value, -1.0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_fp_f (getpayload, min_value, -1.0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_fp_f (getpayload, -min_value, -1.0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_fp_f (getpayload, min_subnorm_value, -1.0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_fp_f (getpayload, -min_subnorm_value, -1.0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
#if HIGH_ORDER_BIT_IS_SET_FOR_SNAN
    TEST_fp_f (getpayload, snan_value_pl ("0x0"), plus_zero, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_fp_f (getpayload, -snan_value_pl ("0x0"), plus_zero, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
#else
    TEST_fp_f (getpayload, qnan_value_pl ("0x0"), plus_zero, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_fp_f (getpayload, -qnan_value_pl ("0x0"), plus_zero, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
#endif
    TEST_fp_f (getpayload, qnan_value_pl ("0x1"), 1, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED|XFAIL_FLOAT128_PAYLOAD),
    TEST_fp_f (getpayload, -qnan_value_pl ("0x1"), 1, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED|XFAIL_FLOAT128_PAYLOAD),
    TEST_fp_f (getpayload, snan_value_pl ("0x1"), 1, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED|XFAIL_FLOAT128_PAYLOAD),
    TEST_fp_f (getpayload, -snan_value_pl ("0x1"), 1, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED|XFAIL_FLOAT128_PAYLOAD),
    TEST_fp_f (getpayload, qnan_value_pl ("0x2"), 2, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED|XFAIL_FLOAT128_PAYLOAD),
    TEST_fp_f (getpayload, -qnan_value_pl ("0x2"), 2, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED|XFAIL_FLOAT128_PAYLOAD),
    TEST_fp_f (getpayload, snan_value_pl ("0x2"), 2, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED|XFAIL_FLOAT128_PAYLOAD),
    TEST_fp_f (getpayload, -snan_value_pl ("0x2"), 2, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED|XFAIL_FLOAT128_PAYLOAD),
    TEST_fp_f (getpayload, qnan_value_pl ("0x3fffff"), 0x3fffff, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED|XFAIL_FLOAT128_PAYLOAD),
    TEST_fp_f (getpayload, -qnan_value_pl ("0x3fffff"), 0x3fffff, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED|XFAIL_FLOAT128_PAYLOAD),
    TEST_fp_f (getpayload, snan_value_pl ("0x3fffff"), 0x3fffff, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED|XFAIL_FLOAT128_PAYLOAD),
    TEST_fp_f (getpayload, -snan_value_pl ("0x3fffff"), 0x3fffff, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED|XFAIL_FLOAT128_PAYLOAD),
#if PAYLOAD_DIG >= 51
    TEST_fp_f (getpayload, qnan_value_pl ("0x7ffffffffffff"), 0x7ffffffffffffp0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED|XFAIL_FLOAT128_PAYLOAD),
    TEST_fp_f (getpayload, -qnan_value_pl ("0x7ffffffffffff"), 0x7ffffffffffffp0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED|XFAIL_FLOAT128_PAYLOAD),
    TEST_fp_f (getpayload, snan_value_pl ("0x7ffffffffffff"), 0x7ffffffffffffp0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED|XFAIL_FLOAT128_PAYLOAD),
    TEST_fp_f (getpayload, -snan_value_pl ("0x7ffffffffffff"), 0x7ffffffffffffp0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED|XFAIL_FLOAT128_PAYLOAD),
#endif
#if PAYLOAD_DIG >= 62
    TEST_fp_f (getpayload, qnan_value_pl ("0x3fffffffffffffff"), 0x3fffffffffffffffp0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED|XFAIL_FLOAT128_PAYLOAD),
    TEST_fp_f (getpayload, -qnan_value_pl ("0x3fffffffffffffff"), 0x3fffffffffffffffp0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED|XFAIL_FLOAT128_PAYLOAD),
    TEST_fp_f (getpayload, snan_value_pl ("0x3fffffffffffffff"), 0x3fffffffffffffffp0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED|XFAIL_FLOAT128_PAYLOAD),
    TEST_fp_f (getpayload, -snan_value_pl ("0x3fffffffffffffff"), 0x3fffffffffffffffp0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED|XFAIL_FLOAT128_PAYLOAD),
#endif
#if PAYLOAD_DIG >= 111
    TEST_fp_f (getpayload, qnan_value_pl ("0x7fffffffffffffffffffffffffff"), 0x7fffffffffffffffffffffffffffp0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED|XFAIL_FLOAT128_PAYLOAD),
    TEST_fp_f (getpayload, -qnan_value_pl ("0x7fffffffffffffffffffffffffff"), 0x7fffffffffffffffffffffffffffp0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED|XFAIL_FLOAT128_PAYLOAD),
    TEST_fp_f (getpayload, snan_value_pl ("0x7fffffffffffffffffffffffffff"), 0x7fffffffffffffffffffffffffffp0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED|XFAIL_FLOAT128_PAYLOAD),
    TEST_fp_f (getpayload, -snan_value_pl ("0x7fffffffffffffffffffffffffff"), 0x7fffffffffffffffffffffffffffp0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED|XFAIL_FLOAT128_PAYLOAD),
#endif
  };

static void
getpayload_test (void)
{
  ALL_RM_TEST (getpayload, 1, getpayload_test_data, RUN_TEST_LOOP_fp_f, END);
}

static void
do_test (void)
{
  getpayload_test ();
}

/*
 * Local Variables:
 * mode:c
 * End:
 */
