/* Test ctanh.
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

static const struct test_c_c_data ctanh_test_data[] =
  {
    TEST_c_c (ctanh, plus_infty, 0, 1.0, 0.0),
    TEST_c_c (ctanh, plus_infty, 1, 1.0, 0.0),
    TEST_c_c (ctanh, plus_infty, 2, 1.0, minus_zero),
    TEST_c_c (ctanh, plus_infty, 3, 1.0, minus_zero),
    TEST_c_c (ctanh, plus_infty, 4, 1.0, 0.0),
    TEST_c_c (ctanh, plus_infty, min_value, 1.0, 0.0),
    TEST_c_c (ctanh, plus_infty, min_subnorm_value, 1.0, 0.0),
    TEST_c_c (ctanh, plus_infty, 0x1p127, 1.0, 0.0),
    TEST_c_c (ctanh, plus_infty, 0x1.11p127, 1.0, minus_zero),
#if !TEST_COND_binary32
    TEST_c_c (ctanh, plus_infty, 0x1p1023, 1.0, minus_zero),
    TEST_c_c (ctanh, plus_infty, 0x1.5p1023, 1.0, 0.0),
#endif
#if MAX_EXP >= 16384
    TEST_c_c (ctanh, plus_infty, 0x1p16383L, 1.0, 0.0),
    TEST_c_c (ctanh, plus_infty, 0x1.2p16383L, 1.0, minus_zero),
#endif
    TEST_c_c (ctanh, plus_infty, minus_zero, 1.0, minus_zero),
    TEST_c_c (ctanh, plus_infty, -1, 1.0, minus_zero),
    TEST_c_c (ctanh, plus_infty, -2, 1.0, 0.0),
    TEST_c_c (ctanh, plus_infty, -3, 1.0, 0.0),
    TEST_c_c (ctanh, plus_infty, -4, 1.0, minus_zero),
    TEST_c_c (ctanh, plus_infty, -min_value, 1.0, minus_zero),
    TEST_c_c (ctanh, plus_infty, -min_subnorm_value, 1.0, minus_zero),
    TEST_c_c (ctanh, plus_infty, -0x1p127, 1.0, minus_zero),
    TEST_c_c (ctanh, plus_infty, -0x1.11p127, 1.0, 0.0),
#if !TEST_COND_binary32
    TEST_c_c (ctanh, plus_infty, -0x1p1023, 1.0, 0.0),
    TEST_c_c (ctanh, plus_infty, -0x1.5p1023, 1.0, minus_zero),
#endif
#if MAX_EXP >= 16384
    TEST_c_c (ctanh, plus_infty, -0x1p16383L, 1.0, minus_zero),
    TEST_c_c (ctanh, plus_infty, -0x1.2p16383L, 1.0, 0.0),
#endif
    TEST_c_c (ctanh, minus_infty, 0, -1.0, 0.0),
    TEST_c_c (ctanh, minus_infty, 1, -1.0, 0.0),
    TEST_c_c (ctanh, minus_infty, 2, -1.0, minus_zero),
    TEST_c_c (ctanh, minus_infty, 3, -1.0, minus_zero),
    TEST_c_c (ctanh, minus_infty, 4, -1.0, 0.0),
    TEST_c_c (ctanh, minus_infty, min_value, -1.0, 0.0),
    TEST_c_c (ctanh, minus_infty, min_subnorm_value, -1.0, 0.0),
    TEST_c_c (ctanh, minus_infty, 0x1p127, -1.0, 0.0),
    TEST_c_c (ctanh, minus_infty, 0x1.11p127, -1.0, minus_zero),
#if !TEST_COND_binary32
    TEST_c_c (ctanh, minus_infty, 0x1p1023, -1.0, minus_zero),
    TEST_c_c (ctanh, minus_infty, 0x1.5p1023, -1.0, 0.0),
#endif
#if MAX_EXP >= 16384
    TEST_c_c (ctanh, minus_infty, 0x1p16383L, -1.0, 0.0),
    TEST_c_c (ctanh, minus_infty, 0x1.2p16383L, -1.0, minus_zero),
#endif
    TEST_c_c (ctanh, minus_infty, minus_zero, -1.0, minus_zero),
    TEST_c_c (ctanh, minus_infty, -1, -1.0, minus_zero),
    TEST_c_c (ctanh, minus_infty, -2, -1.0, 0.0),
    TEST_c_c (ctanh, minus_infty, -3, -1.0, 0.0),
    TEST_c_c (ctanh, minus_infty, -4, -1.0, minus_zero),
    TEST_c_c (ctanh, minus_infty, -min_value, -1.0, minus_zero),
    TEST_c_c (ctanh, minus_infty, -min_subnorm_value, -1.0, minus_zero),
    TEST_c_c (ctanh, minus_infty, -0x1p127, -1.0, minus_zero),
    TEST_c_c (ctanh, minus_infty, -0x1.11p127, -1.0, 0.0),
#if !TEST_COND_binary32
    TEST_c_c (ctanh, minus_infty, -0x1p1023, -1.0, 0.0),
    TEST_c_c (ctanh, minus_infty, -0x1.5p1023, -1.0, minus_zero),
#endif
#if MAX_EXP >= 16384
    TEST_c_c (ctanh, minus_infty, -0x1p16383L, -1.0, minus_zero),
    TEST_c_c (ctanh, minus_infty, -0x1.2p16383L, -1.0, 0.0),
#endif

    TEST_c_c (ctanh, 0, plus_infty, 0, qnan_value, INVALID_EXCEPTION),
    TEST_c_c (ctanh, 2, plus_infty, qnan_value, qnan_value, INVALID_EXCEPTION),
    TEST_c_c (ctanh, 0, minus_infty, 0, qnan_value, INVALID_EXCEPTION),
    TEST_c_c (ctanh, 2, minus_infty, qnan_value, qnan_value, INVALID_EXCEPTION),
    TEST_c_c (ctanh, minus_zero, plus_infty, minus_zero, qnan_value, INVALID_EXCEPTION),
    TEST_c_c (ctanh, -2, plus_infty, qnan_value, qnan_value, INVALID_EXCEPTION),
    TEST_c_c (ctanh, minus_zero, minus_infty, minus_zero, qnan_value, INVALID_EXCEPTION),
    TEST_c_c (ctanh, -2, minus_infty, qnan_value, qnan_value, INVALID_EXCEPTION),

    TEST_c_c (ctanh, plus_infty, qnan_value, 1.0, 0.0, IGNORE_ZERO_INF_SIGN),
    TEST_c_c (ctanh, plus_infty, plus_infty, 1.0, 0.0, IGNORE_ZERO_INF_SIGN),
    TEST_c_c (ctanh, plus_infty, minus_infty, 1.0, 0.0, IGNORE_ZERO_INF_SIGN),
    TEST_c_c (ctanh, minus_infty, qnan_value, -1.0, 0.0, IGNORE_ZERO_INF_SIGN),
    TEST_c_c (ctanh, minus_infty, plus_infty, -1.0, 0.0, IGNORE_ZERO_INF_SIGN),
    TEST_c_c (ctanh, minus_infty, minus_infty, -1.0, 0.0, IGNORE_ZERO_INF_SIGN),

    TEST_c_c (ctanh, qnan_value, 0, qnan_value, 0.0),
    TEST_c_c (ctanh, qnan_value, minus_zero, qnan_value, minus_zero),

    TEST_c_c (ctanh, qnan_value, 0.5, qnan_value, qnan_value, INVALID_EXCEPTION_OK),
    TEST_c_c (ctanh, qnan_value, -4.5, qnan_value, qnan_value, INVALID_EXCEPTION_OK),

    TEST_c_c (ctanh, 0, qnan_value, 0, qnan_value),
    TEST_c_c (ctanh, 5, qnan_value, qnan_value, qnan_value, INVALID_EXCEPTION_OK),
    TEST_c_c (ctanh, minus_zero, qnan_value, minus_zero, qnan_value),
    TEST_c_c (ctanh, -0.25, qnan_value, qnan_value, qnan_value, INVALID_EXCEPTION_OK),

    TEST_c_c (ctanh, qnan_value, qnan_value, qnan_value, qnan_value),

    AUTO_TESTS_c_c (ctanh),
  };

static void
ctanh_test (void)
{
  ALL_RM_TEST (ctanh, 0, ctanh_test_data, RUN_TEST_LOOP_c_c, END_COMPLEX);
}

static void
do_test (void)
{
  ctanh_test ();
}

/*
 * Local Variables:
 * mode:c
 * End:
 */
