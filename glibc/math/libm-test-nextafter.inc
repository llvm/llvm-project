/* Test nextafter.
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

static const struct test_ff_f_data nextafter_test_data[] =
  {
    TEST_ff_f (nextafter, 0, 0, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (nextafter, minus_zero, 0, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (nextafter, 0, minus_zero, minus_zero, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (nextafter, minus_zero, minus_zero, minus_zero, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),

    TEST_ff_f (nextafter, 9, 9, 9, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (nextafter, -9, -9, -9, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (nextafter, plus_infty, plus_infty, plus_infty, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (nextafter, minus_infty, minus_infty, minus_infty, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (nextafter, plus_infty, minus_infty, max_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (nextafter, minus_infty, plus_infty, -max_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),

    TEST_ff_f (nextafter, qnan_value, 1.1L, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (nextafter, -qnan_value, 1.1L, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (nextafter, qnan_value, plus_infty, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (nextafter, -qnan_value, plus_infty, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (nextafter, qnan_value, minus_infty, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (nextafter, -qnan_value, minus_infty, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (nextafter, snan_value, 1.1L, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (nextafter, -snan_value, 1.1L, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (nextafter, snan_value, plus_infty, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (nextafter, -snan_value, plus_infty, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (nextafter, snan_value, minus_infty, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (nextafter, -snan_value, minus_infty, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (nextafter, 1.1L, qnan_value, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (nextafter, 1.1L, -qnan_value, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (nextafter, plus_infty, qnan_value, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (nextafter, plus_infty, -qnan_value, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (nextafter, minus_infty, qnan_value, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (nextafter, minus_infty, -qnan_value, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (nextafter, 1.1L, snan_value, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (nextafter, 1.1L, -snan_value, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (nextafter, plus_infty, snan_value, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (nextafter, plus_infty, -snan_value, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (nextafter, minus_infty, snan_value, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (nextafter, minus_infty, -snan_value, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (nextafter, qnan_value, qnan_value, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (nextafter, qnan_value, -qnan_value, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (nextafter, -qnan_value, qnan_value, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (nextafter, -qnan_value, -qnan_value, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (nextafter, snan_value, qnan_value, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (nextafter, snan_value, -qnan_value, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (nextafter, -snan_value, qnan_value, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (nextafter, -snan_value, -qnan_value, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (nextafter, qnan_value, snan_value, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (nextafter, qnan_value, -snan_value, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (nextafter, -qnan_value, snan_value, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (nextafter, -qnan_value, -snan_value, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (nextafter, snan_value, snan_value, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (nextafter, snan_value, -snan_value, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (nextafter, -snan_value, snan_value, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (nextafter, -snan_value, -snan_value, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),

    TEST_ff_f (nextafter, max_value, plus_infty, plus_infty, INEXACT_EXCEPTION|OVERFLOW_EXCEPTION|ERRNO_ERANGE),
    TEST_ff_f (nextafter, -max_value, minus_infty, minus_infty, INEXACT_EXCEPTION|OVERFLOW_EXCEPTION|ERRNO_ERANGE),

    TEST_ff_f (nextafter, min_subnorm_value, 0, 0, INEXACT_EXCEPTION|UNDERFLOW_EXCEPTION|ERRNO_ERANGE),
    TEST_ff_f (nextafter, min_subnorm_value, minus_zero, 0, INEXACT_EXCEPTION|UNDERFLOW_EXCEPTION|ERRNO_ERANGE),
    TEST_ff_f (nextafter, -min_subnorm_value, 0, minus_zero, INEXACT_EXCEPTION|UNDERFLOW_EXCEPTION|ERRNO_ERANGE),
    TEST_ff_f (nextafter, -min_subnorm_value, minus_zero, minus_zero, INEXACT_EXCEPTION|UNDERFLOW_EXCEPTION|ERRNO_ERANGE),

#if TEST_COND_intel96
    TEST_ff_f (nextafter, -0x0.fffffffep-16382L, 0.0L, -0x0.fffffffdfffffffep-16382L, INEXACT_EXCEPTION|UNDERFLOW_EXCEPTION|ERRNO_ERANGE),
#endif

#if TEST_COND_binary32
    TEST_ff_f (nextafter, 1.0, 2.0, 0x1.000002p0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (nextafter, 1.0, 0.9, 0x0.ffffffp0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (nextafter, -1.0, -2.0, -0x1.000002p0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (nextafter, -1.0, 2.0, -0x0.ffffffp0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
#endif
#if TEST_COND_binary64
    TEST_ff_f (nextafter, 1.0, 2.0, 0x1.0000000000001p+0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (nextafter, 1.0, 0.9, 0x1.fffffffffffffp-1, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (nextafter, -1.0, -2.0, -0x1.0000000000001p+0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (nextafter, -1.0, 2.0, -0x1.fffffffffffffp-1, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
#endif
#if TEST_COND_m68k96 || TEST_COND_intel96
    TEST_ff_f (nextafter, 1.0L, 2.0L, 0x8.0000000000000010p-3L, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (nextafter, 1.0L, -2.0L, 0xf.fffffffffffffff0p-4L, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (nextafter, -1.0L, -2.0L, -0x8.0000000000000010p-3L, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (nextafter, -1.0L, 2.0L, -0xf.fffffffffffffff0p-4L, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
#endif
#if TEST_COND_m68k96
    TEST_ff_f (nextafter, -0x0.fffffffep-16383L, 0.0L, -0x0.fffffffdfffffffep-16383L, INEXACT_EXCEPTION|UNDERFLOW_EXCEPTION|ERRNO_ERANGE),
#endif
#if TEST_COND_ibm128
    TEST_ff_f (nextafter, 1.0L, -10.0L, 1.0L - 0x1p-106L, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (nextafter, 1.0L, 10.0L, 1.0L + 0x1p-105L, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (nextafter, 1.0L - 0x1p-106L, 10.0L, 1.0L, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (nextafter, -1.0L, -10.0L, -1.0L - 0x1p-105L, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (nextafter, -1.0L, 10.0L, -1.0L + 0x1p-106L, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (nextafter, -1.0L + 0x1p-106L, -10.0L, -1.0L, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
#endif
#if TEST_COND_binary128
    TEST_ff_f (nextafter, 1.0L, 10.0L, 0x1.0000000000000000000000000001p0L, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (nextafter, 1.0L, -10.0L, 0x1.ffffffffffffffffffffffffffffp-1L, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (nextafter, -1.0L, 10.0L, -0x1.ffffffffffffffffffffffffffffp-1L, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (nextafter, -1.0L, -10.0L, -0x1.0000000000000000000000000001p+0L, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
#endif

    /* XXX We need the hexadecimal FP number representation here for further
       tests.  */
  };

static void
nextafter_test (void)
{
  ALL_RM_TEST (nextafter, 1, nextafter_test_data, RUN_TEST_LOOP_ff_f, END);
}

static void
do_test (void)
{
  nextafter_test ();
}

/*
 * Local Variables:
 * mode:c
 * End:
 */
