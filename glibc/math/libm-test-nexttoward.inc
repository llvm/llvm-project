/* Test nexttoward.
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

/* Note, the second argument is always typed as long double.  The j type
   argument to TEST_ indicates the literal should not be post-processed by
   gen-libm-test.py.  */

static const struct test_fj_f_data nexttoward_test_data[] =
  {
#if !TEST_FLOATN
    TEST_fj_f (nexttoward, 0, 0, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_fj_f (nexttoward, minus_zero, 0, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_fj_f (nexttoward, 0, minus_zero, minus_zero, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_fj_f (nexttoward, minus_zero, minus_zero, minus_zero, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),

    TEST_fj_f (nexttoward, 9, 9, 9, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_fj_f (nexttoward, -9, -9, -9, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_fj_f (nexttoward, plus_infty, plus_infty, plus_infty, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_fj_f (nexttoward, minus_infty, minus_infty, minus_infty, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_fj_f (nexttoward, plus_infty, minus_infty, max_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_fj_f (nexttoward, minus_infty, plus_infty, -max_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),

    TEST_fj_f (nexttoward, qnan_value, 1.1L, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_fj_f (nexttoward, -qnan_value, 1.1L, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_fj_f (nexttoward, qnan_value, plus_infty, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_fj_f (nexttoward, -qnan_value, plus_infty, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_fj_f (nexttoward, qnan_value, minus_infty, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_fj_f (nexttoward, -qnan_value, minus_infty, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_fj_f (nexttoward, snan_value, 1.1L, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_fj_f (nexttoward, -snan_value, 1.1L, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_fj_f (nexttoward, snan_value, plus_infty, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_fj_f (nexttoward, -snan_value, plus_infty, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_fj_f (nexttoward, snan_value, minus_infty, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_fj_f (nexttoward, -snan_value, minus_infty, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_fj_f (nexttoward, 1.1L, qnan_value, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_fj_f (nexttoward, 1.1L, -qnan_value, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_fj_f (nexttoward, plus_infty, qnan_value, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_fj_f (nexttoward, plus_infty, -qnan_value, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_fj_f (nexttoward, minus_infty, qnan_value, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_fj_f (nexttoward, minus_infty, -qnan_value, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_fj_f (nexttoward, 1.1L, snan_value_ld, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_fj_f (nexttoward, 1.1L, -snan_value_ld, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_fj_f (nexttoward, plus_infty, snan_value_ld, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_fj_f (nexttoward, plus_infty, -snan_value_ld, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_fj_f (nexttoward, minus_infty, snan_value_ld, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_fj_f (nexttoward, minus_infty, -snan_value_ld, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_fj_f (nexttoward, qnan_value, qnan_value, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_fj_f (nexttoward, qnan_value, -qnan_value, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_fj_f (nexttoward, -qnan_value, qnan_value, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_fj_f (nexttoward, -qnan_value, -qnan_value, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_fj_f (nexttoward, snan_value, qnan_value, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_fj_f (nexttoward, snan_value, -qnan_value, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_fj_f (nexttoward, -snan_value, qnan_value, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_fj_f (nexttoward, -snan_value, -qnan_value, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_fj_f (nexttoward, qnan_value, snan_value_ld, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_fj_f (nexttoward, qnan_value, -snan_value_ld, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_fj_f (nexttoward, -qnan_value, snan_value_ld, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_fj_f (nexttoward, -qnan_value, -snan_value_ld, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_fj_f (nexttoward, snan_value, snan_value_ld, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_fj_f (nexttoward, snan_value, -snan_value_ld, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_fj_f (nexttoward, -snan_value, snan_value_ld, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_fj_f (nexttoward, -snan_value, -snan_value_ld, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),

    TEST_fj_f (nexttoward, max_value, plus_infty, plus_infty, INEXACT_EXCEPTION|OVERFLOW_EXCEPTION|ERRNO_ERANGE),
    TEST_fj_f (nexttoward, -max_value, minus_infty, minus_infty, INEXACT_EXCEPTION|OVERFLOW_EXCEPTION|ERRNO_ERANGE),

    TEST_fj_f (nexttoward, min_subnorm_value, 0, 0, INEXACT_EXCEPTION|UNDERFLOW_EXCEPTION|ERRNO_ERANGE),
    TEST_fj_f (nexttoward, min_subnorm_value, minus_zero, 0, INEXACT_EXCEPTION|UNDERFLOW_EXCEPTION|ERRNO_ERANGE),
    TEST_fj_f (nexttoward, -min_subnorm_value, 0, minus_zero, INEXACT_EXCEPTION|UNDERFLOW_EXCEPTION|ERRNO_ERANGE),
    TEST_fj_f (nexttoward, -min_subnorm_value, minus_zero, minus_zero, INEXACT_EXCEPTION|UNDERFLOW_EXCEPTION|ERRNO_ERANGE),
#endif

#if TEST_COND_binary32
    TEST_fj_f (nexttoward, 1.0, 1.1L, 0x1.000002p0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_fj_f (nexttoward, 1.0, LDBL_MAX, 0x1.000002p0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_fj_f (nexttoward, 1.0, 0x1.0000000000001p0, 0x1.000002p0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_fj_f (nexttoward, 1.0, 0.9L, 0x0.ffffffp0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_fj_f (nexttoward, 1.0, -LDBL_MAX, 0x0.ffffffp0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_fj_f (nexttoward, 1.0, 0x0.fffffffffffff8p0, 0x0.ffffffp0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_fj_f (nexttoward, -1.0, -1.1L, -0x1.000002p0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_fj_f (nexttoward, -1.0, -LDBL_MAX, -0x1.000002p0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_fj_f (nexttoward, -1.0, -0x1.0000000000001p0, -0x1.000002p0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_fj_f (nexttoward, -1.0, -0.9L, -0x0.ffffffp0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_fj_f (nexttoward, -1.0, LDBL_MAX, -0x0.ffffffp0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_fj_f (nexttoward, -1.0, -0x0.fffffffffffff8p0, -0x0.ffffffp0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_fj_f (nexttoward, -0x1.3p-145, -0xap-148L, -0x1.4p-145, INEXACT_EXCEPTION|UNDERFLOW_EXCEPTION),
# if LDBL_MANT_DIG >= 64
    TEST_fj_f (nexttoward, 1.0, 0x1.000000000000002p0L, 0x1.000002p0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_fj_f (nexttoward, 1.0, 0x0.ffffffffffffffffp0L, 0x0.ffffffp0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_fj_f (nexttoward, -1.0, -0x1.000000000000002p0L, -0x1.000002p0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_fj_f (nexttoward, -1.0, -0x0.ffffffffffffffffp0L, -0x0.ffffffp0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
# endif
# if LDBL_MANT_DIG >= 106
    TEST_fj_f (nexttoward, 1.0, 0x1.000000000000000000000000008p0L, 0x1.000002p0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_fj_f (nexttoward, 1.0, 0x0.ffffffffffffffffffffffffffcp0L, 0x0.ffffffp0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_fj_f (nexttoward, -1.0, -0x1.000000000000000000000000008p0L, -0x1.000002p0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_fj_f (nexttoward, -1.0, -0x0.ffffffffffffffffffffffffffcp0L, -0x0.ffffffp0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
# endif
# if LDBL_MANT_DIG >= 113
    TEST_fj_f (nexttoward, 1.0, 0x1.0000000000000000000000000001p0L, 0x1.000002p0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_fj_f (nexttoward, 1.0, 0x0.ffffffffffffffffffffffffffff8p0L, 0x0.ffffffp0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_fj_f (nexttoward, -1.0, -0x1.0000000000000000000000000001p0L, -0x1.000002p0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_fj_f (nexttoward, -1.0, -0x0.ffffffffffffffffffffffffffff8p0L, -0x0.ffffffp0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
# endif
#endif
#if TEST_COND_binary64
    TEST_fj_f (nexttoward, 1.0, 1.1L, 0x1.0000000000001p0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_fj_f (nexttoward, 1.0, LDBL_MAX, 0x1.0000000000001p0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_fj_f (nexttoward, 1.0, 0x1.0000000000001p0, 0x1.0000000000001p0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_fj_f (nexttoward, 1.0, 0.9L, 0x0.fffffffffffff8p0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_fj_f (nexttoward, 1.0, -LDBL_MAX, 0x0.fffffffffffff8p0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_fj_f (nexttoward, 1.0, 0x0.fffffffffffff8p0, 0x0.fffffffffffff8p0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_fj_f (nexttoward, -1.0, -1.1L, -0x1.0000000000001p0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_fj_f (nexttoward, -1.0, -LDBL_MAX, -0x1.0000000000001p0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_fj_f (nexttoward, -1.0, -0x1.0000000000001p0, -0x1.0000000000001p0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_fj_f (nexttoward, -1.0, -0.9L, -0x0.fffffffffffff8p0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_fj_f (nexttoward, -1.0, LDBL_MAX, -0x0.fffffffffffff8p0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_fj_f (nexttoward, -1.0, -0x0.fffffffffffff8p0, -0x0.fffffffffffff8p0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_fj_f (nexttoward, -1.0, -0x8.00346dc5d6388p-3L, -0x1.0000000000001p0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_fj_f (nexttoward, 0x1p-1074, 0x1p-1073L, 0x1p-1073, INEXACT_EXCEPTION|UNDERFLOW_EXCEPTION),
# if LDBL_MANT_DIG >= 64
    TEST_fj_f (nexttoward, 1.0, 0x1.000000000000002p0L, 0x1.0000000000001p0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_fj_f (nexttoward, 1.0, 0x0.ffffffffffffffffp0L, 0x0.fffffffffffff8p0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_fj_f (nexttoward, -1.0, -0x1.000000000000002p0L, -0x1.0000000000001p0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_fj_f (nexttoward, -1.0, -0x0.ffffffffffffffffp0L, -0x0.fffffffffffff8p0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
# endif
# if LDBL_MANT_DIG >= 106
    TEST_fj_f (nexttoward, 1.0, 0x1.000000000000000000000000008p0L, 0x1.0000000000001p0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_fj_f (nexttoward, 1.0, 0x0.ffffffffffffffffffffffffffcp0L, 0x0.fffffffffffff8p0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_fj_f (nexttoward, -1.0, -0x1.000000000000000000000000008p0L, -0x1.0000000000001p0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_fj_f (nexttoward, -1.0, -0x0.ffffffffffffffffffffffffffcp0L, -0x0.fffffffffffff8p0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
# endif
# if LDBL_MANT_DIG >= 113
    TEST_fj_f (nexttoward, 1.0, 0x1.0000000000000000000000000001p0L, 0x1.0000000000001p0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_fj_f (nexttoward, 1.0, 0x0.ffffffffffffffffffffffffffff8p0L, 0x0.fffffffffffff8p0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_fj_f (nexttoward, -1.0, -0x1.0000000000000000000000000001p0L, -0x1.0000000000001p0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_fj_f (nexttoward, -1.0, -0x0.ffffffffffffffffffffffffffff8p0L, -0x0.fffffffffffff8p0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
# endif
#endif
  };

static void
nexttoward_test (void)
{
#if !TEST_FLOATN
  ALL_RM_TEST (nexttoward, 1, nexttoward_test_data, RUN_TEST_LOOP_fj_f, END);
#endif
}

static void
do_test (void)
{
  nexttoward_test ();
}

/*
 * Local Variables:
 * mode:c
 * End:
 */
