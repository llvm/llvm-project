/* Test frexp.
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

static const struct test_f_f1_data frexp_test_data[] =
  {
    TEST_fI_f1 (frexp, plus_infty, plus_infty, IGNORE, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_fI_f1 (frexp, minus_infty, minus_infty, IGNORE, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_fI_f1 (frexp, qnan_value, qnan_value, IGNORE, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_fI_f1 (frexp, -qnan_value, qnan_value, IGNORE, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_fI_f1 (frexp, snan_value, qnan_value, IGNORE, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_fI_f1 (frexp, -snan_value, qnan_value, IGNORE, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),

    TEST_fI_f1 (frexp, 0.0, 0.0, 0.0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_fI_f1 (frexp, minus_zero, minus_zero, 0.0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),

    TEST_fI_f1 (frexp, 12.8L, 0.8L, 4, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_fI_f1 (frexp, -27.34L, -0.854375L, 5, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_fI_f1 (frexp, 0.25L, 0.5L, -1, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_fI_f1 (frexp, 0.5L, 0.5L, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_fI_f1 (frexp, 1.0L, 0.5L, 1, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_fI_f1 (frexp, -1.0L, -0.5L, 1, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_fI_f1 (frexp, 2.0L, 0.5L, 2, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_fI_f1 (frexp, 4.0L, 0.5L, 3, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_fI_f1 (frexp, 0x1p127L, 0.5L, 128, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_fI_f1 (frexp, -0x1p127L, -0.5L, 128, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_fI_f1 (frexp, 0x1p-126L, 0.5L, -125, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_fI_f1 (frexp, 0x1p-127L, 0.5L, -126, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_fI_f1 (frexp, 0x1p-149L, 0.5L, -148, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_fI_f1 (frexp, -0x1p-149L, -0.5L, -148, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
#if !TEST_COND_binary32
    TEST_fI_f1 (frexp, 0x1p1023L, 0.5L, 1024, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_fI_f1 (frexp, -0x1p1023L, -0.5L, 1024, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_fI_f1 (frexp, 0x1p-1022L, 0.5L, -1021, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_fI_f1 (frexp, 0x1p-1023L, 0.5L, -1022, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_fI_f1 (frexp, 0x1p-1074L, 0.5L, -1073, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_fI_f1 (frexp, -0x1p-1074L, -0.5L, -1073, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
#endif
#if MIN_EXP <= -16381
    TEST_fI_f1 (frexp, 0x1p16383L, 0.5L, 16384, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_fI_f1 (frexp, -0x1p16383L, -0.5L, 16384, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_fI_f1 (frexp, 0x1p-16382L, 0.5L, -16381, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_fI_f1 (frexp, 0x1p-16383L, 0.5L, -16382, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_fI_f1 (frexp, 0x1p-16445L, 0.5L, -16444, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_fI_f1 (frexp, -0x1p-16445L, -0.5L, -16444, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
# if MANT_DIG >= 113
    TEST_fI_f1 (frexp, 0x1p-16494L, 0.5L, -16493, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_fI_f1 (frexp, -0x1p-16494L, -0.5L, -16493, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
# endif
#endif

#if MANT_DIG >= 106
    TEST_fI_f1 (frexp, 1.0L - 0x1p-106L, 1.0L - 0x1p-106L, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_fI_f1 (frexp, 1.0L, 0.5L, 1, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_fI_f1 (frexp, 1.0L + 0x1p-105L, 0.5L + 0x1p-106L, 1, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_fI_f1 (frexp, -1.0L + 0x1p-106L, -1.0L + 0x1p-106L, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_fI_f1 (frexp, -1.0L, -0.5L, 1, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_fI_f1 (frexp, -1.0L - 0x1p-105L, -0.5L - 0x1p-106L, 1, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
#endif
  };

static void
frexp_test (void)
{
  int x;

  ALL_RM_TEST (frexp, 1, frexp_test_data, RUN_TEST_LOOP_fI_f1, END, x);
}

static void
do_test (void)
{
  frexp_test ();
}

/*
 * Local Variables:
 * mode:c
 * End:
 */
