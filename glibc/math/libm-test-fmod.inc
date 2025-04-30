/* Test fmod.
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

static const struct test_ff_f_data fmod_test_data[] =
  {
    /* fmod (+0, y) == +0 for y != 0.  */
    TEST_ff_f (fmod, 0, 3, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmod, 0, min_subnorm_value, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmod, 0, -min_subnorm_value, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmod, 0, min_value, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmod, 0, -min_value, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmod, 0, max_value, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmod, 0, -max_value, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),

    /* fmod (-0, y) == -0 for y != 0.  */
    TEST_ff_f (fmod, minus_zero, 3, minus_zero, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmod, minus_zero, min_subnorm_value, minus_zero, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmod, minus_zero, -min_subnorm_value, minus_zero, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmod, minus_zero, min_value, minus_zero, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmod, minus_zero, -min_value, minus_zero, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmod, minus_zero, max_value, minus_zero, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmod, minus_zero, -max_value, minus_zero, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),

    /* fmod (+inf, y) == qNaN plus invalid exception.  */
    TEST_ff_f (fmod, plus_infty, 3, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION|ERRNO_EDOM),
    TEST_ff_f (fmod, plus_infty, -1.1L, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION|ERRNO_EDOM),
    TEST_ff_f (fmod, plus_infty, 0, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION|ERRNO_EDOM),
    TEST_ff_f (fmod, plus_infty, minus_zero, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION|ERRNO_EDOM),
    TEST_ff_f (fmod, plus_infty, min_subnorm_value, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION|ERRNO_EDOM),
    TEST_ff_f (fmod, plus_infty, min_value, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION|ERRNO_EDOM),
    TEST_ff_f (fmod, plus_infty, max_value, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION|ERRNO_EDOM),
    TEST_ff_f (fmod, plus_infty, plus_infty, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION|ERRNO_EDOM),
    TEST_ff_f (fmod, plus_infty, minus_infty, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION|ERRNO_EDOM),
    /* fmod (-inf, y) == qNaN plus invalid exception.  */
    TEST_ff_f (fmod, minus_infty, 3, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION|ERRNO_EDOM),
    TEST_ff_f (fmod, minus_infty, -1.1L, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION|ERRNO_EDOM),
    TEST_ff_f (fmod, minus_infty, 0, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION|ERRNO_EDOM),
    TEST_ff_f (fmod, minus_infty, minus_zero, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION|ERRNO_EDOM),
    TEST_ff_f (fmod, minus_infty, min_subnorm_value, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION|ERRNO_EDOM),
    TEST_ff_f (fmod, minus_infty, min_value, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION|ERRNO_EDOM),
    TEST_ff_f (fmod, minus_infty, max_value, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION|ERRNO_EDOM),
    TEST_ff_f (fmod, minus_infty, plus_infty, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION|ERRNO_EDOM),
    TEST_ff_f (fmod, minus_infty, minus_infty, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION|ERRNO_EDOM),
    /* fmod (x, +0) == qNaN plus invalid exception.  */
    TEST_ff_f (fmod, 3, 0, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION|ERRNO_EDOM),
    TEST_ff_f (fmod, -1.1L, 0, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION|ERRNO_EDOM),
    TEST_ff_f (fmod, 0, 0, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION|ERRNO_EDOM),
    TEST_ff_f (fmod, minus_zero, 0, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION|ERRNO_EDOM),
    TEST_ff_f (fmod, min_subnorm_value, 0, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION|ERRNO_EDOM),
    TEST_ff_f (fmod, min_value, 0, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION|ERRNO_EDOM),
    TEST_ff_f (fmod, max_value, 0, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION|ERRNO_EDOM),
    /* fmod (x, -0) == qNaN plus invalid exception.  */
    TEST_ff_f (fmod, 3, minus_zero, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION|ERRNO_EDOM),
    TEST_ff_f (fmod, -1.1L, minus_zero, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION|ERRNO_EDOM),
    TEST_ff_f (fmod, 0, minus_zero, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION|ERRNO_EDOM),
    TEST_ff_f (fmod, minus_zero, minus_zero, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION|ERRNO_EDOM),
    TEST_ff_f (fmod, min_subnorm_value, minus_zero, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION|ERRNO_EDOM),
    TEST_ff_f (fmod, min_value, minus_zero, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION|ERRNO_EDOM),
    TEST_ff_f (fmod, max_value, minus_zero, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION|ERRNO_EDOM),

    /* fmod (x, +inf) == x for x not infinite.  */
    TEST_ff_f (fmod, 0, plus_infty, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmod, minus_zero, plus_infty, minus_zero, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmod, min_subnorm_value, plus_infty, min_subnorm_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmod, min_value, plus_infty, min_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmod, max_value, plus_infty, max_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmod, 3.0, plus_infty, 3.0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    /* fmod (x, -inf) == x for x not infinite.  */
    TEST_ff_f (fmod, 0, minus_infty, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmod, minus_zero, minus_infty, minus_zero, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmod, min_subnorm_value, minus_infty, min_subnorm_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmod, min_value, minus_infty, min_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmod, max_value, minus_infty, max_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmod, 3.0, minus_infty, 3.0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),

    TEST_ff_f (fmod, 0, qnan_value, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmod, 0, -qnan_value, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmod, minus_zero, qnan_value, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmod, minus_zero, -qnan_value, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmod, 1, qnan_value, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmod, 1, -qnan_value, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmod, plus_infty, qnan_value, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmod, plus_infty, -qnan_value, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmod, minus_infty, qnan_value, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmod, minus_infty, -qnan_value, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmod, 0, snan_value, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (fmod, 0, -snan_value, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (fmod, minus_zero, snan_value, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (fmod, minus_zero, -snan_value, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (fmod, 1, snan_value, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (fmod, 1, -snan_value, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (fmod, plus_infty, snan_value, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (fmod, plus_infty, -snan_value, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (fmod, minus_infty, snan_value, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (fmod, minus_infty, -snan_value, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (fmod, qnan_value, 0, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmod, -qnan_value, 0, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmod, qnan_value, minus_zero, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmod, -qnan_value, minus_zero, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmod, qnan_value, 1, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmod, -qnan_value, 1, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmod, qnan_value, plus_infty, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmod, -qnan_value, plus_infty, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmod, qnan_value, minus_infty, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmod, -qnan_value, minus_infty, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmod, snan_value, 0, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (fmod, -snan_value, 0, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (fmod, snan_value, minus_zero, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (fmod, -snan_value, minus_zero, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (fmod, snan_value, 1, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (fmod, -snan_value, 1, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (fmod, snan_value, plus_infty, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (fmod, -snan_value, plus_infty, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (fmod, snan_value, minus_infty, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (fmod, -snan_value, minus_infty, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (fmod, qnan_value, qnan_value, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmod, qnan_value, -qnan_value, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmod, -qnan_value, qnan_value, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmod, -qnan_value, -qnan_value, qnan_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmod, qnan_value, snan_value, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (fmod, qnan_value, -snan_value, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (fmod, -qnan_value, snan_value, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (fmod, -qnan_value, -snan_value, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (fmod, snan_value, qnan_value, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (fmod, snan_value, -qnan_value, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (fmod, -snan_value, qnan_value, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (fmod, -snan_value, -qnan_value, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (fmod, snan_value, snan_value, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (fmod, snan_value, -snan_value, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (fmod, -snan_value, snan_value, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),
    TEST_ff_f (fmod, -snan_value, -snan_value, qnan_value, NO_INEXACT_EXCEPTION|INVALID_EXCEPTION),

    TEST_ff_f (fmod, 6.5, 2.25L, 2.0L, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmod, -6.5, 2.25L, -2.0L, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmod, 6.5, -2.25L, 2.0L, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmod, -6.5, -2.25L, -2.0L, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),

    TEST_ff_f (fmod, max_value, max_value, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmod, max_value, -max_value, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmod, max_value, min_value, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmod, max_value, -min_value, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmod, max_value, min_subnorm_value, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmod, max_value, -min_subnorm_value, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmod, -max_value, max_value, minus_zero, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmod, -max_value, -max_value, minus_zero, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmod, -max_value, min_value, minus_zero, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmod, -max_value, -min_value, minus_zero, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmod, -max_value, min_subnorm_value, minus_zero, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmod, -max_value, -min_subnorm_value, minus_zero, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),

    TEST_ff_f (fmod, min_value, max_value, min_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmod, min_value, -max_value, min_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmod, min_value, min_value, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmod, min_value, -min_value, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmod, min_value, min_subnorm_value, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmod, min_value, -min_subnorm_value, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmod, -min_value, max_value, -min_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmod, -min_value, -max_value, -min_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmod, -min_value, min_value, minus_zero, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmod, -min_value, -min_value, minus_zero, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmod, -min_value, min_subnorm_value, minus_zero, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmod, -min_value, -min_subnorm_value, minus_zero, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),

    TEST_ff_f (fmod, min_subnorm_value, max_value, min_subnorm_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmod, min_subnorm_value, -max_value, min_subnorm_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmod, min_subnorm_value, min_value, min_subnorm_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmod, min_subnorm_value, -min_value, min_subnorm_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmod, min_subnorm_value, min_subnorm_value, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmod, min_subnorm_value, -min_subnorm_value, 0, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmod, -min_subnorm_value, max_value, -min_subnorm_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmod, -min_subnorm_value, -max_value, -min_subnorm_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmod, -min_subnorm_value, min_value, -min_subnorm_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmod, -min_subnorm_value, -min_value, -min_subnorm_value, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmod, -min_subnorm_value, min_subnorm_value, minus_zero, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmod, -min_subnorm_value, -min_subnorm_value, minus_zero, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),

    TEST_ff_f (fmod, 0x1p127L, 0x3p-149L, 0x1p-149L, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmod, 0x1p127L, -0x3p-149L, 0x1p-149L, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmod, 0x1p127L, 0x3p-148L, 0x1p-147L, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmod, 0x1p127L, -0x3p-148L, 0x1p-147L, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmod, 0x1p127L, 0x3p-126L, 0x1p-125L, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmod, 0x1p127L, -0x3p-126L, 0x1p-125L, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmod, -0x1p127L, 0x3p-149L, -0x1p-149L, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmod, -0x1p127L, -0x3p-149L, -0x1p-149L, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmod, -0x1p127L, 0x3p-148L, -0x1p-147L, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmod, -0x1p127L, -0x3p-148L, -0x1p-147L, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmod, -0x1p127L, 0x3p-126L, -0x1p-125L, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmod, -0x1p127L, -0x3p-126L, -0x1p-125L, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
#if !TEST_COND_binary32
    TEST_ff_f (fmod, 0x1p1023L, 0x3p-1074L, 0x1p-1073L, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmod, 0x1p1023L, -0x3p-1074L, 0x1p-1073L, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmod, 0x1p1023L, 0x3p-1073L, 0x1p-1073L, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmod, 0x1p1023L, -0x3p-1073L, 0x1p-1073L, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmod, 0x1p1023L, 0x3p-1022L, 0x1p-1021L, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmod, 0x1p1023L, -0x3p-1022L, 0x1p-1021L, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmod, -0x1p1023L, 0x3p-1074L, -0x1p-1073L, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmod, -0x1p1023L, -0x3p-1074L, -0x1p-1073L, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmod, -0x1p1023L, 0x3p-1073L, -0x1p-1073L, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmod, -0x1p1023L, -0x3p-1073L, -0x1p-1073L, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmod, -0x1p1023L, 0x3p-1022L, -0x1p-1021L, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmod, -0x1p1023L, -0x3p-1022L, -0x1p-1021L, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
#endif
#if MIN_EXP <= -16381
    TEST_ff_f (fmod, 0x1p16383L, 0x3p-16445L, 0x1p-16445L, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmod, 0x1p16383L, -0x3p-16445L, 0x1p-16445L, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmod, 0x1p16383L, 0x3p-16444L, 0x1p-16443L, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmod, 0x1p16383L, -0x3p-16444L, 0x1p-16443L, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmod, 0x1p16383L, 0x3p-16382L, 0x1p-16381L, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmod, 0x1p16383L, -0x3p-16382L, 0x1p-16381L, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmod, -0x1p16383L, 0x3p-16445L, -0x1p-16445L, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmod, -0x1p16383L, -0x3p-16445L, -0x1p-16445L, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmod, -0x1p16383L, 0x3p-16444L, -0x1p-16443L, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmod, -0x1p16383L, -0x3p-16444L, -0x1p-16443L, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmod, -0x1p16383L, 0x3p-16382L, -0x1p-16381L, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmod, -0x1p16383L, -0x3p-16382L, -0x1p-16381L, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
# if MANT_DIG >= 113
    TEST_ff_f (fmod, 0x1p16383L, 0x3p-16494L, 0x1p-16493L, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmod, 0x1p16383L, -0x3p-16494L, 0x1p-16493L, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmod, 0x1p16383L, 0x3p-16493L, 0x1p-16493L, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmod, 0x1p16383L, -0x3p-16493L, 0x1p-16493L, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmod, -0x1p16383L, 0x3p-16494L, -0x1p-16493L, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmod, -0x1p16383L, -0x3p-16494L, -0x1p-16493L, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmod, -0x1p16383L, 0x3p-16493L, -0x1p-16493L, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmod, -0x1p16383L, -0x3p-16493L, -0x1p-16493L, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
# endif
#endif

    TEST_ff_f (fmod, 0x0.fffffep-126L, 0x1p-149L, plus_zero, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
#if !TEST_COND_binary32
    TEST_ff_f (fmod, 0x0.fffffffffffffp-1022L, 0x1p-1074L, plus_zero, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
#endif
#if MIN_EXP <= -16381
    TEST_ff_f (fmod, 0x0.fffffffffffffffep-16382L, 0x1p-16445L, plus_zero, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
#endif
#if MANT_DIG >= 56
    TEST_ff_f (fmod, -0x1.00000000000004p+0L, 0x1.fffffffffffff8p-1L, -0x1p-53L, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmod, 0x1.fffffffffffffap-1L, 0x1.fffffffffffff8p-1L, 0x1p-56L, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmod, -0x1.fffffffffffffap-1L, 0x1.fffffffffffff8p-1L, -0x1p-56L, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmod, 0x1.fffffffffffffap-1L, -0x1.fffffffffffff8p-1L, 0x1p-56L, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
    TEST_ff_f (fmod, -0x1.fffffffffffffap-1L, -0x1.fffffffffffff8p-1L, -0x1p-56L, NO_INEXACT_EXCEPTION|ERRNO_UNCHANGED),
#endif
  };

static void
fmod_test (void)
{
  ALL_RM_TEST (fmod, 1, fmod_test_data, RUN_TEST_LOOP_ff_f, END);
}

static void
do_test (void)
{
  fmod_test ();
}

/*
 * Local Variables:
 * mode:c
 * End:
 */
