/* Test requirements on FP_LLOGB* constants.
   Copyright (C) 2016-2021 Free Software Foundation, Inc.
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

#include <limits.h>
#include <math.h>

#if FP_LLOGB0 >= 0
# error "FP_LLOGB0 is positive"
#endif

#if FP_LLOGB0 != LONG_MIN && FP_LLOGB0 != -LONG_MAX
# error "FP_LLOGB0 must be LONG_MIN or -LONG_MAX"
#endif

#if FP_LLOGBNAN >= 0 && FP_LLOGBNAN != LONG_MAX
# error "FP_LLOGBNAN must be LONG_MIN or LONG_MAX"
#endif

#if FP_LLOGBNAN < 0 && FP_LLOGBNAN != LONG_MIN
# error "FP_LLOGBNAN must be LONG_MIN or LONG_MAX"
#endif

#if (FP_ILOGB0 == INT_MIN) != (FP_LLOGB0 == LONG_MIN)
# error "FP_ILOGB0 and FP_LLOGB0 inconsistent"
#endif

#if (FP_ILOGBNAN == INT_MIN) != (FP_LLOGBNAN == LONG_MIN)
# error "FP_ILOGBNAN and FP_LLOGBNAN inconsistent"
#endif

/* This is a compilation test.  */
#define TEST_FUNCTION 0
#include "../test-skeleton.c"
