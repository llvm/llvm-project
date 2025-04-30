/* Test evaluation method typedefs match FLT_EVAL_METHOD.
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

#include <float.h>
#include <math.h>

#if FLT_EVAL_METHOD == -1
/* Cannot test.  */
typedef float_t my_float_t;
typedef double_t my_double_t;
#elif FLT_EVAL_METHOD == 0 || FLT_EVAL_METHOD == 16
typedef float my_float_t;
typedef double my_double_t;
#elif FLT_EVAL_METHOD == 1
typedef double my_float_t;
typedef double my_double_t;
#elif FLT_EVAL_METHOD == 2
typedef long double my_float_t;
typedef long double my_double_t;
#elif FLT_EVAL_METHOD == 32
typedef _Float32 my_float_t;
typedef double my_double_t;
#elif FLT_EVAL_METHOD == 33
typedef _Float32x my_float_t;
typedef _Float32x my_double_t;
#elif FLT_EVAL_METHOD == 64
typedef _Float64 my_float_t;
typedef _Float64 my_double_t;
#elif FLT_EVAL_METHOD == 65
typedef _Float64x my_float_t;
typedef _Float64x my_double_t;
#elif FLT_EVAL_METHOD == 128
typedef _Float128 my_float_t;
typedef _Float128 my_double_t;
#elif FLT_EVAL_METHOD == 129
typedef _Float128x my_float_t;
typedef _Float128x my_double_t;
#else
# error "Unknown FLT_EVAL_METHOD"
#endif

extern float_t test_float_t;
extern my_float_t test_float_t;

extern double_t test_double_t;
extern my_double_t test_double_t;

/* This is a compilation test.  */
#define TEST_FUNCTION 0
#include "../test-skeleton.c"
