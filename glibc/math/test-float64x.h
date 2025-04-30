/* Common definitions for libm tests for _Float64x.

   Copyright (C) 2017-2021 Free Software Foundation, Inc.
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

#include "test-math-floatn.h"

/* Fixup builtins and constants for older compilers.  */
#include <bits/floatn.h>
#include <float.h>

#define FUNC(function) function ## f64x
#define FLOAT _Float64x
#define CFLOAT __CFLOAT64X
#define BUILD_COMPLEX(real, imag) (CMPLXF64X ((real), (imag)))
#define PREFIX FLT64X
#if FLT64X_MANT_DIG == LDBL_MANT_DIG
# define TYPE_STR "ldouble"
# define ULP_IDX ULP_LDBL
#else
# define TYPE_STR "float128"
# define ULP_IDX ULP_FLT128
#endif
#define LIT(x) __f64x (x)
#define LITM(x) x ## f64x
#define FTOSTR strfromf64x
#define snan_value_MACRO SNANF64X
#define FUNC_NARROW_PREFIX f64x
