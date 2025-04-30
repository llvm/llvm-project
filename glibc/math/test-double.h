/* Common definitions for libm tests for double.
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

#define FUNC(function) function
#define FLOAT double
#define CFLOAT __complex__ double
#define BUILD_COMPLEX(real, imag) (CMPLX ((real), (imag)))
#define PREFIX DBL
#define LIT(x) (x)
#define TYPE_STR "double"
#define ULP_IDX ULP_DBL
#define LITM(x) x
#define FTOSTR strfromd
#define snan_value_MACRO SNAN
#define TEST_FLOATN 0
#define FUNC_NARROW_PREFIX d
