/* Common definitions for libm tests for long double.
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

#include <float.h>

#define FUNC(function) function##l
#define FLOAT long double
#define CFLOAT __complex__ long double
#define BUILD_COMPLEX(real, imag) (CMPLXL ((real), (imag)))
#define PREFIX LDBL
#if LDBL_MANT_DIG == DBL_MANT_DIG
# define TYPE_STR "double"
# define ULP_IDX ULP_DBL
/* On architectures which redirect long double to _Float128 ABI, we must
   choose the float128 ulps.  Similarly, on such architectures, the ABI
   used may be dependent on how the compiler was invoked.  */
#elif __LDOUBLE_REDIRECTS_TO_FLOAT128_ABI == 1
# define TYPE_STR "float128"
# define ULP_IDX ULP_FLT128
#else
# define TYPE_STR "ldouble"
# define ULP_IDX ULP_LDBL
#endif
#define LIT(x) (x ## L)
#define LITM(x) x ## l
#define FTOSTR strfroml
#define snan_value_MACRO SNANL
#define TEST_FLOATN 0
