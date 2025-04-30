/* Common utilities for testing strtod and its derivatives.
   This file is part of the GNU C Library.
   Copyright (C) 2016-2021 Free Software Foundation, Inc.

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

#ifndef _TST_STRTOD_H
#define _TST_STRTOD_H

#define FSTRLENMAX 128

#include <bits/floatn.h>

#define F16 __f16 ()
#define F32 __f32 ()
#define F64 __f64 ()
#define F128 __f128 ()
#define F32X __f32x ()
#define F64X __f64x ()
#define F128X __f128x ()

/* Test strfromfN and strtofN on all platforms that provide them,
   whether or not the type _FloatN is ABI-distinct from other types;
   likewise _FloatNx functions.  */
#if __HAVE_FLOAT16
# define IF_FLOAT16(x) x
#else
# define IF_FLOAT16(x)
#endif

#if __HAVE_FLOAT32
# define IF_FLOAT32(x) x
#else
# define IF_FLOAT32(x)
#endif

#if __HAVE_FLOAT64
# define IF_FLOAT64(x) x
#else
# define IF_FLOAT64(x)
#endif

#if __HAVE_FLOAT128
# define IF_FLOAT128(x) x
#else
# define IF_FLOAT128(x)
#endif

#if __HAVE_FLOAT32X
# define IF_FLOAT32X(x) x
#else
# define IF_FLOAT32X(x)
#endif

#if __HAVE_FLOAT64X
# define IF_FLOAT64X(x) x
#else
# define IF_FLOAT64X(x)
#endif

#if __HAVE_FLOAT128X
# define IF_FLOAT128X(x) x
#else
# define IF_FLOAT128X(x)
#endif

/* Provide an extra parameter expansion for mfunc.  */
#define MMFUNC(mmfunc, ...) mmfunc (__VA_ARGS__)

/* Splat n variants of the same test for the various strtod functions.  */
#define GEN_TEST_STRTOD_FOREACH(mfunc, ...)				      \
  mfunc (  f,       float, strfromf, f, f, ##__VA_ARGS__)		      \
  mfunc (  d,      double, strfromd,  ,  , ##__VA_ARGS__)		      \
  mfunc ( ld, long double, strfroml, L, l, ##__VA_ARGS__)		      \
  IF_FLOAT16 (MMFUNC							      \
   (mfunc, f16, _Float16, strfromf16, F16, f16, ##__VA_ARGS__))		      \
  IF_FLOAT32 (MMFUNC							      \
   (mfunc, f32, _Float32, strfromf32, F32, f32, ##__VA_ARGS__))		      \
  IF_FLOAT64 (MMFUNC							      \
   (mfunc, f64, _Float64, strfromf64, F64, f64, ##__VA_ARGS__))		      \
  IF_FLOAT128 (MMFUNC							      \
   (mfunc, f128, _Float128, strfromf128, F128, f128, ##__VA_ARGS__))	      \
  IF_FLOAT32X (MMFUNC							      \
   (mfunc, f32x, _Float32x, strfromf32x, F32X, f32x, ##__VA_ARGS__))	      \
  IF_FLOAT64X (MMFUNC							      \
   (mfunc, f64x, _Float64x, strfromf64x, F64X, f64x, ##__VA_ARGS__))	      \
  IF_FLOAT128X (MMFUNC							      \
   (mfunc, f128x, _Float128x, strfromf128x, F128X, f128x, ##__VA_ARGS__))
/* The arguments to the generated macros are:
   FSUF - Function suffix
   FTYPE - float type
   FTOSTR - float to string func
   LSUF - Literal suffix
   CSUF - C standardish suffix for many of the math functions
*/



#define STRTOD_TEST_FOREACH(mfunc, ...)				\
({								\
   int result = 0;						\
   result |= mfunc ## f  (__VA_ARGS__);				\
   result |= mfunc ## d  (__VA_ARGS__);				\
   result |= mfunc ## ld (__VA_ARGS__);				\
   IF_FLOAT16 (result |= mfunc ## f16 (__VA_ARGS__));		\
   IF_FLOAT32 (result |= mfunc ## f32 (__VA_ARGS__));		\
   IF_FLOAT64 (result |= mfunc ## f64 (__VA_ARGS__));		\
   IF_FLOAT128 (result |= mfunc ## f128 (__VA_ARGS__));		\
   IF_FLOAT32X (result |= mfunc ## f32x (__VA_ARGS__));		\
   IF_FLOAT64X (result |= mfunc ## f64x (__VA_ARGS__));		\
   IF_FLOAT128X (result |= mfunc ## f128x (__VA_ARGS__));	\
   result;							\
})


#endif
