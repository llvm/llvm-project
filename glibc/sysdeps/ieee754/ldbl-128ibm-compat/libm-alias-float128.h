/* Define aliases for libm _Float128 functions.
   Copyright (C) 2018-2021 Free Software Foundation, Inc.
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

#ifndef _LIBM_ALIAS_FLOAT128_H
#define _LIBM_ALIAS_FLOAT128_H

#include <bits/floatn.h>
#include <ldbl-128ibm-compat-abi.h>

/* This macro should be used on all long double functions that are not part of
   the _Float128 API in order to provide *ieee128 symbols without exposing
   internal *f128 symbols.  */
#define libm_alias_float128_other_r_ldbl(from, to, r) \
  strong_alias (from ## f128 ## r, __ ## to ## ieee128 ## r)

/* Define _FloatN / _FloatNx aliases (other than that for _Float128)
   for a _Float128 libm function that has internal name FROM ## f128
   ## R and public names TO ## suffix ## R for each suffix of a
   supported _FloatN / _FloatNx floating-point type with the same
   format as _Float128.  */
#if __HAVE_FLOAT64X && !__HAVE_FLOAT64X_LONG_DOUBLE
# define libm_alias_float128_other_r(from, to, r) \
  weak_alias (from ## f128 ## r, to ## f64x ## r); \
  libm_alias_float128_other_r_ldbl(from, to, r)
#else
# define libm_alias_float128_other_r(from, to, r) \
  libm_alias_float128_other_r_ldbl(from, to, r)
#endif

/* Likewise, but without the R suffix.  */
#define libm_alias_float128_other(from, to)	\
  libm_alias_float128_other_r (from, to, )

/* Define aliases for a _Float128 libm function that has internal name
   FROM ## f128 ## R and public names TO ## suffix ## R for each
   suffix of a supported floating-point type with the same format as
   _Float128.  This should only be used for functions where such
   public names exist for _FloatN types, not for
   implementation-namespace exported names (where there is one name
   per format, not per type) or for obsolescent functions not provided
   for _FloatN types.  */
#define libm_alias_float128_r(from, to, r)		\
  weak_alias (from ## f128 ## r, to ## f128 ## r);	\
  libm_alias_float128_other_r (from, to, r)

/* Likewise, but without the R suffix.  */
#define libm_alias_float128(from, to) libm_alias_float128_r (from, to, )

#endif
