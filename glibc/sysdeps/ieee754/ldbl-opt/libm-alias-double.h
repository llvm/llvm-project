/* Define aliases for libm double functions.  ldbl-opt version.
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

#ifndef _LIBM_ALIAS_DOUBLE_H
#define _LIBM_ALIAS_DOUBLE_H

#include <bits/floatn.h>
#include <math_ldbl_opt.h>
#include <first-versions.h>
#include <ldbl-compat-choose.h>

#if __HAVE_FLOAT64 && !__HAVE_DISTINCT_FLOAT64
# define libm_alias_double_other_r_f64(from, to, r)	\
  weak_alias (from ## r, to ## f64 ## r)
#else
# define libm_alias_double_other_r_f64(from, to, r)
#endif

#if __HAVE_FLOAT32X && !__HAVE_DISTINCT_FLOAT32X
# define libm_alias_double_other_r_f32x(from, to, r)	\
  weak_alias (from ## r, to ## f32x ## r)
#else
# define libm_alias_double_other_r_f32x(from, to, r)
#endif

/* Define _FloatN / _FloatNx aliases for a double libm function that
   has internal name FROM ## R and public names TO ## suffix ## R for
   each suffix of a supported _FloatN / _FloatNx floating-point type
   with the same format as double.  */
#define libm_alias_double_other_r(from, to, r)	\
  libm_alias_double_other_r_f64 (from, to, r);	\
  libm_alias_double_other_r_f32x (from, to, r)

/* Likewise, but without the R suffix.  */
#define libm_alias_double_other(from, to)	\
  libm_alias_double_other_r (from, to, )

/* Define aliases for a double libm function that has internal name
   FROM ## R and public names TO ## suffix ## R for each suffix of a
   supported floating-point type with the same format as double.  This
   should only be used for functions where such public names exist for
   _FloatN types, not for implementation-namespace exported names
   (where there is one name per format, not per type) or for
   obsolescent functions not provided for _FloatN types.  */
#define libm_alias_double_r(from, to, r)			\
  weak_alias (from ## r, to ## r);				\
  LONG_DOUBLE_COMPAT_CHOOSE_libm_ ## to ## l ## r		\
    (compat_symbol (libm,					\
		    from ## r,					\
		    to ## l ## r,				\
		    FIRST_VERSION_libm_ ## to ## l ## r), );	\
  libm_alias_double_other_r (from, to, r)

/* Likewise, but without the R suffix.  */
#define libm_alias_double(from, to) libm_alias_double_r (from, to, )

#endif
