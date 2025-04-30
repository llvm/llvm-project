/* Helper macros for functions returning a narrower type.  F128-specific.
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
   <http://www.gnu.org/licenses/>.  */

#if __HAVE_FLOAT64X_LONG_DOUBLE
# define libm_alias_float32_float128(func)	\
  libm_alias_float32_float128_main (func)
# define libm_alias_float64_float128(func)	\
  libm_alias_float64_float128_main (func)
#else
# define libm_alias_float32_float128(func)			\
  libm_alias_float32_float128_main (func)			\
  weak_alias (__f32 ## func ## f128, f32 ## func ## f64x)
# define libm_alias_float64_float128(func)			\
  libm_alias_float64_float128_main (func)			\
  weak_alias (__f64 ## func ## f128, f64 ## func ## f64x)	\
  weak_alias (__f64 ## func ## f128, f32x ## func ## f64x)
#endif
