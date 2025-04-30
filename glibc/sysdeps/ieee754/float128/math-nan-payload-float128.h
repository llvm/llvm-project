/* NaN payload handling for _Float128.
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

#include <ieee754_float128.h>

#define SET_NAN_PAYLOAD(flt, mant)			\
  do							\
    {							\
      union ieee854_float128 u;				\
      u.d = (flt);					\
      u.ieee_nan.mantissa0 = 0;				\
      u.ieee_nan.mantissa1 = 0;				\
      u.ieee_nan.mantissa2 = (mant) >> 32;		\
      u.ieee_nan.mantissa3 = (mant);			\
      if ((u.ieee.mantissa0 | u.ieee.mantissa1		\
	   | u.ieee.mantissa2 | u.ieee.mantissa3) != 0)	\
	(flt) = u.d;					\
    }							\
  while (0)
