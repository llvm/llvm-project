/* Fix for conversion of floating point to integer overflow.  S390 version.
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

#ifndef FIX_FP_INT_CONVERT_OVERFLOW_H
#define FIX_FP_INT_CONVERT_OVERFLOW_H	1

/* GCC emits "convert to fixed" instructions for casting floating point values
   to integer values. These instructions raise invalid and inexact exceptions
   if the floating point value exceeds the integer type ranges.  */
#define FIX_FLT_LLONG_CONVERT_OVERFLOW 1
#define FIX_DBL_LLONG_CONVERT_OVERFLOW 1
#define FIX_LDBL_LLONG_CONVERT_OVERFLOW 1

#define FIX_FLT_LONG_CONVERT_OVERFLOW 1
#define FIX_DBL_LONG_CONVERT_OVERFLOW 1
#define FIX_LDBL_LONG_CONVERT_OVERFLOW 1

#endif /* fix-fp-int-convert-overflow.h */
