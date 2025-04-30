/* Fix for conversion of floating point to integer overflow.  Generic version.
   Copyright (C) 2015-2021 Free Software Foundation, Inc.
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

/* Define these macros to 1 to workaround conversions of out-of-range
   floating-point numbers to integer types failing to raise the
   "invalid" exception, or raising spurious "inexact" or other
   exceptions.  */
#define FIX_FLT_LONG_CONVERT_OVERFLOW 0
#define FIX_FLT_LLONG_CONVERT_OVERFLOW 0
#define FIX_DBL_LONG_CONVERT_OVERFLOW 0
#define FIX_DBL_LLONG_CONVERT_OVERFLOW 0
#define FIX_LDBL_LONG_CONVERT_OVERFLOW 0
#define FIX_LDBL_LLONG_CONVERT_OVERFLOW 0
#define FIX_FLT128_LONG_CONVERT_OVERFLOW 0
#define FIX_FLT128_LLONG_CONVERT_OVERFLOW 0

#endif /* fix-fp-int-convert-overflow.h */
