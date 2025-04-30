/* Fix for missing "invalid" exceptions from floating-point
   comparisons.  Generic version.
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

#ifndef FIX_FP_INT_COMPARE_INVALID_H
#define FIX_FP_INT_COMPARE_INVALID_H	1

/* Define this macro to 1 to work around ordered comparison operators
   in C failing to raise the "invalid" exception for NaN operands.  */
#define FIX_COMPARE_INVALID 0

#endif /* fix-fp-int-compare-invalid.h */
