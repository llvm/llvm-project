/* Fix for missing "invalid" exceptions from floating-point
   comparisons.  PowerPC version.
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

/* As of GCC 5, comparisons use unordered comparison instructions when
   they should use ordered comparisons
   <https://gcc.gnu.org/bugzilla/show_bug.cgi?id=58684>.  */
#define FIX_COMPARE_INVALID 1

#endif /* fix-fp-int-compare-invalid.h */
