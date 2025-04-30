/* Fix for conversion of integer 0 to floating point.  Generic version.
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

#ifndef FIX_INT_FP_CONVERT_ZERO_H
#define FIX_INT_FP_CONVERT_ZERO_H	1

/* Define this macro to 1 to work around conversions of integer 0 to
   floating point returning -0 instead of the correct +0 in some
   rounding modes.  */
#define FIX_INT_FP_CONVERT_ZERO 0

#endif /* fix-int-fp-convert-zero.h */
