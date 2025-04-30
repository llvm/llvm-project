/* Specify NaN high-order bit conventions.  Generic version.
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

#ifndef NAN_HIGH_ORDER_BIT_H
#define NAN_HIGH_ORDER_BIT_H	1

/* Define this macro to 1 if the high-order bit of a NaN's mantissa is
   set for signaling NaNs and clear for quiet NaNs, 0 otherwise (the
   preferred IEEE convention).  */
#define HIGH_ORDER_BIT_IS_SET_FOR_SNAN 0

#endif /* nan-high-order-bit.h */
