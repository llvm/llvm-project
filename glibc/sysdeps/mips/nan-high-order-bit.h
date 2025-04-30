/* Specify NaN high-order bit conventions.  MIPS version.
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

#ifdef __mips_nan2008
/* MIPS aligned to IEEE 754-2008.  */
# define HIGH_ORDER_BIT_IS_SET_FOR_SNAN 0
#else
/* One of the few architectures where the meaning of the
   quiet/signaling bit is inverse to IEEE 754-2008 (as well as common
   practice for IEEE 754-1985).  */
# define HIGH_ORDER_BIT_IS_SET_FOR_SNAN 1
#endif

#endif /* nan-high-order-bit.h */
