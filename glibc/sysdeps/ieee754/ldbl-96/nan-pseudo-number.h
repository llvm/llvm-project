/* Pseudo-normal number handling.  Generic version.
   Copyright (C) 2020-2021 Free Software Foundation, Inc.
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

#ifndef NAN_PSEUDO_NUMBER_H
#define NAN_PSEUDO_NUMBER_H	1

/* Default is to assume that pseudo numbers are not signaling.  */
static inline int
is_pseudo_signaling (uint32_t exi, uint32_t hxi)
{
  return 0;
}

#endif /* nan-pseudo-number.h */
