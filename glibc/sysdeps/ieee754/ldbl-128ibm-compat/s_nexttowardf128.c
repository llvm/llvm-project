/* Provide nexttoward[|f] implementations for IEEE long double.
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
   <http://www.gnu.org/licenses/>.  */

#include <float128_private.h>

/* Build nexttoward functions with binary128 format.  */
#undef weak_alias
#define weak_alias(from, to)
#undef libm_alias_ldouble
#define libm_alias_ldouble(from, to)
#undef __nexttoward
#define __nexttoward __nexttoward_to_ieee128
#include <sysdeps/ieee754/ldbl-128/s_nexttoward.c>

#undef weak_alias
#define weak_alias(from, to)
#undef libm_alias_ldouble
#define libm_alias_ldouble(from, to)
#undef __nexttowardf
#define __nexttowardf __nexttowardf_to_ieee128
#include <sysdeps/ieee754/ldbl-128/s_nexttowardf.c>

#include <libm-alias-ldouble.h>
