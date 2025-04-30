/* Finite math compatibility macros.
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

#ifndef _LIBM_ALIAS_FINITE_H
#define _LIBM_ALIAS_FINITE_H

#include <first-versions.h>
#include <shlib-compat.h>

/* The -ffinite-math symbols were added on GLIBC 2.15 and moved to compat
   symbol so newer architectures do not require to support it.  */
#if SHLIB_COMPAT (libm, GLIBC_2_15, GLIBC_2_31)
# define libm_alias_finite(from, to)				\
  libm_alias_finite1(from, to)
# define libm_alias_finite1(from, to)				\
compat_symbol (libm,						\
	       from,						\
	       to ## _finite, 					\
	       FIRST_VERSION_libm_ ## to ## _finite);
#else
# define libm_alias_finite(from, to)
#endif

#endif
