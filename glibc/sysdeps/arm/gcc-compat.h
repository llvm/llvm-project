/* Macros for checking required GCC compatibility.  ARM version.
   Copyright (C) 2014-2021 Free Software Foundation, Inc.
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

#ifndef _ARM_GCC_COMPAT_H
#define _ARM_GCC_COMPAT_H 1

#ifndef GCC_COMPAT_VERSION
# ifdef __ARM_PCS_VFP
/* The hard-float ABI was first supported in 4.5.  */
#  define GCC_COMPAT_VERSION    GCC_VERSION (4, 5)
# else
/* The EABI configurations (the only ones we handle) were first supported
   in 4.1.  */
#  define GCC_COMPAT_VERSION    GCC_VERSION (4, 1)
# endif
#endif

#include_next <gcc-compat.h>

#endif
