/* Determine floating-point rounding mode within libc.  powerpc64 version.
   Copyright (C) 2019-2021 Free Software Foundation, Inc.
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

#ifndef _POWERPC64_GET_ROUNDING_MODE_H
#define _POWERPC64_GET_ROUNDING_MODE_H	1

#include <fenv.h>
#include <fpu_control.h>

/* Return the floating-point rounding mode.  */

static inline int
get_rounding_mode (void)
{
  return _FPU_GET_RC ();
}

#endif /* get-rounding-mode.h */
