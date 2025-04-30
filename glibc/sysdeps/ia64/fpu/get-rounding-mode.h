/* Return current rounding direction within libc.  IA64 version.
   Copyright (C) 1999-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Christian Boissat <Christian.Boissat@cern.ch>, 1999.

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

#ifndef IA64_GET_ROUNDING_MODE_H
#define IA64_GET_ROUNDING_MODE_H	1

#include <fenv.h>

/* Return the floating-point rounding mode.  */

static inline int
get_rounding_mode (void)
{
  fenv_t fpsr;

  __asm__ __volatile__ ("mov.m %0=ar.fpsr" : "=r" (fpsr));

  return (fpsr >> 10) & 3;
}

#endif /* get-rounding-mode.h */
