/* Manipulation of the bit representation of 'long double' quantities.
   Copyright (C) 2006-2021 Free Software Foundation, Inc.
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

#ifndef _MATH_LDBL_H_PPC_
#define _MATH_LDBL_H_PPC_ 1

/* GCC does not optimize the default ldbl_pack code to not spill register
   in the stack. The following optimization tells gcc that pack/unpack
   is really a nop.  We use fr1/fr2 because those are the regs used to
   pass/return a single long double arg.  */
static inline long double
ldbl_pack_ppc (double a, double aa)
{
  register long double x __asm__ ("fr1");
  register double xh __asm__ ("fr1");
  register double xl __asm__ ("fr2");
  xh = a;
  xl = aa;
  __asm__ ("" : "=f" (x) : "f" (xh), "f" (xl));
  return x;
}

static inline void
ldbl_unpack_ppc (long double l, double *a, double *aa)
{
  register long double x __asm__ ("fr1");
  register double xh __asm__ ("fr1");
  register double xl __asm__ ("fr2");
  x = l;
  __asm__ ("" : "=f" (xh), "=f" (xl) : "f" (x));
  *a = xh;
  *aa = xl;
}

#define ldbl_pack   ldbl_pack_ppc
#define ldbl_unpack ldbl_unpack_ppc

#include <sysdeps/ieee754/ldbl-128ibm/math_ldbl.h>

#endif /* math_ldbl.h */
