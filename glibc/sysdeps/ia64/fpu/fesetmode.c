/* Install given floating-point control modes.  IA64 version.
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

#include <fenv.h>

#define FPSR_STATUS 0x1f80UL
#define FPSR_STATUS_ALL ((FPSR_STATUS << 6) | (FPSR_STATUS << 19) \
			 | (FPSR_STATUS << 32) | (FPSR_STATUS << 45))

int
fesetmode (const femode_t *modep)
{
  femode_t mode;

  /* As in fesetenv.  */
  if (((fenv_t) modep >> 62) == 0x03)
    mode = (femode_t) modep & 0x3fffffffffffffffUL;
  else
    mode = *modep;

  femode_t fpsr;
  __asm__ __volatile__ ("mov.m %0=ar.fpsr" : "=r" (fpsr));
  fpsr = (fpsr & FPSR_STATUS_ALL) | (mode & ~FPSR_STATUS_ALL);
  __asm__ __volatile__ ("mov.m ar.fpsr=%0;;" :: "r" (fpsr));

  return 0;
}
