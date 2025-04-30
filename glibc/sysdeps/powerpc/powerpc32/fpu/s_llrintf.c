/* Round a float value to a long long in the current rounding mode.
   Copyright (C) 1997-2021 Free Software Foundation, Inc.
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

#include <math.h>
#include <math_private.h>
#include <stdint.h>
#include <libm-alias-float.h>

long long int
__llrintf (float x)
{
#ifdef _ARCH_PWR4
  /* Assume powerpc64 instructions availability.  */
  long long int ret;
  __asm__ ("fctid %0, %1" : "=d" (ret) : "d" (x));
  return ret;
#else
  float rx = rintf (x);
  if (HAVE_PPC_FCTIDZ || rx != x)
    return (long long int) rx;
  else
    {
      float arx = fabsf (rx);
      /* Avoid incorrect exceptions from libgcc conversions (as of GCC
	 5): <https://gcc.gnu.org/bugzilla/show_bug.cgi?id=59412>.  */
      if (arx < 0x1p31f)
	return (long long int) (long int) rx;
      else if (!(arx < 0x1p55f))
	return (long long int) (long int) (rx * 0x1p-32f) << 32;
      uint32_t i0;
      GET_FLOAT_WORD (i0, rx);
      int exponent = ((i0 >> 23) & 0xff) - 0x7f;
      unsigned long long int mant = (i0 & 0x7fffff) | 0x800000;
      mant <<= exponent - 23;
      return (long long int) ((i0 & 0x80000000) != 0 ? -mant : mant);
    }
#endif
}
libm_alias_float (__llrint, llrint)
