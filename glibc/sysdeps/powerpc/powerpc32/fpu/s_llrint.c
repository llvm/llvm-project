/* Round a double value to a long long in the current rounding mode.
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

#include <limits.h>
#include <math.h>
#include <math_ldbl_opt.h>
#include <math_private.h>
#include <stdint.h>
#include <libm-alias-double.h>

long long int
__llrint (double x)
{
#ifdef _ARCH_PWR4
  /* Assume powerpc64 instructions availability.  */
  long long int ret;
  __asm__ ("fctid %0, %1" : "=d" (ret) : "d" (x));
  return ret;
#else
  double rx = rint (x);
  if (HAVE_PPC_FCTIDZ || rx != x)
    return (long long int) rx;
  else
    {
      /* Avoid incorrect exceptions from libgcc conversions (as of GCC
	 5): <https://gcc.gnu.org/bugzilla/show_bug.cgi?id=59412>.  */
      if (fabs (rx) < 0x1p31)
	return (long long int) (long int) rx;
      uint64_t i0;
      EXTRACT_WORDS64 (i0, rx);
      int exponent = ((i0 >> 52) & 0x7ff) - 0x3ff;
      if (exponent < 63)
	{
	  unsigned long long int mant
	    = (i0 & ((1ULL << 52) - 1)) | (1ULL << 52);
	  if (exponent < 52)
	    mant >>= 52 - exponent;
	  else
	    mant <<= exponent - 52;
	  return (long long int) ((i0 & (1ULL << 63)) != 0 ? -mant : mant);
	}
      else if (rx == (double) LLONG_MIN)
	return LLONG_MIN;
      else
	return (long long int) (long int) rx << 32;
    }
#endif
}
#ifndef __llrint
libm_alias_double (__llrint, llrint)
#endif
