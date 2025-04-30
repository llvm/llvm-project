/* s_logbl.c -- long double version of s_logb.c.
 * Conversion to IEEE quad long double by Jakub Jelinek, jj@ultra.linux.cz.
 */

/*
 * ====================================================
 * Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
 *
 * Developed at SunPro, a Sun Microsystems, Inc. business.
 * Permission to use, copy, modify, and distribute this
 * software is freely granted, provided that this notice
 * is preserved.
 * ====================================================
 */

/*
 * long double logbl(x)
 * IEEE 754 logb. Included to pass IEEE test suite. Not recommend.
 * Use ilogb instead.
 */

#include <math.h>
#include <math_private.h>
#include <math_ldbl_opt.h>
#include <fix-int-fp-convert-zero.h>

long double
__logbl (long double x)
{
  int64_t hx, hxs, rhx;
  double xhi, xlo;

  ldbl_unpack (x, &xhi, &xlo);
  EXTRACT_WORDS64 (hx, xhi);
  hxs = hx;
  hx &= 0x7fffffffffffffffLL;	/* high |x| */
  if (hx == 0)
    return -1.0 / fabs (x);
  if (hx >= 0x7ff0000000000000LL)
    return x * x;
  if (__glibc_unlikely ((rhx = hx >> 52) == 0))
    {
      /* POSIX specifies that denormal number is treated as
         though it were normalized.  */
      rhx -= __builtin_clzll (hx) - 12;
    }
  else if ((hx & 0x000fffffffffffffLL) == 0)
    {
      /* If the high part is a power of 2, and the low part is nonzero
	 with the opposite sign, the low part affects the
	 exponent.  */
      int64_t lx;
      EXTRACT_WORDS64 (lx, xlo);
      if ((hxs ^ lx) < 0 && (lx & 0x7fffffffffffffffLL) != 0)
	rhx--;
    }
  if (FIX_INT_FP_CONVERT_ZERO && rhx == 1023)
    return 0.0L;
  return (long double) (rhx - 1023);
}
#ifndef __logbl
long_double_symbol (libm, __logbl, logbl);
#endif
