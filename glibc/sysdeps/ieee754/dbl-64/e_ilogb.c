/* @(#)s_ilogb.c 5.1 93/09/24 */
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

#if defined(LIBM_SCCS) && !defined(lint)
static char rcsid[] = "$NetBSD: s_ilogb.c,v 1.9 1995/05/10 20:47:28 jtc Exp $";
#endif

/* ilogb(double x)
 * return the binary exponent of non-zero x
 * ilogb(0) = FP_ILOGB0
 * ilogb(NaN) = FP_ILOGBNAN (no signal is raised)
 * ilogb(+-Inf) = INT_MAX (no signal is raised)
 */

#include <limits.h>
#include <math.h>
#include <math_private.h>

int
__ieee754_ilogb (double x)
{
  int32_t hx, lx, ix;

  GET_HIGH_WORD (hx, x);
  hx &= 0x7fffffff;
  if (hx < 0x00100000)
    {
      GET_LOW_WORD (lx, x);
      if ((hx | lx) == 0)
	return FP_ILOGB0;               /* ilogb(0) = FP_ILOGB0 */
      else                              /* subnormal x */
      if (hx == 0)
	{
	  for (ix = -1043; lx > 0; lx <<= 1)
	    ix -= 1;
	}
      else
	{
	  for (ix = -1022, hx <<= 11; hx > 0; hx <<= 1)
	    ix -= 1;
	}
      return ix;
    }
  else if (hx < 0x7ff00000)
    return (hx >> 20) - 1023;
  else if (FP_ILOGBNAN != INT_MAX)
    {
      /* ISO C99 requires ilogb(+-Inf) == INT_MAX.  */
      GET_LOW_WORD (lx, x);
      if (((hx ^ 0x7ff00000) | lx) == 0)
	return INT_MAX;
    }
  return FP_ILOGBNAN;
}
