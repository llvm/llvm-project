/* @(#)e_log10.c 5.1 93/09/24 */
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

/* __ieee754_log10(x)
 * Return the base 10 logarithm of x
 *
 * Method :
 *	Let log10_2hi = leading 40 bits of log10(2) and
 *	    log10_2lo = log10(2) - log10_2hi,
 *	    ivln10   = 1/log(10) rounded.
 *	Then
 *		n = ilogb(x),
 *		if(n<0)  n = n+1;
 *		x = scalbn(x,-n);
 *		log10(x) := n*log10_2hi + (n*log10_2lo + ivln10*log(x))
 *
 * Note 1:
 *	To guarantee log10(10**n)=n, where 10**n is normal, the rounding
 *	mode must set to Round-to-Nearest.
 * Note 2:
 *	[1/log(10)] rounded to 53 bits has error  .198   ulps;
 *	log10 is monotonic at all binary break points.
 *
 * Special cases:
 *	log10(x) is NaN with signal if x < 0;
 *	log10(+INF) is +INF with no signal; log10(0) is -INF with signal;
 *	log10(NaN) is that NaN with no signal;
 *	log10(10**N) = N  for N=0,1,...,22.
 *
 * Constants:
 * The hexadecimal values are the intended ones for the following constants.
 * The decimal values may be used, provided that the compiler will convert
 * from decimal to binary accurately enough to produce the hexadecimal values
 * shown.
 */

#include <math.h>
#include <fix-int-fp-convert-zero.h>
#include <math_private.h>
#include <stdint.h>
#include <libm-alias-finite.h>

static const double two54 = 1.80143985094819840000e+16;		/* 0x4350000000000000 */
static const double ivln10 = 4.34294481903251816668e-01;	/* 0x3FDBCB7B1526E50E */
static const double log10_2hi = 3.01029995663611771306e-01;	/* 0x3FD34413509F6000 */
static const double log10_2lo = 3.69423907715893078616e-13;	/* 0x3D59FEF311F12B36 */

double
__ieee754_log10 (double x)
{
  double y, z;
  int64_t i, hx;
  int32_t k;

  EXTRACT_WORDS64 (hx, x);

  k = 0;
  if (hx < INT64_C(0x0010000000000000))
    {				/* x < 2**-1022  */
      if (__glibc_unlikely ((hx & UINT64_C(0x7fffffffffffffff)) == 0))
	return -two54 / fabs (x);	/* log(+-0)=-inf */
      if (__glibc_unlikely (hx < 0))
	return (x - x) / (x - x);	/* log(-#) = NaN */
      k -= 54;
      x *= two54;		/* subnormal number, scale up x */
      EXTRACT_WORDS64 (hx, x);
    }
  /* scale up resulted in a NaN number  */
  if (__glibc_unlikely (hx >= UINT64_C(0x7ff0000000000000)))
    return x + x;
  k += (hx >> 52) - 1023;
  i = ((uint64_t) k & UINT64_C(0x8000000000000000)) >> 63;
  hx = (hx & UINT64_C(0x000fffffffffffff)) | ((0x3ff - i) << 52);
  y = (double) (k + i);
  if (FIX_INT_FP_CONVERT_ZERO && y == 0.0)
    y = 0.0;
  INSERT_WORDS64 (x, hx);
  z = y * log10_2lo + ivln10 * __ieee754_log (x);
  return z + y * log10_2hi;
}
libm_alias_finite (__ieee754_log10, __log10)
