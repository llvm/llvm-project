/* s_frexpl.c -- long double version of s_frexp.c.
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

#if defined(LIBM_SCCS) && !defined(lint)
static char rcsid[] = "$NetBSD: $";
#endif

/*
 * for non-zero x
 *	x = frexpl(arg,&exp);
 * return a long double fp quantity x such that 0.5 <= |x| <1.0
 * and the corresponding binary exponent "exp". That is
 *	arg = x*2^exp.
 * If arg is inf, 0.0, or NaN, then frexpl(arg,&exp) returns arg
 * with *exp=0.
 */

#include <math.h>
#include <math_private.h>
#include <math_ldbl_opt.h>

long double __frexpl(long double x, int *eptr)
{
  uint64_t hx, lx, ix, ixl;
  int64_t explo, expon;
  double xhi, xlo;

  ldbl_unpack (x, &xhi, &xlo);
  EXTRACT_WORDS64 (hx, xhi);
  EXTRACT_WORDS64 (lx, xlo);
  ixl = 0x7fffffffffffffffULL & lx;
  ix  = 0x7fffffffffffffffULL & hx;
  expon = 0;
  if (ix >= 0x7ff0000000000000ULL || ix == 0)
    {
      /* 0,inf,nan.  */
      *eptr = expon;
      return x + x;
    }
  expon = ix >> 52;
  if (expon == 0)
    {
      /* Denormal high double, the low double must be 0.0.  */
      int cnt;

      /* Normalize.  */
      if (sizeof (ix) == sizeof (long))
	cnt = __builtin_clzl (ix);
      else if ((ix >> 32) != 0)
	cnt = __builtin_clzl ((long) (ix >> 32));
      else
	cnt = __builtin_clzl ((long) ix) + 32;
      cnt = cnt - 12;
      expon -= cnt;
      ix <<= cnt + 1;
    }
  expon -= 1022;
  ix &= 0x000fffffffffffffULL;
  hx &= 0x8000000000000000ULL;
  hx |= (1022LL << 52) | ix;

  if (ixl != 0)
    {
      /* If the high double is an exact power of two and the low
	 double has the opposite sign, then the exponent calculated
	 from the high double is one too big.  */
      if (ix == 0
	  && (int64_t) (hx ^ lx) < 0)
	{
	  hx += 1LL << 52;
	  expon -= 1;
	}

      explo = ixl >> 52;
      if (explo == 0)
	{
	  /* The low double started out as a denormal.  Normalize its
	     mantissa and adjust the exponent.  */
	  int cnt;

	  if (sizeof (ixl) == sizeof (long))
	    cnt = __builtin_clzl (ixl);
	  else if ((ixl >> 32) != 0)
	    cnt = __builtin_clzl ((long) (ixl >> 32));
	  else
	    cnt = __builtin_clzl ((long) ixl) + 32;
	  cnt = cnt - 12;
	  explo -= cnt;
	  ixl <<= cnt + 1;
	}

      /* With variable precision we can't assume much about the
	 magnitude of the returned low double.  It may even be a
	 denormal.  */
      explo -= expon;
      ixl &= 0x000fffffffffffffULL;
      lx  &= 0x8000000000000000ULL;
      if (explo <= 0)
	{
	  /* Handle denormal low double.  */
	  if (explo > -52)
	    {
	      ixl |= 1LL << 52;
	      ixl >>= 1 - explo;
	    }
	  else
	    {
	      ixl = 0;
	      lx = 0;
	      if ((hx & 0x7ff0000000000000ULL) == (1023LL << 52))
		{
		  /* Oops, the adjustment we made above for values a
		     little smaller than powers of two turned out to
		     be wrong since the returned low double will be
		     zero.  This can happen if the input was
		     something weird like 0x1p1000 - 0x1p-1000.  */
		  hx -= 1LL << 52;
		  expon += 1;
		}
	    }
	  explo = 0;
	}
      lx |= (explo << 52) | ixl;
    }

  INSERT_WORDS64 (xhi, hx);
  INSERT_WORDS64 (xlo, lx);
  x = ldbl_pack (xhi, xlo);
  *eptr = expon;
  return x;
}
#if IS_IN (libm)
long_double_symbol (libm, __frexpl, frexpl);
#else
long_double_symbol (libc, __frexpl, frexpl);
#endif
