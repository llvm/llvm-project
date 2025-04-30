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

/* Expansions and modifications for 128-bit long double are
   Copyright (C) 2001 Stephen L. Moshier <moshier@na-net.ornl.gov>
   and are incorporated herein by permission of the author.  The author
   reserves the right to distribute this material elsewhere under different
   copying permissions.  These modifications are distributed here under
   the following terms:

    This library is free software; you can redistribute it and/or
    modify it under the terms of the GNU Lesser General Public
    License as published by the Free Software Foundation; either
    version 2.1 of the License, or (at your option) any later version.

    This library is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with this library; if not, see
    <https://www.gnu.org/licenses/>.  */

/* __ieee754_powl(x,y) return x**y
 *
 *		      n
 * Method:  Let x =  2   * (1+f)
 *	1. Compute and return log2(x) in two pieces:
 *		log2(x) = w1 + w2,
 *	   where w1 has 113-53 = 60 bit trailing zeros.
 *	2. Perform y*log2(x) = n+y' by simulating muti-precision
 *	   arithmetic, where |y'|<=0.5.
 *	3. Return x**y = 2**n*exp(y'*log2)
 *
 * Special cases:
 *	1.  (anything) ** 0  is 1
 *	2.  (anything) ** 1  is itself
 *	3.  (anything) ** NAN is NAN
 *	4.  NAN ** (anything except 0) is NAN
 *	5.  +-(|x| > 1) **  +INF is +INF
 *	6.  +-(|x| > 1) **  -INF is +0
 *	7.  +-(|x| < 1) **  +INF is +0
 *	8.  +-(|x| < 1) **  -INF is +INF
 *	9.  +-1         ** +-INF is NAN
 *	10. +0 ** (+anything except 0, NAN)               is +0
 *	11. -0 ** (+anything except 0, NAN, odd integer)  is +0
 *	12. +0 ** (-anything except 0, NAN)               is +INF
 *	13. -0 ** (-anything except 0, NAN, odd integer)  is +INF
 *	14. -0 ** (odd integer) = -( +0 ** (odd integer) )
 *	15. +INF ** (+anything except 0,NAN) is +INF
 *	16. +INF ** (-anything except 0,NAN) is +0
 *	17. -INF ** (anything)  = -0 ** (-anything)
 *	18. (-anything) ** (integer) is (-1)**(integer)*(+anything**integer)
 *	19. (-anything except 0 and inf) ** (non-integer) is NAN
 *
 */

#include <math.h>
#include <math_private.h>
#include <math-underflow.h>
#include <libm-alias-finite.h>

static const long double bp[] = {
  1.0L,
  1.5L,
};

/* log_2(1.5) */
static const long double dp_h[] = {
  0.0,
  5.8496250072115607565592654282227158546448E-1L
};

/* Low part of log_2(1.5) */
static const long double dp_l[] = {
  0.0,
  1.0579781240112554492329533686862998106046E-16L
};

static const long double zero = 0.0L,
  one = 1.0L,
  two = 2.0L,
  two113 = 1.0384593717069655257060992658440192E34L,
  huge = 1.0e300L,
  tiny = 1.0e-300L;

/* 3/2 log x = 3 z + z^3 + z^3 (z^2 R(z^2))
   z = (x-1)/(x+1)
   1 <= x <= 1.25
   Peak relative error 2.3e-37 */
static const long double LN[] =
{
 -3.0779177200290054398792536829702930623200E1L,
  6.5135778082209159921251824580292116201640E1L,
 -4.6312921812152436921591152809994014413540E1L,
  1.2510208195629420304615674658258363295208E1L,
 -9.9266909031921425609179910128531667336670E-1L
};
static const long double LD[] =
{
 -5.129862866715009066465422805058933131960E1L,
  1.452015077564081884387441590064272782044E2L,
 -1.524043275549860505277434040464085593165E2L,
  7.236063513651544224319663428634139768808E1L,
 -1.494198912340228235853027849917095580053E1L
  /* 1.0E0 */
};

/* exp(x) = 1 + x - x / (1 - 2 / (x - x^2 R(x^2)))
   0 <= x <= 0.5
   Peak relative error 5.7e-38  */
static const long double PN[] =
{
  5.081801691915377692446852383385968225675E8L,
  9.360895299872484512023336636427675327355E6L,
  4.213701282274196030811629773097579432957E4L,
  5.201006511142748908655720086041570288182E1L,
  9.088368420359444263703202925095675982530E-3L,
};
static const long double PD[] =
{
  3.049081015149226615468111430031590411682E9L,
  1.069833887183886839966085436512368982758E8L,
  8.259257717868875207333991924545445705394E5L,
  1.872583833284143212651746812884298360922E3L,
  /* 1.0E0 */
};

static const long double
  /* ln 2 */
  lg2 = 6.9314718055994530941723212145817656807550E-1L,
  lg2_h = 6.9314718055994528622676398299518041312695E-1L,
  lg2_l = 2.3190468138462996154948554638754786504121E-17L,
  ovt = 8.0085662595372944372e-0017L,
  /* 2/(3*log(2)) */
  cp = 9.6179669392597560490661645400126142495110E-1L,
  cp_h = 9.6179669392597555432899980587535537779331E-1L,
  cp_l = 5.0577616648125906047157785230014751039424E-17L;

long double
__ieee754_powl (long double x, long double y)
{
  long double z, ax, z_h, z_l, p_h, p_l;
  long double y1, t1, t2, r, s, sgn, t, u, v, w;
  long double s2, s_h, s_l, t_h, t_l, ay;
  int32_t i, j, k, yisint, n;
  uint32_t ix, iy;
  int32_t hx, hy, hax;
  double ohi, xhi, xlo, yhi, ylo;
  uint32_t lx, ly, lj;

  ldbl_unpack (x, &xhi, &xlo);
  EXTRACT_WORDS (hx, lx, xhi);
  ix = hx & 0x7fffffff;

  ldbl_unpack (y, &yhi, &ylo);
  EXTRACT_WORDS (hy, ly, yhi);
  iy = hy & 0x7fffffff;

  /* y==zero: x**0 = 1 */
  if ((iy | ly) == 0 && !issignaling (x))
    return one;

  /* 1.0**y = 1; -1.0**+-Inf = 1 */
  if (x == one && !issignaling (y))
    return one;
  if (x == -1.0L && ((iy - 0x7ff00000) | ly) == 0)
    return one;

  /* +-NaN return x+y */
  if ((ix >= 0x7ff00000 && ((ix - 0x7ff00000) | lx) != 0)
      || (iy >= 0x7ff00000 && ((iy - 0x7ff00000) | ly) != 0))
    return x + y;

  /* determine if y is an odd int when x < 0
   * yisint = 0       ... y is not an integer
   * yisint = 1       ... y is an odd int
   * yisint = 2       ... y is an even int
   */
  yisint = 0;
  if (hx < 0)
    {
      uint32_t low_ye;

      GET_HIGH_WORD (low_ye, ylo);
      if ((low_ye & 0x7fffffff) >= 0x43400000)	/* Low part >= 2^53 */
	yisint = 2;		/* even integer y */
      else if (iy >= 0x3ff00000)	/* 1.0 */
	{
	  if (floorl (y) == y)
	    {
	      z = 0.5 * y;
	      if (floorl (z) == z)
		yisint = 2;
	      else
		yisint = 1;
	    }
	}
    }

  ax = fabsl (x);

  /* special value of y */
  if (ly == 0)
    {
      if (iy == 0x7ff00000)	/* y is +-inf */
	{
	  if (ax > one)
	    /* (|x|>1)**+-inf = inf,0 */
	    return (hy >= 0) ? y : zero;
	  else
	    /* (|x|<1)**-,+inf = inf,0 */
	    return (hy < 0) ? -y : zero;
	}
      if (ylo == 0.0)
	{
	  if (iy == 0x3ff00000)
	    {			/* y is  +-1 */
	      if (hy < 0)
		return one / x;
	      else
		return x;
	    }
	  if (hy == 0x40000000)
	    return x * x;		/* y is  2 */
	  if (hy == 0x3fe00000)
	    {			/* y is  0.5 */
	      if (hx >= 0)		/* x >= +0 */
		return sqrtl (x);
	    }
	}
    }

  /* special value of x */
  if (lx == 0)
    {
      if (ix == 0x7ff00000 || ix == 0 || (ix == 0x3ff00000 && xlo == 0.0))
	{
	  z = ax;		/*x is +-0,+-inf,+-1 */
	  if (hy < 0)
	    z = one / z;	/* z = (1/|x|) */
	  if (hx < 0)
	    {
	      if (((ix - 0x3ff00000) | yisint) == 0)
		{
		  z = (z - z) / (z - z);	/* (-1)**non-int is NaN */
		}
	      else if (yisint == 1)
		z = -z;		/* (x<0)**odd = -(|x|**odd) */
	    }
	  return z;
	}
    }

  /* (x<0)**(non-int) is NaN */
  if (((((uint32_t) hx >> 31) - 1) | yisint) == 0)
    return (x - x) / (x - x);

  /* sgn (sign of result -ve**odd) = -1 else = 1 */
  sgn = one;
  if (((((uint32_t) hx >> 31) - 1) | (yisint - 1)) == 0)
    sgn = -one;			/* (-ve)**(odd int) */

  /* |y| is huge.
     2^-16495 = 1/2 of smallest representable value.
     If (1 - 1/131072)^y underflows, y > 1.4986e9 */
  if (iy > 0x41d654b0)
    {
      /* if (1 - 2^-113)^y underflows, y > 1.1873e38 */
      if (iy > 0x47d654b0)
	{
	  if (ix <= 0x3fefffff)
	    return (hy < 0) ? sgn * huge * huge : sgn * tiny * tiny;
	  if (ix >= 0x3ff00000)
	    return (hy > 0) ? sgn * huge * huge : sgn * tiny * tiny;
	}
      /* over/underflow if x is not close to one */
      if (ix < 0x3fefffff)
	return (hy < 0) ? sgn * huge * huge : sgn * tiny * tiny;
      if (ix > 0x3ff00000)
	return (hy > 0) ? sgn * huge * huge : sgn * tiny * tiny;
    }

  ay = y > 0 ? y : -y;
  if (ay < 0x1p-117)
    y = y < 0 ? -0x1p-117 : 0x1p-117;

  n = 0;
  /* take care subnormal number */
  if (ix < 0x00100000)
    {
      ax *= two113;
      n -= 113;
      ohi = ldbl_high (ax);
      GET_HIGH_WORD (ix, ohi);
    }
  n += ((ix) >> 20) - 0x3ff;
  j = ix & 0x000fffff;
  /* determine interval */
  ix = j | 0x3ff00000;		/* normalize ix */
  if (j <= 0x39880)
    k = 0;			/* |x|<sqrt(3/2) */
  else if (j < 0xbb670)
    k = 1;			/* |x|<sqrt(3)   */
  else
    {
      k = 0;
      n += 1;
      ix -= 0x00100000;
    }

  ohi = ldbl_high (ax);
  GET_HIGH_WORD (hax, ohi);
  ax = __scalbnl (ax, ((int) ((ix - hax) * 2)) >> 21);

  /* compute s = s_h+s_l = (x-1)/(x+1) or (x-1.5)/(x+1.5) */
  u = ax - bp[k];		/* bp[0]=1.0, bp[1]=1.5 */
  v = one / (ax + bp[k]);
  s = u * v;
  s_h = ldbl_high (s);

  /* t_h=ax+bp[k] High */
  t_h = ax + bp[k];
  t_h = ldbl_high (t_h);
  t_l = ax - (t_h - bp[k]);
  s_l = v * ((u - s_h * t_h) - s_h * t_l);
  /* compute log(ax) */
  s2 = s * s;
  u = LN[0] + s2 * (LN[1] + s2 * (LN[2] + s2 * (LN[3] + s2 * LN[4])));
  v = LD[0] + s2 * (LD[1] + s2 * (LD[2] + s2 * (LD[3] + s2 * (LD[4] + s2))));
  r = s2 * s2 * u / v;
  r += s_l * (s_h + s);
  s2 = s_h * s_h;
  t_h = 3.0 + s2 + r;
  t_h = ldbl_high (t_h);
  t_l = r - ((t_h - 3.0) - s2);
  /* u+v = s*(1+...) */
  u = s_h * t_h;
  v = s_l * t_h + t_l * s;
  /* 2/(3log2)*(s+...) */
  p_h = u + v;
  p_h = ldbl_high (p_h);
  p_l = v - (p_h - u);
  z_h = cp_h * p_h;		/* cp_h+cp_l = 2/(3*log2) */
  z_l = cp_l * p_h + p_l * cp + dp_l[k];
  /* log2(ax) = (s+..)*2/(3*log2) = n + dp_h + z_h + z_l */
  t = (long double) n;
  t1 = (((z_h + z_l) + dp_h[k]) + t);
  t1 = ldbl_high (t1);
  t2 = z_l - (((t1 - t) - dp_h[k]) - z_h);

  /* split up y into y1+y2 and compute (y1+y2)*(t1+t2) */
  y1 = ldbl_high (y);
  p_l = (y - y1) * t1 + y * t2;
  p_h = y1 * t1;
  z = p_l + p_h;
  ohi = ldbl_high (z);
  EXTRACT_WORDS (j, lj, ohi);
  if (j >= 0x40d00000) /* z >= 16384 */
    {
      /* if z > 16384 */
      if (((j - 0x40d00000) | lj) != 0)
	return sgn * huge * huge;	/* overflow */
      else
	{
	  if (p_l + ovt > z - p_h)
	    return sgn * huge * huge;	/* overflow */
	}
    }
  else if ((j & 0x7fffffff) >= 0x40d01b90)	/* z <= -16495 */
    {
      /* z < -16495 */
      if (((j - 0xc0d01bc0) | lj) != 0)
	return sgn * tiny * tiny;	/* underflow */
      else
	{
	  if (p_l <= z - p_h)
	    return sgn * tiny * tiny;	/* underflow */
	}
    }
  /* compute 2**(p_h+p_l) */
  i = j & 0x7fffffff;
  k = (i >> 20) - 0x3ff;
  n = 0;
  if (i > 0x3fe00000)
    {				/* if |z| > 0.5, set n = [z+0.5] */
      n = floorl (z + 0.5L);
      t = n;
      p_h -= t;
    }
  t = p_l + p_h;
  t = ldbl_high (t);
  u = t * lg2_h;
  v = (p_l - (t - p_h)) * lg2 + t * lg2_l;
  z = u + v;
  w = v - (z - u);
  /*  exp(z) */
  t = z * z;
  u = PN[0] + t * (PN[1] + t * (PN[2] + t * (PN[3] + t * PN[4])));
  v = PD[0] + t * (PD[1] + t * (PD[2] + t * (PD[3] + t)));
  t1 = z - t * u / v;
  r = (z * t1) / (t1 - two) - (w + z * w);
  z = one - (r - z);
  z = __scalbnl (sgn * z, n);
  math_check_force_underflow (z);
  return z;
}
libm_alias_finite (__ieee754_powl, __powl)
