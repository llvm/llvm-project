/*
 * IBM Accurate Mathematical Library
 * written by International Business Machines Corp.
 * Copyright (C) 2001-2021 Free Software Foundation, Inc.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 2.1 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program; if not, see <https://www.gnu.org/licenses/>.
 */
/************************************************************************/
/*  MODULE_NAME: atnat2.c                                               */
/*                                                                      */
/*  FUNCTIONS: uatan2                                                   */
/*             signArctan2                                              */
/*                                                                      */
/*  FILES NEEDED: dla.h endian.h mydefs.h atnat2.h                      */
/*                uatan.tbl                                             */
/*                                                                      */
/************************************************************************/

#include <dla.h>
#include "mydefs.h"
#include "uatan.tbl"
#include "atnat2.h"
#include <fenv.h>
#include <float.h>
#include <math.h>
#include <math-barriers.h>
#include <math_private.h>
#include <fenv_private.h>
#include <libm-alias-finite.h>

#ifndef SECTION
# define SECTION
#endif

#define  TWO52     0x1.0p52
#define  TWOM1022  0x1.0p-1022

  /* Fix the sign and return after stage 1 or stage 2 */
static double
signArctan2 (double y, double z)
{
  return copysign (z, y);
}

/* atan2 with max ULP of ~0.524 based on random sampling.  */
double
SECTION
__ieee754_atan2 (double y, double x)
{
  int i, de, ux, dx, uy, dy;
  double ax, ay, u, du, v, vv, dv, t1, t2, t3,
	 z, zz, cor;
  mynumber num;

  static const int ep = 59768832,      /*  57*16**5   */
		   em = -59768832;      /* -57*16**5   */

  /* x=NaN or y=NaN */
  num.d = x;
  ux = num.i[HIGH_HALF];
  dx = num.i[LOW_HALF];
  if ((ux & 0x7ff00000) == 0x7ff00000)
    {
      if (((ux & 0x000fffff) | dx) != 0x00000000)
	return x + y;
    }
  num.d = y;
  uy = num.i[HIGH_HALF];
  dy = num.i[LOW_HALF];
  if ((uy & 0x7ff00000) == 0x7ff00000)
    {
      if (((uy & 0x000fffff) | dy) != 0x00000000)
	return y + y;
    }

  /* y=+-0 */
  if (uy == 0x00000000)
    {
      if (dy == 0x00000000)
	{
	  if ((ux & 0x80000000) == 0x00000000)
	    return 0;
	  else
	    return opi.d;
	}
    }
  else if (uy == 0x80000000)
    {
      if (dy == 0x00000000)
	{
	  if ((ux & 0x80000000) == 0x00000000)
	    return -0.0;
	  else
	    return mopi.d;
	}
    }

  /* x=+-0 */
  if (x == 0)
    {
      if ((uy & 0x80000000) == 0x00000000)
	return hpi.d;
      else
	return mhpi.d;
    }

  /* x=+-INF */
  if (ux == 0x7ff00000)
    {
      if (dx == 0x00000000)
	{
	  if (uy == 0x7ff00000)
	    {
	      if (dy == 0x00000000)
		return qpi.d;
	    }
	  else if (uy == 0xfff00000)
	    {
	      if (dy == 0x00000000)
		return mqpi.d;
	    }
	  else
	    {
	      if ((uy & 0x80000000) == 0x00000000)
		return 0;
	      else
		return -0.0;
	    }
	}
    }
  else if (ux == 0xfff00000)
    {
      if (dx == 0x00000000)
	{
	  if (uy == 0x7ff00000)
	    {
	      if (dy == 0x00000000)
		return tqpi.d;
	    }
	  else if (uy == 0xfff00000)
	    {
	      if (dy == 0x00000000)
		return mtqpi.d;
	    }
	  else
	    {
	      if ((uy & 0x80000000) == 0x00000000)
		return opi.d;
	      else
		return mopi.d;
	    }
	}
    }

  /* y=+-INF */
  if (uy == 0x7ff00000)
    {
      if (dy == 0x00000000)
	return hpi.d;
    }
  else if (uy == 0xfff00000)
    {
      if (dy == 0x00000000)
	return mhpi.d;
    }

  SET_RESTORE_ROUND (FE_TONEAREST);
  /* either x/y or y/x is very close to zero */
  ax = (x < 0) ? -x : x;
  ay = (y < 0) ? -y : y;
  de = (uy & 0x7ff00000) - (ux & 0x7ff00000);
  if (de >= ep)
    {
      return ((y > 0) ? hpi.d : mhpi.d);
    }
  else if (de <= em)
    {
      if (x > 0)
	{
	  double ret;
	  z = ay / ax;
	  ret = signArctan2 (y, z);
	  if (fabs (ret) < DBL_MIN)
	    {
	      double vret = ret ? ret : DBL_MIN;
	      double force_underflow = vret * vret;
	      math_force_eval (force_underflow);
	    }
	  return ret;
	}
      else
	{
	  return ((y > 0) ? opi.d : mopi.d);
	}
    }

  /* if either x or y is extremely close to zero, scale abs(x), abs(y). */
  if (ax < twom500.d || ay < twom500.d)
    {
      ax *= two500.d;
      ay *= two500.d;
    }

  /* Likewise for large x and y.  */
  if (ax > two500.d || ay > two500.d)
    {
      ax *= twom500.d;
      ay *= twom500.d;
    }

  /* x,y which are neither special nor extreme */
  if (ay < ax)
    {
      u = ay / ax;
      EMULV (ax, u, v, vv);
      du = ((ay - v) - vv) / ax;
    }
  else
    {
      u = ax / ay;
      EMULV (ay, u, v, vv);
      du = ((ax - v) - vv) / ay;
    }

  if (x > 0)
    {
      /* (i)   x>0, abs(y)< abs(x):  atan(ay/ax) */
      if (ay < ax)
	{
	  if (u < inv16.d)
	    {
	      v = u * u;

	      zz = du + u * v * (d3.d
				 + v * (d5.d
					+ v * (d7.d
					       + v * (d9.d
						      + v * (d11.d
							     + v * d13.d)))));

	      z = u + zz;
	      /* Max ULP is 0.504.  */
	      return signArctan2 (y, z);
	    }

	  i = (TWO52 + 256 * u) - TWO52;
	  i -= 16;
	  t3 = u - cij[i][0].d;
	  EADD (t3, du, v, dv);
	  t1 = cij[i][1].d;
	  t2 = cij[i][2].d;
	  zz = v * t2 + (dv * t2
			 + v * v * (cij[i][3].d
				    + v * (cij[i][4].d
					   + v * (cij[i][5].d
						  + v * cij[i][6].d))));
	  z = t1 + zz;
	  /* Max ULP is 0.56.  */
	  return signArctan2 (y, z);
	}

      /* (ii)  x>0, abs(x)<=abs(y):  pi/2-atan(ax/ay) */
      if (u < inv16.d)
	{
	  v = u * u;
	  zz = u * v * (d3.d
			+ v * (d5.d
			       + v * (d7.d
				      + v * (d9.d
					     + v * (d11.d
						    + v * d13.d)))));
	  ESUB (hpi.d, u, t2, cor);
	  t3 = ((hpi1.d + cor) - du) - zz;
	  z = t2 + t3;
	  /* Max ULP is 0.501.  */
	  return signArctan2 (y, z);
	}

      i = (TWO52 + 256 * u) - TWO52;
      i -= 16;
      v = (u - cij[i][0].d) + du;

      zz = hpi1.d - v * (cij[i][2].d
			 + v * (cij[i][3].d
				+ v * (cij[i][4].d
				       + v * (cij[i][5].d
					      + v * cij[i][6].d))));
      t1 = hpi.d - cij[i][1].d;
      z = t1 + zz;
      /* Max ULP is 0.503.  */
      return signArctan2 (y, z);
    }

  /* (iii) x<0, abs(x)< abs(y):  pi/2+atan(ax/ay) */
  if (ax < ay)
    {
      if (u < inv16.d)
	{
	  v = u * u;
	  zz = u * v * (d3.d
			+ v * (d5.d
			       + v * (d7.d
				      + v * (d9.d
					     + v * (d11.d + v * d13.d)))));
	  EADD (hpi.d, u, t2, cor);
	  t3 = ((hpi1.d + cor) + du) + zz;
	  z = t2 + t3;
	  /* Max ULP is 0.501.  */
	  return signArctan2 (y, z);
	}

      i = (TWO52 + 256 * u) - TWO52;
      i -= 16;
      v = (u - cij[i][0].d) + du;
      zz = hpi1.d + v * (cij[i][2].d
			 + v * (cij[i][3].d
				+ v * (cij[i][4].d
				       + v * (cij[i][5].d
					      + v * cij[i][6].d))));
      t1 = hpi.d + cij[i][1].d;
      z = t1 + zz;
      /* Max ULP is 0.503.  */
      return signArctan2 (y, z);
    }

  /* (iv)  x<0, abs(y)<=abs(x):  pi-atan(ax/ay) */
  if (u < inv16.d)
    {
      v = u * u;
      zz = u * v * (d3.d
		    + v * (d5.d
			   + v * (d7.d
				  + v * (d9.d + v * (d11.d + v * d13.d)))));
      ESUB (opi.d, u, t2, cor);
      t3 = ((opi1.d + cor) - du) - zz;
      z = t2 + t3;
      /* Max ULP is 0.501.  */
      return signArctan2 (y, z);
    }

  i = (TWO52 + 256 * u) - TWO52;
  i -= 16;
  v = (u - cij[i][0].d) + du;
  zz = opi1.d - v * (cij[i][2].d
		     + v * (cij[i][3].d
			    + v * (cij[i][4].d
				   + v * (cij[i][5].d + v * cij[i][6].d))));
  t1 = opi.d - cij[i][1].d;
  z = t1 + zz;
  /* Max ULP is 0.502.  */
  return signArctan2 (y, z);
}

#ifndef __ieee754_atan2
libm_alias_finite (__ieee754_atan2, __atan2)
#endif
