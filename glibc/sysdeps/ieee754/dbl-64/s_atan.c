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
/*  MODULE_NAME: atnat.c                                                */
/*                                                                      */
/*  FUNCTIONS:  uatan                                                   */
/*              signArctan                                              */
/*                                                                      */
/*  FILES NEEDED: dla.h endian.h mydefs.h atnat.h                       */
/*                uatan.tbl                                             */
/*                                                                      */
/************************************************************************/

#include <dla.h>
#include "mydefs.h"
#include "uatan.tbl"
#include "atnat.h"
#include <fenv.h>
#include <float.h>
#include <libm-alias-double.h>
#include <math.h>
#include <fenv_private.h>
#include <math-underflow.h>

#define  TWO52     0x1.0p52

  /* Fix the sign of y and return */
static double
__signArctan (double x, double y)
{
  return copysign (y, x);
}

/* atan with max ULP of ~0.523 based on random sampling.  */
double
__atan (double x)
{
  double cor, t1, t2, t3, u,
	 v, w, ww, y, yy, z;
  int i, ux, dx;
  mynumber num;

  num.d = x;
  ux = num.i[HIGH_HALF];
  dx = num.i[LOW_HALF];

  /* x=NaN */
  if (((ux & 0x7ff00000) == 0x7ff00000)
      && (((ux & 0x000fffff) | dx) != 0x00000000))
    return x + x;

  /* Regular values of x, including denormals +-0 and +-INF */
  SET_RESTORE_ROUND (FE_TONEAREST);
  u = (x < 0) ? -x : x;
  if (u < C)
    {
      if (u < B)
	{
	  if (u < A)
	    {
	      math_check_force_underflow_nonneg (u);
	      return x;
	    }
	  else
	    {			/* A <= u < B */
	      v = x * x;
	      yy = d11.d + v * d13.d;
	      yy = d9.d + v * yy;
	      yy = d7.d + v * yy;
	      yy = d5.d + v * yy;
	      yy = d3.d + v * yy;
	      yy *= x * v;

	      y = x + yy;
	      /* Max ULP is 0.511.  */
	      return y;
	    }
	}
      else
	{			/* B <= u < C */
	  i = (TWO52 + 256 * u) - TWO52;
	  i -= 16;
	  z = u - cij[i][0].d;
	  yy = cij[i][5].d + z * cij[i][6].d;
	  yy = cij[i][4].d + z * yy;
	  yy = cij[i][3].d + z * yy;
	  yy = cij[i][2].d + z * yy;
	  yy *= z;

	  t1 = cij[i][1].d;
	  y = t1 + yy;
	  /* Max ULP is 0.56.  */
	  return __signArctan (x, y);
	}
    }
  else
    {
      if (u < D)
	{			/* C <= u < D */
	  w = 1 / u;
	  EMULV (w, u, t1, t2);
	  ww = w * ((1 - t1) - t2);
	  i = (TWO52 + 256 * w) - TWO52;
	  i -= 16;
	  z = (w - cij[i][0].d) + ww;

	  yy = cij[i][5].d + z * cij[i][6].d;
	  yy = cij[i][4].d + z * yy;
	  yy = cij[i][3].d + z * yy;
	  yy = cij[i][2].d + z * yy;
	  yy = HPI1 - z * yy;

	  t1 = HPI - cij[i][1].d;
	  y = t1 + yy;
	  /* Max ULP is 0.503.  */
	  return __signArctan (x, y);
	}
      else
	{
	  if (u < E)
	    {                   /* D <= u < E */
	      w = 1 / u;
	      v = w * w;
	      EMULV (w, u, t1, t2);

	      yy = d11.d + v * d13.d;
	      yy = d9.d + v * yy;
	      yy = d7.d + v * yy;
	      yy = d5.d + v * yy;
	      yy = d3.d + v * yy;
	      yy *= w * v;

	      ww = w * ((1 - t1) - t2);
	      ESUB (HPI, w, t3, cor);
	      yy = ((HPI1 + cor) - ww) - yy;
	      y = t3 + yy;
	      /* Max ULP is 0.5003.  */
	      return __signArctan (x, y);
	    }
	  else
	    {
	      /* u >= E */
	      if (x > 0)
		return HPI;
	      else
		return MHPI;
	    }
	}
    }
}

#ifndef __atan
libm_alias_double (__atan, atan)
#endif
