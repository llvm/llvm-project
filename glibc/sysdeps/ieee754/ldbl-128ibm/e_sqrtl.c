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
/*********************************************************************/
/* MODULE_NAME: uroot.c                                              */
/*                                                                   */
/* FUNCTION:    usqrt                                                */
/*                                                                   */
/* FILES NEEDED: dla.h endian.h mydefs.h uroot.h                     */
/*               uroot.tbl                                           */
/*                                                                   */
/* An ultimate sqrt routine. Given an IEEE double machine number x   */
/* it computes the correctly rounded (to nearest) value of square    */
/* root of x.                                                        */
/* Assumption: Machine arithmetic operations are performed in        */
/* round to nearest mode of IEEE 754 standard.                       */
/*                                                                   */
/*********************************************************************/

#include <math_private.h>
#include <libm-alias-finite.h>

typedef union {int64_t i[2]; long double x; double d[2]; } mynumber;

static const double
  t512 = 0x1p512,
  tm256 = 0x1p-256,
  two54 = 0x1p54,	/* 0x4350000000000000 */
  twom54 = 0x1p-54;	/* 0x3C90000000000000 */

/*********************************************************************/
/* An ultimate sqrt routine. Given an IEEE double machine number x   */
/* it computes the correctly rounded (to nearest) value of square    */
/* root of x.                                                        */
/*********************************************************************/
long double __ieee754_sqrtl(long double x)
{
  static const long double big = 134217728.0, big1 = 134217729.0;
  long double t,s,i;
  mynumber a,c;
  uint64_t k, l;
  int64_t m, n;
  double d;

  a.x=x;
  k=a.i[0] & INT64_C(0x7fffffffffffffff);
  /*----------------- 2^-1022  <= | x |< 2^1024  -----------------*/
  if (k>INT64_C(0x000fffff00000000) && k<INT64_C(0x7ff0000000000000)) {
    if (x < 0) return (big1-big1)/(big-big);
    l = (k&INT64_C(0x001fffffffffffff))|INT64_C(0x3fe0000000000000);
    if ((a.i[1] & INT64_C(0x7fffffffffffffff)) != 0) {
      n = (int64_t) ((l - k) * 2) >> 53;
      m = (a.i[1] >> 52) & 0x7ff;
      if (m == 0) {
	a.d[1] *= two54;
	m = ((a.i[1] >> 52) & 0x7ff) - 54;
      }
      m += n;
      if (m > 0)
	a.i[1] = (a.i[1] & INT64_C(0x800fffffffffffff)) | (m << 52);
      else if (m <= -54) {
	a.i[1] &= INT64_C(0x8000000000000000);
      } else {
	m += 54;
	a.i[1] = (a.i[1] & INT64_C(0x800fffffffffffff)) | (m << 52);
	a.d[1] *= twom54;
      }
    }
    a.i[0] = l;
    s = a.x;
    d = __ieee754_sqrt (a.d[0]);
    c.i[0] = INT64_C(0x2000000000000000)+((k&INT64_C(0x7fe0000000000000))>>1);
    c.i[1] = 0;
    i = d;
    t = 0.5L * (i + s / i);
    i = 0.5L * (t + s / t);
    return c.x * i;
  }
  else {
    if (k>=INT64_C(0x7ff0000000000000))
      /* sqrt (-Inf) = NaN, sqrt (NaN) = NaN, sqrt (+Inf) = +Inf.  */
      return x * x + x;
    if (x == 0) return x;
    if (x < 0) return (big1-big1)/(big-big);
    return tm256*__ieee754_sqrtl(x*t512);
  }
}
libm_alias_finite (__ieee754_sqrtl, __sqrtl)
