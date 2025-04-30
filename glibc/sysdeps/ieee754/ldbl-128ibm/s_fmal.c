/* Compute x * y + z as ternary operation.
   Copyright (C) 2011-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by David Flaherty <flaherty@linux.vnet.ibm.com>.

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

#include <fenv.h>
#include <float.h>
#include <math.h>
#include <math-barriers.h>
#include <math_private.h>
#include <fenv_private.h>
#include <math-underflow.h>
#include <math_ldbl_opt.h>
#include <mul_split.h>
#include <stdlib.h>

/* Calculate X + Y exactly and store the result in *HI + *LO.  It is
   given that |X| >= |Y| and the values are small enough that no
   overflow occurs.  */

static void
add_split (double *hi, double *lo, double x, double y)
{
  /* Apply Dekker's algorithm.  */
  *hi = x + y;
  *lo = (x - *hi) + y;
}

/* Value with extended range, used in intermediate computations.  */
typedef struct
{
  /* Value in [0.5, 1), as from frexp, or 0.  */
  double val;
  /* Exponent of power of 2 it is multiplied by, or 0 for zero.  */
  int exp;
} ext_val;

/* Store D as an ext_val value.  */

static void
store_ext_val (ext_val *v, double d)
{
  v->val = __frexp (d, &v->exp);
}

/* Store X * Y as ext_val values *V0 and *V1.  */

static void
mul_ext_val (ext_val *v0, ext_val *v1, double x, double y)
{
  int xexp, yexp;
  x = __frexp (x, &xexp);
  y = __frexp (y, &yexp);
  double hi, lo;
  mul_split (&hi, &lo, x, y);
  store_ext_val (v0, hi);
  if (hi != 0)
    v0->exp += xexp + yexp;
  store_ext_val (v1, lo);
  if (lo != 0)
    v1->exp += xexp + yexp;
}

/* Compare absolute values of ext_val values pointed to by P and Q for
   qsort.  */

static int
compare (const void *p, const void *q)
{
  const ext_val *pe = p;
  const ext_val *qe = q;
  if (pe->val == 0)
    return qe->val == 0 ? 0 : -1;
  else if (qe->val == 0)
    return 1;
  else if (pe->exp < qe->exp)
    return -1;
  else if (pe->exp > qe->exp)
    return 1;
  else
    {
      double pd = fabs (pe->val);
      double qd = fabs (qe->val);
      if (pd < qd)
	return -1;
      else if (pd == qd)
	return 0;
      else
	return 1;
    }
}

/* Calculate *X + *Y exactly, storing the high part in *X (rounded to
   nearest) and the low part in *Y.  It is given that |X| >= |Y|.  */

static void
add_split_ext (ext_val *x, ext_val *y)
{
  int xexp = x->exp, yexp = y->exp;
  if (y->val == 0 || xexp - yexp > 53)
    return;
  double hi = x->val;
  double lo = __scalbn (y->val, yexp - xexp);
  add_split (&hi, &lo, hi, lo);
  store_ext_val (x, hi);
  if (hi != 0)
    x->exp += xexp;
  store_ext_val (y, lo);
  if (lo != 0)
    y->exp += xexp;
}

long double
__fmal (long double x, long double y, long double z)
{
  double xhi, xlo, yhi, ylo, zhi, zlo;
  int64_t hx, hy, hz;
  int xexp, yexp, zexp;
  double scale_val;
  int scale_exp;
  ldbl_unpack (x, &xhi, &xlo);
  EXTRACT_WORDS64 (hx, xhi);
  xexp = (hx & 0x7ff0000000000000LL) >> 52;
  ldbl_unpack (y, &yhi, &ylo);
  EXTRACT_WORDS64 (hy, yhi);
  yexp = (hy & 0x7ff0000000000000LL) >> 52;
  ldbl_unpack (z, &zhi, &zlo);
  EXTRACT_WORDS64 (hz, zhi);
  zexp = (hz & 0x7ff0000000000000LL) >> 52;

  /* If z is Inf or NaN, but x and y are finite, avoid any exceptions
     from computing x * y.  */
  if (zexp == 0x7ff && xexp != 0x7ff && yexp != 0x7ff)
    return (z + x) + y;

  /* If z is zero and x are y are nonzero, compute the result as x * y
     to avoid the wrong sign of a zero result if x * y underflows to
     0.  */
  if (z == 0 && x != 0 && y != 0)
    return x * y;

  /* If x or y or z is Inf/NaN, or if x * y is zero, compute as x * y
     + z.  */
  if (xexp == 0x7ff || yexp == 0x7ff || zexp == 0x7ff
      || x == 0 || y == 0)
    return (x * y) + z;

  {
    SET_RESTORE_ROUND (FE_TONEAREST);

    ext_val vals[10];
    store_ext_val (&vals[0], zhi);
    store_ext_val (&vals[1], zlo);
    mul_ext_val (&vals[2], &vals[3], xhi, yhi);
    mul_ext_val (&vals[4], &vals[5], xhi, ylo);
    mul_ext_val (&vals[6], &vals[7], xlo, yhi);
    mul_ext_val (&vals[8], &vals[9], xlo, ylo);
    qsort (vals, 10, sizeof (ext_val), compare);
    /* Add up the values so that each element of VALS has absolute
       value at most equal to the last set bit of the next nonzero
       element.  */
    for (size_t i = 0; i <= 8; i++)
      {
	add_split_ext (&vals[i + 1], &vals[i]);
	qsort (vals + i + 1, 9 - i, sizeof (ext_val), compare);
      }
    /* Add up the values in the other direction, so that each element
       of VALS has absolute value less than 5ulp of the next
       value.  */
    size_t dstpos = 9;
    for (size_t i = 1; i <= 9; i++)
      {
	if (vals[dstpos].val == 0)
	  {
	    vals[dstpos] = vals[9 - i];
	    vals[9 - i].val = 0;
	    vals[9 - i].exp = 0;
	  }
	else
	  {
	    add_split_ext (&vals[dstpos], &vals[9 - i]);
	    if (vals[9 - i].val != 0)
	      {
		if (9 - i < dstpos - 1)
		  {
		    vals[dstpos - 1] = vals[9 - i];
		    vals[9 - i].val = 0;
		    vals[9 - i].exp = 0;
		  }
		dstpos--;
	      }
	  }
      }
    /* If the result is an exact zero, it results from adding two
       values with opposite signs; recompute in the original rounding
       mode.  */
    if (vals[9].val == 0)
      goto zero_out;
    /* Adding the top three values will now give a result as accurate
       as the underlying long double arithmetic.  */
    add_split_ext (&vals[9], &vals[8]);
    if (compare (&vals[8], &vals[7]) < 0)
      {
	ext_val tmp = vals[7];
	vals[7] = vals[8];
	vals[8] = tmp;
      }
    add_split_ext (&vals[8], &vals[7]);
    add_split_ext (&vals[9], &vals[8]);
    if (vals[9].exp > DBL_MAX_EXP || vals[9].exp < DBL_MIN_EXP)
      {
	/* Overflow or underflow, with the result depending on the
	   original rounding mode, but not on the low part computed
	   here.  */
	scale_val = vals[9].val;
	scale_exp = vals[9].exp;
	goto scale_out;
      }
    double hi = __scalbn (vals[9].val, vals[9].exp);
    double lo = __scalbn (vals[8].val, vals[8].exp);
    /* It is possible that the low part became subnormal and was
       rounded so that the result is no longer canonical.  */
    ldbl_canonicalize (&hi, &lo);
    long double ret = ldbl_pack (hi, lo);
    math_check_force_underflow (ret);
    return ret;
  }

 scale_out:
  scale_val = math_opt_barrier (scale_val);
  scale_val = __scalbn (scale_val, scale_exp);
  if (fabs (scale_val) == DBL_MAX)
    return copysignl (LDBL_MAX, scale_val);
  math_check_force_underflow (scale_val);
  return scale_val;

 zero_out:;
  double zero = 0.0;
  zero = math_opt_barrier (zero);
  return zero - zero;
}
#if IS_IN (libm)
long_double_symbol (libm, __fmal, fmal);
#else
long_double_symbol (libc, __fmal, fmal);
#endif
