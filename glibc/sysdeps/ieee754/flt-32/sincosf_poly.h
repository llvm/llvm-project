/* Used by sinf, cosf and sincosf functions.
   Copyright (C) 2018-2021 Free Software Foundation, Inc.
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

/* The constants and polynomials for sine and cosine.  */
typedef struct
{
  double sign[4];		/* Sign of sine in quadrants 0..3.  */
  double hpi_inv;		/* 2 / PI ( * 2^24 if !TOINT_INTRINSICS).  */
  double hpi;			/* PI / 2.  */
  double c0, c1, c2, c3, c4;	/* Cosine polynomial.  */
  double s1, s2, s3;		/* Sine polynomial.  */
} sincos_t;

/* Compute the sine and cosine of inputs X and X2 (X squared), using the
   polynomial P and store the results in SINP and COSP.  N is the quadrant,
   if odd the cosine and sine polynomials are swapped.  */
static inline void
sincosf_poly (double x, double x2, const sincos_t *p, int n, float *sinp,
	      float *cosp)
{
  double x3, x4, x5, x6, s, c, c1, c2, s1;

  x4 = x2 * x2;
  x3 = x2 * x;
  c2 = p->c3 + x2 * p->c4;
  s1 = p->s2 + x2 * p->s3;

  /* Swap sin/cos result based on quadrant.  */
  float *tmp = (n & 1 ? cosp : sinp);
  cosp = (n & 1 ? sinp : cosp);
  sinp = tmp;

  c1 = p->c0 + x2 * p->c1;
  x5 = x3 * x2;
  x6 = x4 * x2;

  s = x + x3 * p->s1;
  c = c1 + x4 * p->c2;

  *sinp = s + x5 * s1;
  *cosp = c + x6 * c2;
}

/* Return the sine of inputs X and X2 (X squared) using the polynomial P.
   N is the quadrant, and if odd the cosine polynomial is used.  */
static inline float
sinf_poly (double x, double x2, const sincos_t *p, int n)
{
  double x3, x4, x6, x7, s, c, c1, c2, s1;

  if ((n & 1) == 0)
    {
      x3 = x * x2;
      s1 = p->s2 + x2 * p->s3;

      x7 = x3 * x2;
      s = x + x3 * p->s1;

      return s + x7 * s1;
    }
  else
    {
      x4 = x2 * x2;
      c2 = p->c3 + x2 * p->c4;
      c1 = p->c0 + x2 * p->c1;

      x6 = x4 * x2;
      c = c1 + x4 * p->c2;

      return c + x6 * c2;
    }
}
