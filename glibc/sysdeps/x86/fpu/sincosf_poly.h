/* Used by sinf, cosf and sincosf functions.  X86-64 version.
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

typedef double v2df_t __attribute__ ((vector_size (2 * sizeof (double))));

#ifdef __SSE2_MATH__
typedef float v4sf_t __attribute__ ((vector_size (4 * sizeof (float))));

static inline void
v2df_to_sf (v2df_t v2df, float *f0p, float *f1p)
{
  v4sf_t v4sf = __builtin_ia32_cvtpd2ps (v2df);
  *f0p = v4sf[0];
  *f1p = v4sf[1];
}
#else
static inline void
v2df_to_sf (v2df_t v2df, float *f0p, float *f1p)
{
  *f0p = (float) v2df[0];
  *f1p = (float) v2df[1];
}
#endif

/* The constants and polynomials for sine and cosine.  */
typedef struct
{
  double sign[4];		/* Sign of sine in quadrants 0..3.  */
  double hpi_inv;		/* 2 / PI ( * 2^24 if !TOINT_INTRINSICS).  */
  double hpi;			/* PI / 2.  */
  /* Cosine polynomial: c0, c1, c2, c3, c4.
     Sine polynomial: s1, s2, s3.  */
  double c0, c1;
  v2df_t s1c2, s2c3, s3c4;
} sincos_t;

/* Compute the sine and cosine of inputs X and X2 (X squared), using the
   polynomial P and store the results in SINP and COSP.  N is the quadrant,
   if odd the cosine and sine polynomials are swapped.  */
static inline void
sincosf_poly (double x, double x2, const sincos_t *p, int n, float *sinp,
	      float *cosp)
{
  v2df_t vx2x2 = { x2, x2 };
  v2df_t vxx2 = { x, x2 };
  v2df_t vx3x4, vs1c2;

  vx3x4 = vx2x2 * vxx2;
  vs1c2 = p->s2c3 + vx2x2 * p->s3c4;

  /* Swap sin/cos result based on quadrant.  */
  if (n & 1)
    {
      float *tmp = cosp;
      cosp = sinp;
      sinp = tmp;
    }

  double c1 = p->c0 + x2 * p->c1;
  v2df_t vxc1 = { x, c1 };
  v2df_t vx5x6 = vx3x4 * vx2x2;

  v2df_t vsincos = vxc1 + vx3x4 * p->s1c2;
  vsincos = vsincos + vx5x6 * vs1c2;
  v2df_to_sf (vsincos, sinp, cosp);
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
      s1 = p->s2c3[0] + x2 * p->s3c4[0];

      x7 = x3 * x2;
      s = x + x3 * p->s1c2[0];

      return s + x7 * s1;
    }
  else
    {
      x4 = x2 * x2;
      c2 = p->s2c3[1] + x2 * p->s3c4[1];
      c1 = p->c0 + x2 * p->c1;

      x6 = x4 * x2;
      c = c1 + x4 * p->s1c2[1];

      return c + x6 * c2;
    }
}
