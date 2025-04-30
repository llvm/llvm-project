/* Quad-precision floating point cosine on <-pi/4,pi/4>.
   Copyright (C) 1999-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Jakub Jelinek <jj@ultra.linux.cz>

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

#include <math.h>
#include <math_private.h>

static const long double c[] = {
#define ONE c[0]
 1.00000000000000000000000000000000000E+00L, /* 3fff0000000000000000000000000000 */

/* cos x ~ ONE + x^2 ( SCOS1 + SCOS2 * x^2 + ... + SCOS4 * x^6 + SCOS5 * x^8 )
   x in <0,1/256>  */
#define SCOS1 c[1]
#define SCOS2 c[2]
#define SCOS3 c[3]
#define SCOS4 c[4]
#define SCOS5 c[5]
-5.00000000000000000000000000000000000E-01L, /* bffe0000000000000000000000000000 */
 4.16666666666666666666666666556146073E-02L, /* 3ffa5555555555555555555555395023 */
-1.38888888888888888888309442601939728E-03L, /* bff56c16c16c16c16c16a566e42c0375 */
 2.48015873015862382987049502531095061E-05L, /* 3fefa01a01a019ee02dcf7da2d6d5444 */
-2.75573112601362126593516899592158083E-07L, /* bfe927e4f5dce637cb0b54908754bde0 */

/* cos x ~ ONE + x^2 ( COS1 + COS2 * x^2 + ... + COS7 * x^12 + COS8 * x^14 )
   x in <0,0.1484375>  */
#define COS1 c[6]
#define COS2 c[7]
#define COS3 c[8]
#define COS4 c[9]
#define COS5 c[10]
#define COS6 c[11]
#define COS7 c[12]
#define COS8 c[13]
-4.99999999999999999999999999999999759E-01L, /* bffdfffffffffffffffffffffffffffb */
 4.16666666666666666666666666651287795E-02L, /* 3ffa5555555555555555555555516f30 */
-1.38888888888888888888888742314300284E-03L, /* bff56c16c16c16c16c16c16a463dfd0d */
 2.48015873015873015867694002851118210E-05L, /* 3fefa01a01a01a01a0195cebe6f3d3a5 */
-2.75573192239858811636614709689300351E-07L, /* bfe927e4fb7789f5aa8142a22044b51f */
 2.08767569877762248667431926878073669E-09L, /* 3fe21eed8eff881d1e9262d7adff4373 */
-1.14707451049343817400420280514614892E-11L, /* bfda9397496922a9601ed3d4ca48944b */
 4.77810092804389587579843296923533297E-14L, /* 3fd2ae5f8197cbcdcaf7c3fb4523414c */

/* sin x ~ ONE * x + x^3 ( SSIN1 + SSIN2 * x^2 + ... + SSIN4 * x^6 + SSIN5 * x^8 )
   x in <0,1/256>  */
#define SSIN1 c[14]
#define SSIN2 c[15]
#define SSIN3 c[16]
#define SSIN4 c[17]
#define SSIN5 c[18]
-1.66666666666666666666666666666666659E-01L, /* bffc5555555555555555555555555555 */
 8.33333333333333333333333333146298442E-03L, /* 3ff81111111111111111111110fe195d */
-1.98412698412698412697726277416810661E-04L, /* bff2a01a01a01a01a019e7121e080d88 */
 2.75573192239848624174178393552189149E-06L, /* 3fec71de3a556c640c6aaa51aa02ab41 */
-2.50521016467996193495359189395805639E-08L, /* bfe5ae644ee90c47dc71839de75b2787 */
};

#define SINCOSL_COS_HI 0
#define SINCOSL_COS_LO 1
#define SINCOSL_SIN_HI 2
#define SINCOSL_SIN_LO 3
extern const long double __sincosl_table[];

long double
__kernel_cosl(long double x, long double y)
{
  long double h, l, z, sin_l, cos_l_m1;
  int64_t ix;
  uint32_t tix, hix, index;
  double xhi, hhi;

  xhi = ldbl_high (x);
  EXTRACT_WORDS64 (ix, xhi);
  tix = ((uint64_t)ix) >> 32;
  tix &= ~0x80000000;			/* tix = |x|'s high 32 bits */
  if (tix < 0x3fc30000)			/* |x| < 0.1484375 */
    {
      /* Argument is small enough to approximate it by a Chebyshev
	 polynomial of degree 16.  */
      if (tix < 0x3c600000)		/* |x| < 2^-57 */
	if (!((int)x)) return ONE;	/* generate inexact */
      z = x * x;
      return ONE + (z*(COS1+z*(COS2+z*(COS3+z*(COS4+
		    z*(COS5+z*(COS6+z*(COS7+z*COS8))))))));
    }
  else
    {
      /* So that we don't have to use too large polynomial,  we find
	 l and h such that x = l + h,  where fabsl(l) <= 1.0/256 with 83
	 possible values for h.  We look up cosl(h) and sinl(h) in
	 pre-computed tables,  compute cosl(l) and sinl(l) using a
	 Chebyshev polynomial of degree 10(11) and compute
	 cosl(h+l) = cosl(h)cosl(l) - sinl(h)sinl(l).  */
      int six = tix;
      tix = ((six - 0x3ff00000) >> 4) + 0x3fff0000;
      index = 0x3ffe - (tix >> 16);
      hix = (tix + (0x200 << index)) & (0xfffffc00 << index);
      x = fabsl (x);
      switch (index)
	{
	case 0: index = ((45 << 10) + hix - 0x3ffe0000) >> 8; break;
	case 1: index = ((13 << 11) + hix - 0x3ffd0000) >> 9; break;
	default:
	case 2: index = (hix - 0x3ffc3000) >> 10; break;
	}
      hix = (hix << 4) & 0x3fffffff;
/*
    The following should work for double but generates the wrong index.
    For now the code above converts double to ieee extended to compute
    the index back to double for the h value.

      index = 0x3fe - (tix >> 20);
      hix = (tix + (0x200 << index)) & (0xfffffc00 << index);
      if (signbit (x))
	{
	  x = -x;
	  y = -y;
	}
      switch (index)
	{
	case 0: index = ((45 << 14) + hix - 0x3fe00000) >> 12; break;
	case 1: index = ((13 << 15) + hix - 0x3fd00000) >> 13; break;
	default:
	case 2: index = (hix - 0x3fc30000) >> 14; break;
	}
*/
      INSERT_WORDS64 (hhi, ((uint64_t)hix) << 32);
      h = hhi;
      l = y - (h - x);
      z = l * l;
      sin_l = l*(ONE+z*(SSIN1+z*(SSIN2+z*(SSIN3+z*(SSIN4+z*SSIN5)))));
      cos_l_m1 = z*(SCOS1+z*(SCOS2+z*(SCOS3+z*(SCOS4+z*SCOS5))));
      return __sincosl_table [index + SINCOSL_COS_HI]
	     + (__sincosl_table [index + SINCOSL_COS_LO]
		- (__sincosl_table [index + SINCOSL_SIN_HI] * sin_l
		   - __sincosl_table [index + SINCOSL_COS_HI] * cos_l_m1));
    }
}
