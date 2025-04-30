/* Quad-precision floating point sine on <-pi/4,pi/4>.
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

#include <float.h>
#include <math.h>
#include <math_private.h>
#include <math-underflow.h>

static const _Float128 c[] = {
#define ONE c[0]
 L(1.00000000000000000000000000000000000E+00), /* 3fff0000000000000000000000000000 */

/* cos x ~ ONE + x^2 ( SCOS1 + SCOS2 * x^2 + ... + SCOS4 * x^6 + SCOS5 * x^8 )
   x in <0,1/256>  */
#define SCOS1 c[1]
#define SCOS2 c[2]
#define SCOS3 c[3]
#define SCOS4 c[4]
#define SCOS5 c[5]
L(-5.00000000000000000000000000000000000E-01), /* bffe0000000000000000000000000000 */
 L(4.16666666666666666666666666556146073E-02), /* 3ffa5555555555555555555555395023 */
L(-1.38888888888888888888309442601939728E-03), /* bff56c16c16c16c16c16a566e42c0375 */
 L(2.48015873015862382987049502531095061E-05), /* 3fefa01a01a019ee02dcf7da2d6d5444 */
L(-2.75573112601362126593516899592158083E-07), /* bfe927e4f5dce637cb0b54908754bde0 */

/* sin x ~ ONE * x + x^3 ( SIN1 + SIN2 * x^2 + ... + SIN7 * x^12 + SIN8 * x^14 )
   x in <0,0.1484375>  */
#define SIN1 c[6]
#define SIN2 c[7]
#define SIN3 c[8]
#define SIN4 c[9]
#define SIN5 c[10]
#define SIN6 c[11]
#define SIN7 c[12]
#define SIN8 c[13]
L(-1.66666666666666666666666666666666538e-01), /* bffc5555555555555555555555555550 */
 L(8.33333333333333333333333333307532934e-03), /* 3ff811111111111111111111110e7340 */
L(-1.98412698412698412698412534478712057e-04), /* bff2a01a01a01a01a01a019e7a626296 */
 L(2.75573192239858906520896496653095890e-06), /* 3fec71de3a556c7338fa38527474b8f5 */
L(-2.50521083854417116999224301266655662e-08), /* bfe5ae64567f544e16c7de65c2ea551f */
 L(1.60590438367608957516841576404938118e-10), /* 3fde6124613a811480538a9a41957115 */
L(-7.64716343504264506714019494041582610e-13), /* bfd6ae7f3d5aef30c7bc660b060ef365 */
 L(2.81068754939739570236322404393398135e-15), /* 3fce9510115aabf87aceb2022a9a9180 */

/* sin x ~ ONE * x + x^3 ( SSIN1 + SSIN2 * x^2 + ... + SSIN4 * x^6 + SSIN5 * x^8 )
   x in <0,1/256>  */
#define SSIN1 c[14]
#define SSIN2 c[15]
#define SSIN3 c[16]
#define SSIN4 c[17]
#define SSIN5 c[18]
L(-1.66666666666666666666666666666666659E-01), /* bffc5555555555555555555555555555 */
 L(8.33333333333333333333333333146298442E-03), /* 3ff81111111111111111111110fe195d */
L(-1.98412698412698412697726277416810661E-04), /* bff2a01a01a01a01a019e7121e080d88 */
 L(2.75573192239848624174178393552189149E-06), /* 3fec71de3a556c640c6aaa51aa02ab41 */
L(-2.50521016467996193495359189395805639E-08), /* bfe5ae644ee90c47dc71839de75b2787 */
};

#define SINCOSL_COS_HI 0
#define SINCOSL_COS_LO 1
#define SINCOSL_SIN_HI 2
#define SINCOSL_SIN_LO 3
extern const _Float128 __sincosl_table[];

_Float128
__kernel_sinl(_Float128 x, _Float128 y, int iy)
{
  _Float128 h, l, z, sin_l, cos_l_m1;
  int64_t ix;
  uint32_t tix, hix, index;
  GET_LDOUBLE_MSW64 (ix, x);
  tix = ((uint64_t)ix) >> 32;
  tix &= ~0x80000000;			/* tix = |x|'s high 32 bits */
  if (tix < 0x3ffc3000)			/* |x| < 0.1484375 */
    {
      /* Argument is small enough to approximate it by a Chebyshev
	 polynomial of degree 17.  */
      if (tix < 0x3fc60000)		/* |x| < 2^-57 */
	{
	  math_check_force_underflow (x);
	  if (!((int)x)) return x;	/* generate inexact */
	}
      z = x * x;
      return x + (x * (z*(SIN1+z*(SIN2+z*(SIN3+z*(SIN4+
		       z*(SIN5+z*(SIN6+z*(SIN7+z*SIN8)))))))));
    }
  else
    {
      /* So that we don't have to use too large polynomial,  we find
	 l and h such that x = l + h,  where fabsl(l) <= 1.0/256 with 83
	 possible values for h.  We look up cosl(h) and sinl(h) in
	 pre-computed tables,  compute cosl(l) and sinl(l) using a
	 Chebyshev polynomial of degree 10(11) and compute
	 sinl(h+l) = sinl(h)cosl(l) + cosl(h)sinl(l).  */
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

      SET_LDOUBLE_WORDS64(h, ((uint64_t)hix) << 32, 0);
      if (iy)
	l = (ix < 0 ? -y : y) - (h - x);
      else
	l = x - h;
      z = l * l;
      sin_l = l*(ONE+z*(SSIN1+z*(SSIN2+z*(SSIN3+z*(SSIN4+z*SSIN5)))));
      cos_l_m1 = z*(SCOS1+z*(SCOS2+z*(SCOS3+z*(SCOS4+z*SCOS5))));
      z = __sincosl_table [index + SINCOSL_SIN_HI]
	  + (__sincosl_table [index + SINCOSL_SIN_LO]
	     + (__sincosl_table [index + SINCOSL_SIN_HI] * cos_l_m1)
	     + (__sincosl_table [index + SINCOSL_COS_HI] * sin_l));
      return (ix < 0) ? -z : z;
    }
}
