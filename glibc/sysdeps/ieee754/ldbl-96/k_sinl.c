/* Quad-precision floating point sine on <-pi/4,pi/4>.
   Copyright (C) 1999-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Based on quad-precision sine by Jakub Jelinek <jj@ultra.linux.cz>

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

/* The polynomials have not been optimized for extended-precision and
   may contain more terms than needed.  */

#include <float.h>
#include <math.h>
#include <math_private.h>
#include <math-underflow.h>

/* The polynomials have not been optimized for extended-precision and
   may contain more terms than needed.  */

static const long double c[] = {
#define ONE c[0]
 1.00000000000000000000000000000000000E+00L,

/* cos x ~ ONE + x^2 ( SCOS1 + SCOS2 * x^2 + ... + SCOS4 * x^6 + SCOS5 * x^8 )
   x in <0,1/256>  */
#define SCOS1 c[1]
#define SCOS2 c[2]
#define SCOS3 c[3]
#define SCOS4 c[4]
#define SCOS5 c[5]
-5.00000000000000000000000000000000000E-01L,
 4.16666666666666666666666666556146073E-02L,
-1.38888888888888888888309442601939728E-03L,
 2.48015873015862382987049502531095061E-05L,
-2.75573112601362126593516899592158083E-07L,

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
-1.66666666666666666666666666666666538e-01L,
 8.33333333333333333333333333307532934e-03L,
-1.98412698412698412698412534478712057e-04L,
 2.75573192239858906520896496653095890e-06L,
-2.50521083854417116999224301266655662e-08L,
 1.60590438367608957516841576404938118e-10L,
-7.64716343504264506714019494041582610e-13L,
 2.81068754939739570236322404393398135e-15L,

/* sin x ~ ONE * x + x^3 ( SSIN1 + SSIN2 * x^2 + ... + SSIN4 * x^6 + SSIN5 * x^8 )
   x in <0,1/256>  */
#define SSIN1 c[14]
#define SSIN2 c[15]
#define SSIN3 c[16]
#define SSIN4 c[17]
#define SSIN5 c[18]
-1.66666666666666666666666666666666659E-01L,
 8.33333333333333333333333333146298442E-03L,
-1.98412698412698412697726277416810661E-04L,
 2.75573192239848624174178393552189149E-06L,
-2.50521016467996193495359189395805639E-08L,
};

#define SINCOSL_COS_HI 0
#define SINCOSL_COS_LO 1
#define SINCOSL_SIN_HI 2
#define SINCOSL_SIN_LO 3
extern const long double __sincosl_table[];

long double
__kernel_sinl(long double x, long double y, int iy)
{
  long double absx, h, l, z, sin_l, cos_l_m1;
  int index;

  absx = fabsl (x);
  if (absx < 0.1484375L)
    {
      /* Argument is small enough to approximate it by a Chebyshev
	 polynomial of degree 17.  */
      if (absx < 0x1p-33L)
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
      index = (int) (128 * (absx - (0.1484375L - 1.0L / 256.0L)));
      h = 0.1484375L + index / 128.0;
      index *= 4;
      if (iy)
	l = (x < 0 ? -y : y) - (h - absx);
      else
	l = absx - h;
      z = l * l;
      sin_l = l*(ONE+z*(SSIN1+z*(SSIN2+z*(SSIN3+z*(SSIN4+z*SSIN5)))));
      cos_l_m1 = z*(SCOS1+z*(SCOS2+z*(SCOS3+z*(SCOS4+z*SCOS5))));
      z = __sincosl_table [index + SINCOSL_SIN_HI]
	  + (__sincosl_table [index + SINCOSL_SIN_LO]
	     + (__sincosl_table [index + SINCOSL_SIN_HI] * cos_l_m1)
	     + (__sincosl_table [index + SINCOSL_COS_HI] * sin_l));
      return (x < 0) ? -z : z;
    }
}
