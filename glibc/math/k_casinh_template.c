/* Return arc hyperbolic sine for a complex float type, with the
   imaginary part of the result possibly adjusted for use in
   computing other functions.
   Copyright (C) 1997-2021 Free Software Foundation, Inc.
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

#include <complex.h>
#include <math.h>
#include <math_private.h>
#include <math-underflow.h>
#include <float.h>

/* Return the complex inverse hyperbolic sine of finite nonzero Z,
   with the imaginary part of the result subtracted from pi/2 if ADJ
   is nonzero.  */

CFLOAT
M_DECL_FUNC (__kernel_casinh) (CFLOAT x, int adj)
{
  CFLOAT res;
  FLOAT rx, ix;
  CFLOAT y;

  /* Avoid cancellation by reducing to the first quadrant.  */
  rx = M_FABS (__real__ x);
  ix = M_FABS (__imag__ x);

  if (rx >= 1 / M_EPSILON || ix >= 1 / M_EPSILON)
    {
      /* For large x in the first quadrant, x + csqrt (1 + x * x)
	 is sufficiently close to 2 * x to make no significant
	 difference to the result; avoid possible overflow from
	 the squaring and addition.  */
      __real__ y = rx;
      __imag__ y = ix;

      if (adj)
	{
	  FLOAT t = __real__ y;
	  __real__ y = M_COPYSIGN (__imag__ y, __imag__ x);
	  __imag__ y = t;
	}

      res = M_SUF (__clog) (y);
      __real__ res += (FLOAT) M_MLIT (M_LN2);
    }
  else if (rx >= M_LIT (0.5) && ix < M_EPSILON / 8)
    {
      FLOAT s = M_HYPOT (1, rx);

      __real__ res = M_LOG (rx + s);
      if (adj)
	__imag__ res = M_ATAN2 (s, __imag__ x);
      else
	__imag__ res = M_ATAN2 (ix, s);
    }
  else if (rx < M_EPSILON / 8 && ix >= M_LIT (1.5))
    {
      FLOAT s = M_SQRT ((ix + 1) * (ix - 1));

      __real__ res = M_LOG (ix + s);
      if (adj)
	__imag__ res = M_ATAN2 (rx, M_COPYSIGN (s, __imag__ x));
      else
	__imag__ res = M_ATAN2 (s, rx);
    }
  else if (ix > 1 && ix < M_LIT (1.5) && rx < M_LIT (0.5))
    {
      if (rx < M_EPSILON * M_EPSILON)
	{
	  FLOAT ix2m1 = (ix + 1) * (ix - 1);
	  FLOAT s = M_SQRT (ix2m1);

	  __real__ res = M_LOG1P (2 * (ix2m1 + ix * s)) / 2;
	  if (adj)
	    __imag__ res = M_ATAN2 (rx, M_COPYSIGN (s, __imag__ x));
	  else
	    __imag__ res = M_ATAN2 (s, rx);
	}
      else
	{
	  FLOAT ix2m1 = (ix + 1) * (ix - 1);
	  FLOAT rx2 = rx * rx;
	  FLOAT f = rx2 * (2 + rx2 + 2 * ix * ix);
	  FLOAT d = M_SQRT (ix2m1 * ix2m1 + f);
	  FLOAT dp = d + ix2m1;
	  FLOAT dm = f / dp;
	  FLOAT r1 = M_SQRT ((dm + rx2) / 2);
	  FLOAT r2 = rx * ix / r1;

	  __real__ res = M_LOG1P (rx2 + dp + 2 * (rx * r1 + ix * r2)) / 2;
	  if (adj)
	    __imag__ res = M_ATAN2 (rx + r1, M_COPYSIGN (ix + r2, __imag__ x));
	  else
	    __imag__ res = M_ATAN2 (ix + r2, rx + r1);
	}
    }
  else if (ix == 1 && rx < M_LIT (0.5))
    {
      if (rx < M_EPSILON / 8)
	{
	  __real__ res = M_LOG1P (2 * (rx + M_SQRT (rx))) / 2;
	  if (adj)
	    __imag__ res = M_ATAN2 (M_SQRT (rx), M_COPYSIGN (1, __imag__ x));
	  else
	    __imag__ res = M_ATAN2 (1, M_SQRT (rx));
	}
      else
	{
	  FLOAT d = rx * M_SQRT (4 + rx * rx);
	  FLOAT s1 = M_SQRT ((d + rx * rx) / 2);
	  FLOAT s2 = M_SQRT ((d - rx * rx) / 2);

	  __real__ res = M_LOG1P (rx * rx + d + 2 * (rx * s1 + s2)) / 2;
	  if (adj)
	    __imag__ res = M_ATAN2 (rx + s1, M_COPYSIGN (1 + s2, __imag__ x));
	  else
	    __imag__ res = M_ATAN2 (1 + s2, rx + s1);
	}
    }
  else if (ix < 1 && rx < M_LIT (0.5))
    {
      if (ix >= M_EPSILON)
	{
	  if (rx < M_EPSILON * M_EPSILON)
	    {
	      FLOAT onemix2 = (1 + ix) * (1 - ix);
	      FLOAT s = M_SQRT (onemix2);

	      __real__ res = M_LOG1P (2 * rx / s) / 2;
	      if (adj)
		__imag__ res = M_ATAN2 (s, __imag__ x);
	      else
		__imag__ res = M_ATAN2 (ix, s);
	    }
	  else
	    {
	      FLOAT onemix2 = (1 + ix) * (1 - ix);
	      FLOAT rx2 = rx * rx;
	      FLOAT f = rx2 * (2 + rx2 + 2 * ix * ix);
	      FLOAT d = M_SQRT (onemix2 * onemix2 + f);
	      FLOAT dp = d + onemix2;
	      FLOAT dm = f / dp;
	      FLOAT r1 = M_SQRT ((dp + rx2) / 2);
	      FLOAT r2 = rx * ix / r1;

	      __real__ res = M_LOG1P (rx2 + dm + 2 * (rx * r1 + ix * r2)) / 2;
	      if (adj)
		__imag__ res = M_ATAN2 (rx + r1, M_COPYSIGN (ix + r2,
							     __imag__ x));
	      else
		__imag__ res = M_ATAN2 (ix + r2, rx + r1);
	    }
	}
      else
	{
	  FLOAT s = M_HYPOT (1, rx);

	  __real__ res = M_LOG1P (2 * rx * (rx + s)) / 2;
	  if (adj)
	    __imag__ res = M_ATAN2 (s, __imag__ x);
	  else
	    __imag__ res = M_ATAN2 (ix, s);
	}
      math_check_force_underflow_nonneg (__real__ res);
    }
  else
    {
      __real__ y = (rx - ix) * (rx + ix) + 1;
      __imag__ y = 2 * rx * ix;

      y = M_SUF (__csqrt) (y);

      __real__ y += rx;
      __imag__ y += ix;

      if (adj)
	{
	  FLOAT t = __real__ y;
	  __real__ y = M_COPYSIGN (__imag__ y, __imag__ x);
	  __imag__ y = t;
	}

      res = M_SUF (__clog) (y);
    }

  /* Give results the correct sign for the original argument.  */
  __real__ res = M_COPYSIGN (__real__ res, __real__ x);
  __imag__ res = M_COPYSIGN (__imag__ res, (adj ? 1 : __imag__ x));

  return res;
}
