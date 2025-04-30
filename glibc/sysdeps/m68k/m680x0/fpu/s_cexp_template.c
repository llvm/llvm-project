/* Complex exponential function.  m68k fpu version
   Copyright (C) 1997-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Andreas Schwab <schwab@issan.informatik.uni-dortmund.de>

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library.  If not, see
   <https://www.gnu.org/licenses/>.  */

#include <float.h>
#include <complex.h>
#include <math.h>
#include "mathimpl.h"

#define CONCATX(a,b) __CONCAT(a,b)
#define s(name) M_SUF (name)
#define m81(func) __m81_u(s(func))

CFLOAT
s(__cexp) (CFLOAT x)
{
  CFLOAT retval;
  unsigned long ix_cond;

  ix_cond = __m81_test (__imag__ x);

  if ((ix_cond & (__M81_COND_NAN|__M81_COND_INF)) == 0)
    {
      /* Imaginary part is finite.  */
      unsigned long rx_cond = __m81_test (__real__ x);

      if ((rx_cond & (__M81_COND_NAN|__M81_COND_INF)) == 0)
	{
	  const int t = (int) ((LDBL_MAX_EXP - 1) * M_LN2l);
	  long double sin_ix, cos_ix, exp_val;

	  __m81_u (__sincosl) (__imag__ x, &sin_ix, &cos_ix);

	  if (__real__ x > t)
	    {
	      long double exp_t = __m81_u(__ieee754_expl) (t);
	      __real__ x -= t;
	      sin_ix *= exp_t;
	      cos_ix *= exp_t;
	      if (__real__ x > t)
		{
		  __real__ x -= t;
		  sin_ix *= exp_t;
		  cos_ix *= exp_t;
		}
	    }

	  exp_val = __m81_u(__ieee754_expl) (__real__ x);
	  __real__ retval = exp_val * cos_ix;
	  if (ix_cond & __M81_COND_ZERO)
	    __imag__ retval = __imag__ x;
	  else
	    __imag__ retval = exp_val * sin_ix;
	}
      else
	{
	  /* Compute the sign of the result.  */
	  long double remainder, pi_2;
	  int quadrant;

	  if ((rx_cond & (__M81_COND_NAN|__M81_COND_NEG)) == __M81_COND_NEG)
	    __real__ retval = __imag__ retval = 0.0;
	  else
	    __real__ retval = __imag__ retval = __real__ x;
	  __asm ("fmovecr %#0,%0\n\tfscale%.w %#-1,%0" : "=f" (pi_2));
	  __asm ("fmod%.x %2,%0\n\tfmove%.l %/fpsr,%1"
		 : "=f" (remainder), "=dm" (quadrant)
		 : "f" (pi_2), "0" (__imag__ x));
	  quadrant = (quadrant >> 16) & 0x83;
	  if (quadrant & 0x80)
	    quadrant ^= 0x83;
	  switch (quadrant)
	    {
	    default:
	      break;
	    case 1:
	      __real__ retval = -__real__ retval;
	      break;
	    case 2:
	      __real__ retval = -__real__ retval;
	      /* Fall through.  */
	    case 3:
	      __imag__ retval = -__imag__ retval;
	      break;
	    }
	  if (ix_cond & __M81_COND_ZERO && (rx_cond & __M81_COND_NAN) == 0)
	    __imag__ retval = __imag__ x;
	}
    }
  else
    {
      unsigned long rx_cond = __m81_test (__real__ x);

      if (rx_cond & __M81_COND_INF)
	{
	  /* Real part is infinite.  */
	  if (rx_cond & __M81_COND_NEG)
	    {
	      __real__ retval = __imag__ retval = 0.0;
	      if (ix_cond & __M81_COND_NEG)
		__imag__ retval = -__imag__ retval;
	    }
	  else
	    {
	      __real__ retval = __real__ x;
	      __imag__ retval = __imag__ x - __imag__ x;
	    }
	}
      else
	__real__ retval = __imag__ retval = __imag__ x - __imag__ x;
    }

  return retval;
}
declare_mgen_alias (__cexp, cexp)
