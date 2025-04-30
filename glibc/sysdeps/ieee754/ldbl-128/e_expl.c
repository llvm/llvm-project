/* Quad-precision floating point e^x.
   Copyright (C) 1999-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Jakub Jelinek <jj@ultra.linux.cz>
   Partly based on double-precision code
   by Geoffrey Keating <geoffk@ozemail.com.au>

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

/* The basic design here is from
   Abraham Ziv, "Fast Evaluation of Elementary Mathematical Functions with
   Correctly Rounded Last Bit", ACM Trans. Math. Soft., 17 (3), September 1991,
   pp. 410-423.

   We work with number pairs where the first number is the high part and
   the second one is the low part. Arithmetic with the high part numbers must
   be exact, without any roundoff errors.

   The input value, X, is written as
   X = n * ln(2)_0 + arg1[t1]_0 + arg2[t2]_0 + x
       - n * ln(2)_1 + arg1[t1]_1 + arg2[t2]_1 + xl

   where:
   - n is an integer, 16384 >= n >= -16495;
   - ln(2)_0 is the first 93 bits of ln(2), and |ln(2)_0-ln(2)-ln(2)_1| < 2^-205
   - t1 is an integer, 89 >= t1 >= -89
   - t2 is an integer, 65 >= t2 >= -65
   - |arg1[t1]-t1/256.0| < 2^-53
   - |arg2[t2]-t2/32768.0| < 2^-53
   - x + xl is whatever is left, |x + xl| < 2^-16 + 2^-53

   Then e^x is approximated as

   e^x = 2^n_1 ( 2^n_0 e^(arg1[t1]_0 + arg1[t1]_1) e^(arg2[t2]_0 + arg2[t2]_1)
	       + 2^n_0 e^(arg1[t1]_0 + arg1[t1]_1) e^(arg2[t2]_0 + arg2[t2]_1)
		 * p (x + xl + n * ln(2)_1))
   where:
   - p(x) is a polynomial approximating e(x)-1
   - e^(arg1[t1]_0 + arg1[t1]_1) is obtained from a table
   - e^(arg2[t2]_0 + arg2[t2]_1) likewise
   - n_1 + n_0 = n, so that |n_0| < -LDBL_MIN_EXP-1.

   If it happens that n_1 == 0 (this is the usual case), that multiplication
   is omitted.
   */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <float.h>
#include <ieee754.h>
#include <math.h>
#include <fenv.h>
#include <inttypes.h>
#include <math-barriers.h>
#include <math_private.h>
#include <math-underflow.h>
#include <stdlib.h>
#include "t_expl.h"
#include <libm-alias-finite.h>

static const _Float128 C[] = {
/* Smallest integer x for which e^x overflows.  */
#define himark C[0]
 L(11356.523406294143949491931077970765),

/* Largest integer x for which e^x underflows.  */
#define lomark C[1]
L(-11433.4627433362978788372438434526231),

/* 3x2^96 */
#define THREEp96 C[2]
 L(59421121885698253195157962752.0),

/* 3x2^103 */
#define THREEp103 C[3]
 L(30423614405477505635920876929024.0),

/* 3x2^111 */
#define THREEp111 C[4]
 L(7788445287802241442795744493830144.0),

/* 1/ln(2) */
#define M_1_LN2 C[5]
 L(1.44269504088896340735992468100189204),

/* first 93 bits of ln(2) */
#define M_LN2_0 C[6]
 L(0.693147180559945309417232121457981864),

/* ln2_0 - ln(2) */
#define M_LN2_1 C[7]
L(-1.94704509238074995158795957333327386E-31),

/* very small number */
#define TINY C[8]
 L(1.0e-4900),

/* 2^16383 */
#define TWO16383 C[9]
 L(5.94865747678615882542879663314003565E+4931),

/* 256 */
#define TWO8 C[10]
 256,

/* 32768 */
#define TWO15 C[11]
 32768,

/* Chebyshev polynom coefficients for (exp(x)-1)/x */
#define P1 C[12]
#define P2 C[13]
#define P3 C[14]
#define P4 C[15]
#define P5 C[16]
#define P6 C[17]
 L(0.5),
 L(1.66666666666666666666666666666666683E-01),
 L(4.16666666666666666666654902320001674E-02),
 L(8.33333333333333333333314659767198461E-03),
 L(1.38888888889899438565058018857254025E-03),
 L(1.98412698413981650382436541785404286E-04),
};

_Float128
__ieee754_expl (_Float128 x)
{
  /* Check for usual case.  */
  if (isless (x, himark) && isgreater (x, lomark))
    {
      int tval1, tval2, unsafe, n_i;
      _Float128 x22, n, t, result, xl;
      union ieee854_long_double ex2_u, scale_u;
      fenv_t oldenv;

      feholdexcept (&oldenv);
#ifdef FE_TONEAREST
      fesetround (FE_TONEAREST);
#endif

      /* Calculate n.  */
      n = x * M_1_LN2 + THREEp111;
      n -= THREEp111;
      x = x - n * M_LN2_0;
      xl = n * M_LN2_1;

      /* Calculate t/256.  */
      t = x + THREEp103;
      t -= THREEp103;

      /* Compute tval1 = t.  */
      tval1 = (int) (t * TWO8);

      x -= __expl_table[T_EXPL_ARG1+2*tval1];
      xl -= __expl_table[T_EXPL_ARG1+2*tval1+1];

      /* Calculate t/32768.  */
      t = x + THREEp96;
      t -= THREEp96;

      /* Compute tval2 = t.  */
      tval2 = (int) (t * TWO15);

      x -= __expl_table[T_EXPL_ARG2+2*tval2];
      xl -= __expl_table[T_EXPL_ARG2+2*tval2+1];

      x = x + xl;

      /* Compute ex2 = 2^n_0 e^(argtable[tval1]) e^(argtable[tval2]).  */
      ex2_u.d = __expl_table[T_EXPL_RES1 + tval1]
		* __expl_table[T_EXPL_RES2 + tval2];
      n_i = (int)n;
      /* 'unsafe' is 1 iff n_1 != 0.  */
      unsafe = abs(n_i) >= 15000;
      ex2_u.ieee.exponent += n_i >> unsafe;

      /* Compute scale = 2^n_1.  */
      scale_u.d = 1;
      scale_u.ieee.exponent += n_i - (n_i >> unsafe);

      /* Approximate e^x2 - 1, using a seventh-degree polynomial,
	 with maximum error in [-2^-16-2^-53,2^-16+2^-53]
	 less than 4.8e-39.  */
      x22 = x + x*x*(P1+x*(P2+x*(P3+x*(P4+x*(P5+x*P6)))));
      math_force_eval (x22);

      /* Return result.  */
      fesetenv (&oldenv);

      result = x22 * ex2_u.d + ex2_u.d;

      /* Now we can test whether the result is ultimate or if we are unsure.
	 In the later case we should probably call a mpn based routine to give
	 the ultimate result.
	 Empirically, this routine is already ultimate in about 99.9986% of
	 cases, the test below for the round to nearest case will be false
	 in ~ 99.9963% of cases.
	 Without proc2 routine maximum error which has been seen is
	 0.5000262 ulp.

	  union ieee854_long_double ex3_u;

	  #ifdef FE_TONEAREST
	    fesetround (FE_TONEAREST);
	  #endif
	  ex3_u.d = (result - ex2_u.d) - x22 * ex2_u.d;
	  ex2_u.d = result;
	  ex3_u.ieee.exponent += LDBL_MANT_DIG + 15 + IEEE854_LONG_DOUBLE_BIAS
				 - ex2_u.ieee.exponent;
	  n_i = abs (ex3_u.d);
	  n_i = (n_i + 1) / 2;
	  fesetenv (&oldenv);
	  #ifdef FE_TONEAREST
	  if (fegetround () == FE_TONEAREST)
	    n_i -= 0x4000;
	  #endif
	  if (!n_i) {
	    return __ieee754_expl_proc2 (origx);
	  }
       */
      if (!unsafe)
	return result;
      else
	{
	  result *= scale_u.d;
	  math_check_force_underflow_nonneg (result);
	  return result;
	}
    }
  /* Exceptional cases:  */
  else if (isless (x, himark))
    {
      if (isinf (x))
	/* e^-inf == 0, with no error.  */
	return 0;
      else
	/* Underflow */
	return TINY * TINY;
    }
  else
    /* Return x, if x is a NaN or Inf; or overflow, otherwise.  */
    return TWO16383*x;
}
libm_alias_finite (__ieee754_expl, __expl)
