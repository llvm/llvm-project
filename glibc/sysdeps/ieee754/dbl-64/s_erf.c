/* @(#)s_erf.c 5.1 93/09/24 */
/*
 * ====================================================
 * Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
 *
 * Developed at SunPro, a Sun Microsystems, Inc. business.
 * Permission to use, copy, modify, and distribute this
 * software is freely granted, provided that this notice
 * is preserved.
 * ====================================================
 */
/* Modified by Naohiko Shimizu/Tokai University, Japan 1997/08/25,
   for performance improvement on pipelined processors.
*/

#if defined(LIBM_SCCS) && !defined(lint)
static char rcsid[] = "$NetBSD: s_erf.c,v 1.8 1995/05/10 20:47:05 jtc Exp $";
#endif

/* double erf(double x)
 * double erfc(double x)
 *			     x
 *		      2      |\
 *     erf(x)  =  ---------  | exp(-t*t)dt
 *	 	   sqrt(pi) \|
 *			     0
 *
 *     erfc(x) =  1-erf(x)
 *  Note that
 *		erf(-x) = -erf(x)
 *		erfc(-x) = 2 - erfc(x)
 *
 * Method:
 *	1. For |x| in [0, 0.84375]
 *	    erf(x)  = x + x*R(x^2)
 *          erfc(x) = 1 - erf(x)           if x in [-.84375,0.25]
 *                  = 0.5 + ((0.5-x)-x*R)  if x in [0.25,0.84375]
 *	   where R = P/Q where P is an odd poly of degree 8 and
 *	   Q is an odd poly of degree 10.
 *						 -57.90
 *			| R - (erf(x)-x)/x | <= 2
 *
 *
 *	   Remark. The formula is derived by noting
 *          erf(x) = (2/sqrt(pi))*(x - x^3/3 + x^5/10 - x^7/42 + ....)
 *	   and that
 *          2/sqrt(pi) = 1.128379167095512573896158903121545171688
 *	   is close to one. The interval is chosen because the fix
 *	   point of erf(x) is near 0.6174 (i.e., erf(x)=x when x is
 *	   near 0.6174), and by some experiment, 0.84375 is chosen to
 * 	   guarantee the error is less than one ulp for erf.
 *
 *      2. For |x| in [0.84375,1.25], let s = |x| - 1, and
 *         c = 0.84506291151 rounded to single (24 bits)
 *         	erf(x)  = sign(x) * (c  + P1(s)/Q1(s))
 *         	erfc(x) = (1-c)  - P1(s)/Q1(s) if x > 0
 *			  1+(c+P1(s)/Q1(s))    if x < 0
 *         	|P1/Q1 - (erf(|x|)-c)| <= 2**-59.06
 *	   Remark: here we use the taylor series expansion at x=1.
 *		erf(1+s) = erf(1) + s*Poly(s)
 *			 = 0.845.. + P1(s)/Q1(s)
 *	   That is, we use rational approximation to approximate
 *			erf(1+s) - (c = (single)0.84506291151)
 *	   Note that |P1/Q1|< 0.078 for x in [0.84375,1.25]
 *	   where
 *		P1(s) = degree 6 poly in s
 *		Q1(s) = degree 6 poly in s
 *
 *      3. For x in [1.25,1/0.35(~2.857143)],
 *         	erfc(x) = (1/x)*exp(-x*x-0.5625+R1/S1)
 *         	erf(x)  = 1 - erfc(x)
 *	   where
 *		R1(z) = degree 7 poly in z, (z=1/x^2)
 *		S1(z) = degree 8 poly in z
 *
 *      4. For x in [1/0.35,28]
 *         	erfc(x) = (1/x)*exp(-x*x-0.5625+R2/S2) if x > 0
 *			= 2.0 - (1/x)*exp(-x*x-0.5625+R2/S2) if -6<x<0
 *			= 2.0 - tiny		(if x <= -6)
 *         	erf(x)  = sign(x)*(1.0 - erfc(x)) if x < 6, else
 *         	erf(x)  = sign(x)*(1.0 - tiny)
 *	   where
 *		R2(z) = degree 6 poly in z, (z=1/x^2)
 *		S2(z) = degree 7 poly in z
 *
 *      Note1:
 *	   To compute exp(-x*x-0.5625+R/S), let s be a single
 *	   precision number and s := x; then
 *		-x*x = -s*s + (s-x)*(s+x)
 *	        exp(-x*x-0.5626+R/S) =
 *			exp(-s*s-0.5625)*exp((s-x)*(s+x)+R/S);
 *      Note2:
 *	   Here 4 and 5 make use of the asymptotic series
 *			  exp(-x*x)
 *		erfc(x) ~ ---------- * ( 1 + Poly(1/x^2) )
 *			  x*sqrt(pi)
 *	   We use rational approximation to approximate
 *      	g(s)=f(1/x^2) = log(erfc(x)*x) - x*x + 0.5625
 *	   Here is the error bound for R1/S1 and R2/S2
 *      	|R1/S1 - f(x)|  < 2**(-62.57)
 *      	|R2/S2 - f(x)|  < 2**(-61.52)
 *
 *      5. For inf > x >= 28
 *         	erf(x)  = sign(x) *(1 - tiny)  (raise inexact)
 *         	erfc(x) = tiny*tiny (raise underflow) if x > 0
 *			= 2 - tiny if x<0
 *
 *      7. Special case:
 *         	erf(0)  = 0, erf(inf)  = 1, erf(-inf) = -1,
 *         	erfc(0) = 1, erfc(inf) = 0, erfc(-inf) = 2,
 *	   	erfc/erf(NaN) is NaN
 */


#include <errno.h>
#include <float.h>
#include <math.h>
#include <math-narrow-eval.h>
#include <math_private.h>
#include <math-underflow.h>
#include <libm-alias-double.h>
#include <fix-int-fp-convert-zero.h>

static const double
  tiny = 1e-300,
  half = 5.00000000000000000000e-01, /* 0x3FE00000, 0x00000000 */
  one = 1.00000000000000000000e+00, /* 0x3FF00000, 0x00000000 */
  two = 2.00000000000000000000e+00, /* 0x40000000, 0x00000000 */
/* c = (float)0.84506291151 */
  erx = 8.45062911510467529297e-01, /* 0x3FEB0AC1, 0x60000000 */
/*
 * Coefficients for approximation to  erf on [0,0.84375]
 */
  efx = 1.28379167095512586316e-01, /* 0x3FC06EBA, 0x8214DB69 */
  pp[] = { 1.28379167095512558561e-01, /* 0x3FC06EBA, 0x8214DB68 */
	   -3.25042107247001499370e-01, /* 0xBFD4CD7D, 0x691CB913 */
	   -2.84817495755985104766e-02, /* 0xBF9D2A51, 0xDBD7194F */
	   -5.77027029648944159157e-03, /* 0xBF77A291, 0x236668E4 */
	   -2.37630166566501626084e-05 }, /* 0xBEF8EAD6, 0x120016AC */
  qq[] = { 0.0, 3.97917223959155352819e-01, /* 0x3FD97779, 0xCDDADC09 */
	   6.50222499887672944485e-02, /* 0x3FB0A54C, 0x5536CEBA */
	   5.08130628187576562776e-03, /* 0x3F74D022, 0xC4D36B0F */
	   1.32494738004321644526e-04, /* 0x3F215DC9, 0x221C1A10 */
	   -3.96022827877536812320e-06 }, /* 0xBED09C43, 0x42A26120 */
/*
 * Coefficients for approximation to  erf  in [0.84375,1.25]
 */
  pa[] = { -2.36211856075265944077e-03, /* 0xBF6359B8, 0xBEF77538 */
	   4.14856118683748331666e-01, /* 0x3FDA8D00, 0xAD92B34D */
	   -3.72207876035701323847e-01, /* 0xBFD7D240, 0xFBB8C3F1 */
	   3.18346619901161753674e-01, /* 0x3FD45FCA, 0x805120E4 */
	   -1.10894694282396677476e-01, /* 0xBFBC6398, 0x3D3E28EC */
	   3.54783043256182359371e-02, /* 0x3FA22A36, 0x599795EB */
	   -2.16637559486879084300e-03 }, /* 0xBF61BF38, 0x0A96073F */
  qa[] = { 0.0, 1.06420880400844228286e-01, /* 0x3FBB3E66, 0x18EEE323 */
	   5.40397917702171048937e-01, /* 0x3FE14AF0, 0x92EB6F33 */
	   7.18286544141962662868e-02, /* 0x3FB2635C, 0xD99FE9A7 */
	   1.26171219808761642112e-01, /* 0x3FC02660, 0xE763351F */
	   1.36370839120290507362e-02, /* 0x3F8BEDC2, 0x6B51DD1C */
	   1.19844998467991074170e-02 }, /* 0x3F888B54, 0x5735151D */
/*
 * Coefficients for approximation to  erfc in [1.25,1/0.35]
 */
  ra[] = { -9.86494403484714822705e-03, /* 0xBF843412, 0x600D6435 */
	   -6.93858572707181764372e-01, /* 0xBFE63416, 0xE4BA7360 */
	   -1.05586262253232909814e+01, /* 0xC0251E04, 0x41B0E726 */
	   -6.23753324503260060396e+01, /* 0xC04F300A, 0xE4CBA38D */
	   -1.62396669462573470355e+02, /* 0xC0644CB1, 0x84282266 */
	   -1.84605092906711035994e+02, /* 0xC067135C, 0xEBCCABB2 */
	   -8.12874355063065934246e+01, /* 0xC0545265, 0x57E4D2F2 */
	   -9.81432934416914548592e+00 }, /* 0xC023A0EF, 0xC69AC25C */
  sa[] = { 0.0, 1.96512716674392571292e+01, /* 0x4033A6B9, 0xBD707687 */
	   1.37657754143519042600e+02, /* 0x4061350C, 0x526AE721 */
	   4.34565877475229228821e+02, /* 0x407B290D, 0xD58A1A71 */
	   6.45387271733267880336e+02, /* 0x40842B19, 0x21EC2868 */
	   4.29008140027567833386e+02, /* 0x407AD021, 0x57700314 */
	   1.08635005541779435134e+02, /* 0x405B28A3, 0xEE48AE2C */
	   6.57024977031928170135e+00, /* 0x401A47EF, 0x8E484A93 */
	   -6.04244152148580987438e-02 }, /* 0xBFAEEFF2, 0xEE749A62 */
/*
 * Coefficients for approximation to  erfc in [1/.35,28]
 */
  rb[] = { -9.86494292470009928597e-03, /* 0xBF843412, 0x39E86F4A */
	   -7.99283237680523006574e-01, /* 0xBFE993BA, 0x70C285DE */
	   -1.77579549177547519889e+01, /* 0xC031C209, 0x555F995A */
	   -1.60636384855821916062e+02, /* 0xC064145D, 0x43C5ED98 */
	   -6.37566443368389627722e+02, /* 0xC083EC88, 0x1375F228 */
	   -1.02509513161107724954e+03, /* 0xC0900461, 0x6A2E5992 */
	   -4.83519191608651397019e+02 }, /* 0xC07E384E, 0x9BDC383F */
  sb[] = { 0.0, 3.03380607434824582924e+01, /* 0x403E568B, 0x261D5190 */
	   3.25792512996573918826e+02, /* 0x40745CAE, 0x221B9F0A */
	   1.53672958608443695994e+03, /* 0x409802EB, 0x189D5118 */
	   3.19985821950859553908e+03, /* 0x40A8FFB7, 0x688C246A */
	   2.55305040643316442583e+03, /* 0x40A3F219, 0xCEDF3BE6 */
	   4.74528541206955367215e+02, /* 0x407DA874, 0xE79FE763 */
	   -2.24409524465858183362e+01 }; /* 0xC03670E2, 0x42712D62 */

double
__erf (double x)
{
  int32_t hx, ix, i;
  double R, S, P, Q, s, y, z, r;
  GET_HIGH_WORD (hx, x);
  ix = hx & 0x7fffffff;
  if (ix >= 0x7ff00000)                 /* erf(nan)=nan */
    {
      i = ((uint32_t) hx >> 31) << 1;
      return (double) (1 - i) + one / x; /* erf(+-inf)=+-1 */
    }

  if (ix < 0x3feb0000)                  /* |x|<0.84375 */
    {
      double r1, r2, s1, s2, s3, z2, z4;
      if (ix < 0x3e300000)              /* |x|<2**-28 */
	{
	  if (ix < 0x00800000)
	    {
	      /* Avoid spurious underflow.  */
	      double ret = 0.0625 * (16.0 * x + (16.0 * efx) * x);
	      math_check_force_underflow (ret);
	      return ret;
	    }
	  return x + efx * x;
	}
      z = x * x;
      r1 = pp[0] + z * pp[1]; z2 = z * z;
      r2 = pp[2] + z * pp[3]; z4 = z2 * z2;
      s1 = one + z * qq[1];
      s2 = qq[2] + z * qq[3];
      s3 = qq[4] + z * qq[5];
      r = r1 + z2 * r2 + z4 * pp[4];
      s = s1 + z2 * s2 + z4 * s3;
      y = r / s;
      return x + x * y;
    }
  if (ix < 0x3ff40000)                  /* 0.84375 <= |x| < 1.25 */
    {
      double s2, s4, s6, P1, P2, P3, P4, Q1, Q2, Q3, Q4;
      s = fabs (x) - one;
      P1 = pa[0] + s * pa[1]; s2 = s * s;
      Q1 = one + s * qa[1];   s4 = s2 * s2;
      P2 = pa[2] + s * pa[3]; s6 = s4 * s2;
      Q2 = qa[2] + s * qa[3];
      P3 = pa[4] + s * pa[5];
      Q3 = qa[4] + s * qa[5];
      P4 = pa[6];
      Q4 = qa[6];
      P = P1 + s2 * P2 + s4 * P3 + s6 * P4;
      Q = Q1 + s2 * Q2 + s4 * Q3 + s6 * Q4;
      if (hx >= 0)
	return erx + P / Q;
      else
	return -erx - P / Q;
    }
  if (ix >= 0x40180000)                 /* inf>|x|>=6 */
    {
      if (hx >= 0)
	return one - tiny;
      else
	return tiny - one;
    }
  x = fabs (x);
  s = one / (x * x);
  if (ix < 0x4006DB6E)          /* |x| < 1/0.35 */
    {
      double R1, R2, R3, R4, S1, S2, S3, S4, s2, s4, s6, s8;
      R1 = ra[0] + s * ra[1]; s2 = s * s;
      S1 = one + s * sa[1];  s4 = s2 * s2;
      R2 = ra[2] + s * ra[3]; s6 = s4 * s2;
      S2 = sa[2] + s * sa[3]; s8 = s4 * s4;
      R3 = ra[4] + s * ra[5];
      S3 = sa[4] + s * sa[5];
      R4 = ra[6] + s * ra[7];
      S4 = sa[6] + s * sa[7];
      R = R1 + s2 * R2 + s4 * R3 + s6 * R4;
      S = S1 + s2 * S2 + s4 * S3 + s6 * S4 + s8 * sa[8];
    }
  else                  /* |x| >= 1/0.35 */
    {
      double R1, R2, R3, S1, S2, S3, S4, s2, s4, s6;
      R1 = rb[0] + s * rb[1]; s2 = s * s;
      S1 = one + s * sb[1];  s4 = s2 * s2;
      R2 = rb[2] + s * rb[3]; s6 = s4 * s2;
      S2 = sb[2] + s * sb[3];
      R3 = rb[4] + s * rb[5];
      S3 = sb[4] + s * sb[5];
      S4 = sb[6] + s * sb[7];
      R = R1 + s2 * R2 + s4 * R3 + s6 * rb[6];
      S = S1 + s2 * S2 + s4 * S3 + s6 * S4;
    }
  z = x;
  SET_LOW_WORD (z, 0);
  r = __ieee754_exp (-z * z - 0.5625) *
      __ieee754_exp ((z - x) * (z + x) + R / S);
  if (hx >= 0)
    return one - r / x;
  else
    return r / x - one;
}
libm_alias_double (__erf, erf)

double
__erfc (double x)
{
  int32_t hx, ix;
  double R, S, P, Q, s, y, z, r;
  GET_HIGH_WORD (hx, x);
  ix = hx & 0x7fffffff;
  if (ix >= 0x7ff00000)                         /* erfc(nan)=nan */
    {                                           /* erfc(+-inf)=0,2 */
      double ret = (double) (((uint32_t) hx >> 31) << 1) + one / x;
      if (FIX_INT_FP_CONVERT_ZERO && ret == 0.0)
	return 0.0;
      return ret;
    }

  if (ix < 0x3feb0000)                  /* |x|<0.84375 */
    {
      double r1, r2, s1, s2, s3, z2, z4;
      if (ix < 0x3c700000)              /* |x|<2**-56 */
	return one - x;
      z = x * x;
      r1 = pp[0] + z * pp[1]; z2 = z * z;
      r2 = pp[2] + z * pp[3]; z4 = z2 * z2;
      s1 = one + z * qq[1];
      s2 = qq[2] + z * qq[3];
      s3 = qq[4] + z * qq[5];
      r = r1 + z2 * r2 + z4 * pp[4];
      s = s1 + z2 * s2 + z4 * s3;
      y = r / s;
      if (hx < 0x3fd00000)              /* x<1/4 */
	{
	  return one - (x + x * y);
	}
      else
	{
	  r = x * y;
	  r += (x - half);
	  return half - r;
	}
    }
  if (ix < 0x3ff40000)                  /* 0.84375 <= |x| < 1.25 */
    {
      double s2, s4, s6, P1, P2, P3, P4, Q1, Q2, Q3, Q4;
      s = fabs (x) - one;
      P1 = pa[0] + s * pa[1]; s2 = s * s;
      Q1 = one + s * qa[1];   s4 = s2 * s2;
      P2 = pa[2] + s * pa[3]; s6 = s4 * s2;
      Q2 = qa[2] + s * qa[3];
      P3 = pa[4] + s * pa[5];
      Q3 = qa[4] + s * qa[5];
      P4 = pa[6];
      Q4 = qa[6];
      P = P1 + s2 * P2 + s4 * P3 + s6 * P4;
      Q = Q1 + s2 * Q2 + s4 * Q3 + s6 * Q4;
      if (hx >= 0)
	{
	  z = one - erx; return z - P / Q;
	}
      else
	{
	  z = erx + P / Q; return one + z;
	}
    }
  if (ix < 0x403c0000)                  /* |x|<28 */
    {
      x = fabs (x);
      s = one / (x * x);
      if (ix < 0x4006DB6D)              /* |x| < 1/.35 ~ 2.857143*/
	{
	  double R1, R2, R3, R4, S1, S2, S3, S4, s2, s4, s6, s8;
	  R1 = ra[0] + s * ra[1]; s2 = s * s;
	  S1 = one + s * sa[1];  s4 = s2 * s2;
	  R2 = ra[2] + s * ra[3]; s6 = s4 * s2;
	  S2 = sa[2] + s * sa[3]; s8 = s4 * s4;
	  R3 = ra[4] + s * ra[5];
	  S3 = sa[4] + s * sa[5];
	  R4 = ra[6] + s * ra[7];
	  S4 = sa[6] + s * sa[7];
	  R = R1 + s2 * R2 + s4 * R3 + s6 * R4;
	  S = S1 + s2 * S2 + s4 * S3 + s6 * S4 + s8 * sa[8];
	}
      else                              /* |x| >= 1/.35 ~ 2.857143 */
	{
	  double R1, R2, R3, S1, S2, S3, S4, s2, s4, s6;
	  if (hx < 0 && ix >= 0x40180000)
	    return two - tiny;                           /* x < -6 */
	  R1 = rb[0] + s * rb[1]; s2 = s * s;
	  S1 = one + s * sb[1];  s4 = s2 * s2;
	  R2 = rb[2] + s * rb[3]; s6 = s4 * s2;
	  S2 = sb[2] + s * sb[3];
	  R3 = rb[4] + s * rb[5];
	  S3 = sb[4] + s * sb[5];
	  S4 = sb[6] + s * sb[7];
	  R = R1 + s2 * R2 + s4 * R3 + s6 * rb[6];
	  S = S1 + s2 * S2 + s4 * S3 + s6 * S4;
	}
      z = x;
      SET_LOW_WORD (z, 0);
      r = __ieee754_exp (-z * z - 0.5625) *
	  __ieee754_exp ((z - x) * (z + x) + R / S);
      if (hx > 0)
	{
	  double ret = math_narrow_eval (r / x);
	  if (ret == 0)
	    __set_errno (ERANGE);
	  return ret;
	}
      else
	return two - r / x;
    }
  else
    {
      if (hx > 0)
	{
	  __set_errno (ERANGE);
	  return tiny * tiny;
	}
      else
	return two - tiny;
    }
}
libm_alias_double (__erfc, erfc)
