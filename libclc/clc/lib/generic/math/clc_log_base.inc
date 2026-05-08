//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

/*
   Algorithm:

   Based on:
   Ping-Tak Peter Tang
   "Table-driven implementation of the logarithm function in IEEE
   floating-point arithmetic"
   ACM Transactions on Mathematical Software (TOMS)
   Volume 16, Issue 4 (December 1990)


   x very close to 1.0 is handled differently, for x everywhere else
   a brief explanation is given below

   x = (2^m)*A
   x = (2^m)*(G+g) with (1 <= G < 2) and (g <= 2^(-8))
   x = (2^m)*2*(G/2+g/2)
   x = (2^m)*2*(F+f) with (0.5 <= F < 1) and (f <= 2^(-9))

   Y = (2^(-1))*(2^(-m))*(2^m)*A
   Now, range of Y is: 0.5 <= Y < 1

   F = 0x80 + (first 7 mantissa bits) + (8th mantissa bit)
   Now, range of F is: 128 <= F <= 256
   F = F / 256
   Now, range of F is: 0.5 <= F <= 1

   f = -(Y-F), with (f <= 2^(-9))

   log(x) = m*log(2) + log(2) + log(F-f)
   log(x) = m*log(2) + log(2) + log(F) + log(1-(f/F))
   log(x) = m*log(2) + log(2*F) + log(1-r)

   r = (f/F), with (r <= 2^(-8))
   r = f*(1/F) with (1/F) precomputed to avoid division

   log(x) = m*log(2) + log(G) - poly

   log(G) is precomputed
   poly = (r + (r^2)/2 + (r^3)/3 + (r^4)/4) + (r^5)/5))

   log(2) and log(G) need to be maintained in extra precision
   to avoid losing precision in the calculations


   For x close to 1.0, we employ the following technique to
   ensure faster convergence.

   log(x) = log((1+s)/(1-s)) = 2*s + (2/3)*s^3 + (2/5)*s^5 + (2/7)*s^7
   x = ((1+s)/(1-s))
   x = 1 + r
   s = r/(2+r)

*/

#ifdef __CLC_SCALAR

#if __CLC_FPSIZE == 32
_CLC_OVERLOAD _CLC_DEF __CLC_FLOATN
#if defined(COMPILING_LOG2)
__clc_log2(__CLC_FLOATN x)
#elif defined(COMPILING_LOG10)
__clc_log10(__CLC_FLOATN x)
#else
__clc_log(__CLC_FLOATN x)
#endif
{

#if defined(COMPILING_LOG2)
  const __CLC_FLOATN LOG2E = 0x1.715476p+0f;      // 1.4426950408889634
  const __CLC_FLOATN LOG2E_HEAD = 0x1.700000p+0f; // 1.4375
  const __CLC_FLOATN LOG2E_TAIL = 0x1.547652p-8f; // 0.00519504072
#elif defined(COMPILING_LOG10)
  const __CLC_FLOATN LOG10E = 0x1.bcb7b2p-2f;        // 0.43429448190325182
  const __CLC_FLOATN LOG10E_HEAD = 0x1.bc0000p-2f;   // 0.43359375
  const __CLC_FLOATN LOG10E_TAIL = 0x1.6f62a4p-11f;  // 0.0007007319
  const __CLC_FLOATN LOG10_2_HEAD = 0x1.340000p-2f;  // 0.30078125
  const __CLC_FLOATN LOG10_2_TAIL = 0x1.04d426p-12f; // 0.000248745637
#else
  const __CLC_FLOATN LOG2_HEAD = 0x1.62e000p-1f;  // 0.693115234
  const __CLC_FLOATN LOG2_TAIL = 0x1.0bfbe8p-15f; // 0.0000319461833
#endif

  uint xi = __clc_as_uint(x);
  uint ax = xi & EXSIGNBIT_SP32;

  // Calculations for |x-1| < 2^-4
  __CLC_FLOATN r = x - 1.0f;
  int near1 = __clc_fabs(r) < 0x1.0p-4f;
  __CLC_FLOATN u2 = MATH_DIVIDE(r, 2.0f + r);
  __CLC_FLOATN corr = u2 * r;
  __CLC_FLOATN u = u2 + u2;
  __CLC_FLOATN v = u * u;
  __CLC_FLOATN znear1, z1, z2;

  // 2/(5 * 2^5), 2/(3 * 2^3)
  z2 = __clc_mad(u, __clc_mad(v, 0x1.99999ap-7f, 0x1.555556p-4f) * v, -corr);

#if defined(COMPILING_LOG2)
  z1 = __CLC_AS_FLOATN(__CLC_AS_INTN(r) & 0xffff0000);
  z2 = z2 + (r - z1);
  znear1 = __clc_mad(
      z1, LOG2E_HEAD,
      __clc_mad(z2, LOG2E_HEAD, __clc_mad(z1, LOG2E_TAIL, z2 * LOG2E_TAIL)));
#elif defined(COMPILING_LOG10)
  z1 = __CLC_AS_FLOATN(__clc_as_int(r) & 0xffff0000);
  z2 = z2 + (r - z1);
  znear1 = __clc_mad(
      z1, LOG10E_HEAD,
      __clc_mad(z2, LOG10E_HEAD, __clc_mad(z1, LOG10E_TAIL, z2 * LOG10E_TAIL)));
#else
  znear1 = z2 + r;
#endif

  // Calculations for x not near 1
  int m = (int)(xi >> EXPSHIFTBITS_SP32) - EXPBIAS_SP32;

  // Normalize subnormal
  uint xis = __CLC_AS_UINTN(__CLC_AS_FLOATN(xi | 0x3f800000) - 1.0f);
  int ms = (int)(xis >> EXPSHIFTBITS_SP32) - 253;
  int c = m == -127;
  m = c ? ms : m;
  uint xin = c ? xis : xi;

  __CLC_FLOATN mf = (__CLC_FLOATN)m;
  uint indx = (xin & 0x007f0000) + ((xin & 0x00008000) << 1);

  // F - Y
  __CLC_FLOATN f = __CLC_AS_FLOATN(0x3f000000 | indx) -
                   __CLC_AS_FLOATN(0x3f000000 | (xin & MANTBITS_SP32));

  indx = indx >> 16;
  r = f * __CLC_USE_TABLE(log_inv_tbl, indx);

  // 1/3,  1/2
  __CLC_FLOATN poly = __clc_mad(__clc_mad(r, 0x1.555556p-2f, 0.5f), r * r, r);

#if defined(COMPILING_LOG2)
  float2 tv = __CLC_USE_TABLE(log2_tbl, indx);
  z1 = tv.s0 + mf;
  z2 = __clc_mad(poly, -LOG2E, tv.s1);
#elif defined(COMPILING_LOG10)
  float2 tv = __CLC_USE_TABLE(log10_tbl, indx);
  z1 = __clc_mad(mf, LOG10_2_HEAD, tv.s0);
  z2 = __clc_mad(poly, -LOG10E, mf * LOG10_2_TAIL) + tv.s1;
#else
  float2 tv = __CLC_USE_TABLE(log_tbl, indx);
  z1 = __clc_mad(mf, LOG2_HEAD, tv.s0);
  z2 = __clc_mad(mf, LOG2_TAIL, -poly) + tv.s1;
#endif

  __CLC_FLOATN z = z1 + z2;
  z = near1 ? znear1 : z;

  // Corner cases
  z = ax >= PINFBITPATT_SP32 ? x : z;
  z = xi != ax ? __CLC_AS_FLOATN(QNANBITPATT_SP32) : z;
  z = ax == 0 ? __CLC_AS_FLOATN(NINFBITPATT_SP32) : z;

  return z;
}

#elif __CLC_FPSIZE == 64

_CLC_OVERLOAD _CLC_DEF __CLC_DOUBLEN
#if defined(COMPILING_LOG2)
__clc_log2(__CLC_DOUBLEN a)
#elif defined(COMPILING_LOG10)
__clc_log10(__CLC_DOUBLEN a)
#else
__clc_log(__CLC_DOUBLEN a)
#endif
{
  __CLC_INTN a_exp;
  __CLC_DOUBLEN m = __clc_frexp(a, &a_exp);
  __CLC_INTN b = m < (2.0 / 3.0);
  m = __clc_ldexp(m, b);
  __CLC_INTN e = a_exp - b;

  __CLC_EP_PAIR x = __clc_ep_div(m - 1.0, __clc_ep_fast_add(1.0, m));
  __CLC_DOUBLEN s = x.hi * x.hi;
  __CLC_DOUBLEN p = __clc_mad(s, __clc_mad(s, __clc_mad(s,
             __clc_mad(s, __clc_mad(s, __clc_mad(s, 0x1.3ab76bf559e2bp-3, 0x1.385386b47b09ap-3),
               0x1.7474dd7f4df2ep-3), 0x1.c71c016291751p-3),
               0x1.249249b27acf1p-2), 0x1.99999998ef7b6p-2), 0x1.5555555555780p-1);
  __CLC_EP_PAIR r =
      __clc_ep_fast_add(__clc_ep_ldexp(x, 1), s * x.hi * p);

#if defined COMPILING_LOG2
  r = __clc_ep_add(
      __CLC_CONVERT_DOUBLEN(e),
      __clc_ep_mul(
          __clc_ep_make_pair(0x1.71547652b82fep+0, 0x1.777d0ffda0d24p-56), r));
#elif defined COMPILING_LOG10
  r = __clc_ep_add(
      __clc_ep_mul(
          __clc_ep_make_pair(0x1.34413509f79ffp-2, -0x1.9dc1da994fd21p-59),
          __CLC_CONVERT_DOUBLEN(e)),
      __clc_ep_mul(
          __clc_ep_make_pair(0x1.bcb7b1526e50ep-2, 0x1.95355baaafad3p-57), r));
#else
  r = __clc_ep_add(__clc_ep_mul(__clc_ep_make_pair(0x1.62e42fefa39efp-1,
                                                   0x1.abc9e3b39803fp-56),
                                __CLC_CONVERT_DOUBLEN(e)),
                   r);
#endif

  __CLC_DOUBLEN ret = r.hi;

  ret = __clc_isinf(a) ? a : ret;
  ret = a < 0.0 ? DBL_NAN : ret;
  ret = a == 0.0 ? -INFINITY : ret;

  return ret;
}

#elif __CLC_FPSIZE == 16

_CLC_OVERLOAD _CLC_DEF __CLC_HALFN
#if defined(COMPILING_LOG2)
__clc_log2(__CLC_HALFN x) {
  return __CLC_CONVERT_HALFN(__clc_log2(__CLC_CONVERT_FLOATN(x)));
}
#elif defined(COMPILING_LOG10)
__clc_log10(__CLC_HALFN x) {
  return __CLC_CONVERT_HALFN(__clc_log10(__CLC_CONVERT_FLOATN(x)));
}
#else
__clc_log(__CLC_HALFN x) {
  return __CLC_CONVERT_HALFN(__clc_log(__CLC_CONVERT_FLOATN(x));
}
#endif

#endif // __CLC_FPSIZE
#endif // __CLC_SCALAR
