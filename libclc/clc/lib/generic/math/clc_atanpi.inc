//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if __CLC_FPSIZE == 32

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __clc_atanpi(__CLC_GENTYPE x) {
  const __CLC_GENTYPE pi = __CLC_FP_LIT(3.1415926535897932);

  __CLC_UINTN ux = __CLC_AS_UINTN(x);
  __CLC_UINTN aux = ux & EXSIGNBIT_SP32;
  __CLC_UINTN sx = ux ^ aux;

  __CLC_GENTYPE xbypi = MATH_DIVIDE(x, pi);
  __CLC_GENTYPE shalf =
      __CLC_AS_GENTYPE(sx | __CLC_AS_UINTN(__CLC_FP_LIT(0.5)));

  __CLC_GENTYPE v = __CLC_AS_GENTYPE(aux);

  // Return for NaN
  __CLC_GENTYPE ret = x;

  // 2^26 <= |x| <= Inf => atan(x) is close to piby2
  ret = aux <= PINFBITPATT_SP32 ? shalf : ret;

  // Reduce arguments 2^-19 <= |x| < 2^26

  // 39/16 <= x < 2^26
  x = -MATH_RECIP(v);
  __CLC_GENTYPE c = 1.57079632679489655800f; // atan(infinity)

  // 19/16 <= x < 39/16
  __CLC_INTN l = aux < 0x401c0000;
  __CLC_GENTYPE xx = MATH_DIVIDE(v - 1.5f, __clc_mad(v, 1.5f, 1.0f));
  x = l ? xx : x;
  c = l ? 9.82793723247329054082e-01f : c; // atan(1.5)

  // 11/16 <= x < 19/16
  l = aux < 0x3f980000U;
  xx = MATH_DIVIDE(v - 1.0f, 1.0f + v);
  x = l ? xx : x;
  c = l ? 7.85398163397448278999e-01f : c; // atan(1)

  // 7/16 <= x < 11/16
  l = aux < 0x3f300000;
  xx = MATH_DIVIDE(__clc_mad(v, 2.0f, -1.0f), 2.0f + v);
  x = l ? xx : x;
  c = l ? 4.63647609000806093515e-01f : c; // atan(0.5)

  // 2^-19 <= x < 7/16
  l = aux < 0x3ee00000;
  x = l ? v : x;
  c = l ? 0.0f : c;

  // Core approximation: Remez(2,2) on [-7/16,7/16]

  __CLC_GENTYPE s = x * x;
  __CLC_GENTYPE a = __clc_mad(s,
                              __clc_mad(s, 0.470677934286149214138357545549e-2f,
                                        0.192324546402108583211697690500f),
                              0.296528598819239217902158651186f);

  __CLC_GENTYPE b = __clc_mad(s,
                              __clc_mad(s, 0.299309699959659728404442796915f,
                                        0.111072499995399550138837673349e1f),
                              0.889585796862432286486651434570f);

  __CLC_GENTYPE q = x * s * MATH_DIVIDE(a, b);

  __CLC_GENTYPE z = c - (q - x);
  z = MATH_DIVIDE(z, pi);
  __CLC_GENTYPE zs = __CLC_AS_GENTYPE(sx | __CLC_AS_UINTN(z));

  ret = aux < 0x4c800000 ? zs : ret;

  // |x| < 2^-19
  ret = aux < 0x36000000 ? xbypi : ret;
  return ret;
}

#elif __CLC_FPSIZE == 64

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __clc_atanpi(__CLC_GENTYPE x) {
  const __CLC_GENTYPE pi = __CLC_FP_LIT(0x1.921fb54442d18p+1);

  __CLC_GENTYPE v = __clc_fabs(x);

  // 2^56 > v > 39/16
  __CLC_GENTYPE a = -1.0;
  __CLC_GENTYPE b = v;
  // (chi + clo) = arctan(infinity)
  __CLC_GENTYPE chi = 1.57079632679489655800e+00;
  __CLC_GENTYPE clo = 6.12323399573676480327e-17;

  __CLC_GENTYPE ta = v - 1.5;
  __CLC_GENTYPE tb = 1.0 + 1.5 * v;
  __CLC_LONGN l = v <= 0x1.38p+1; // 39/16 > v > 19/16
  a = l ? ta : a;
  b = l ? tb : b;
  // (chi + clo) = arctan(1.5)
  chi = l ? 9.82793723247329054082e-01 : chi;
  clo = l ? 1.39033110312309953701e-17 : clo;

  ta = v - 1.0;
  tb = 1.0 + v;
  l = v <= 0x1.3p+0; // 19/16 > v > 11/16
  a = l ? ta : a;
  b = l ? tb : b;
  // (chi + clo) = arctan(1.)
  chi = l ? 7.85398163397448278999e-01 : chi;
  clo = l ? 3.06161699786838240164e-17 : clo;

  ta = 2.0 * v - 1.0;
  tb = 2.0 + v;
  l = v <= 0x1.6p-1; // 11/16 > v > 7/16
  a = l ? ta : a;
  b = l ? tb : b;
  // (chi + clo) = arctan(0.5)
  chi = l ? 4.63647609000806093515e-01 : chi;
  clo = l ? 2.26987774529616809294e-17 : clo;

  l = v <= 0x1.cp-2; // v < 7/16
  a = l ? v : a;
  b = l ? 1.0 : b;
  ;
  chi = l ? 0.0 : chi;
  clo = l ? 0.0 : clo;

  // Core approximation: Remez(4,4) on [-7/16,7/16]
  __CLC_GENTYPE r = a / b;
  __CLC_GENTYPE s = r * r;
  __CLC_GENTYPE qn =
      __clc_fma(s,
                __clc_fma(s,
                          __clc_fma(s,
                                    __clc_fma(s, 0.142316903342317766e-3,
                                              0.304455919504853031e-1),
                                    0.220638780716667420e0),
                          0.447677206805497472e0),
                0.268297920532545909e0);

  __CLC_GENTYPE qd =
      __clc_fma(s,
                __clc_fma(s,
                          __clc_fma(s,
                                    __clc_fma(s, 0.389525873944742195e-1,
                                              0.424602594203847109e0),
                                    0.141254259931958921e1),
                          0.182596787737507063e1),
                0.804893761597637733e0);

  __CLC_GENTYPE q = r * s * qn / qd;
  r = (chi - ((q - clo) - r)) / pi;
  __CLC_GENTYPE vp = v / pi;

  __CLC_GENTYPE z = __clc_isnan(x) ? x : 0.5;
  z = v <= 0x1.0p+56 ? r : z;
  z = v < 0x1.0p-26 ? vp : z;
  return x == v ? z : -z;
}

#elif __CLC_FPSIZE == 16

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __clc_atanpi(__CLC_GENTYPE x) {
  return __CLC_CONVERT_GENTYPE(__clc_atanpi(__CLC_CONVERT_FLOATN(x)));
}

#endif
