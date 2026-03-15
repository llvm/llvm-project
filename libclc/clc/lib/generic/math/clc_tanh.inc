//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if __CLC_FPSIZE == 32

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __clc_tanh(__CLC_GENTYPE x) {
  // The definition of tanh(x) is sinh(x)/cosh(x), which is also equivalent
  // to the following three formulae:
  // 1.  (exp(x) - exp(-x))/(exp(x) + exp(-x))
  // 2.  (1 - (2/(exp(2*x) + 1 )))
  // 3.  (exp(2*x) - 1)/(exp(2*x) + 1)
  // but computationally, some formulae are better on some ranges.

  const __CLC_GENTYPE large_threshold = 0x1.0a2b24p+3f;

  __CLC_UINTN ux = __CLC_AS_UINTN(x);
  __CLC_UINTN aux = ux & EXSIGNBIT_SP32;
  __CLC_UINTN xs = ux ^ aux;

  __CLC_GENTYPE y = __CLC_AS_GENTYPE(aux);
  __CLC_GENTYPE y2 = y * y;

  __CLC_GENTYPE a1 = __clc_mad(
      y2, __clc_mad(y2, 0.4891631088530669873e-4F, -0.14628356048797849e-2F),
      -0.28192806108402678e0F);
  __CLC_GENTYPE b1 =
      __clc_mad(y2, 0.3427017942262751343e0F, 0.845784192581041099e0F);

  __CLC_GENTYPE a2 = __clc_mad(
      y2, __clc_mad(y2, 0.3827534993599483396e-4F, -0.12325644183611929e-2F),
      -0.24069858695196524e0F);
  __CLC_GENTYPE b2 =
      __clc_mad(y2, 0.292529068698052819e0F, 0.72209738473684982e0F);

  __CLC_INTN c = y < 0.9f;
  __CLC_GENTYPE a = c ? a1 : a2;
  __CLC_GENTYPE b = c ? b1 : b2;
  __CLC_GENTYPE zlo = __clc_mad(MATH_DIVIDE(a, b), y * y2, y);

  __CLC_GENTYPE p = __clc_exp(2.0f * y) + 1.0f;
  __CLC_GENTYPE zhi = 1.0F - MATH_DIVIDE(2.0F, p);

  __CLC_GENTYPE z = y <= 1.0f ? zlo : zhi;
  z = __CLC_AS_GENTYPE(xs | __CLC_AS_UINTN(z));

  // Edge cases
  __CLC_GENTYPE sone = __CLC_AS_GENTYPE(0x3f800000U | xs);
  z = y > large_threshold ? sone : z;
  z = aux < 0x39000000 || aux > 0x7f800000 ? x : z;

  return z;
}

#elif __CLC_FPSIZE == 64

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __clc_tanh(__CLC_GENTYPE x) {
  // The definition of tanh(x) is sinh(x)/cosh(x), which is also equivalent
  // to the following three formulae:
  // 1.  (exp(x) - exp(-x))/(exp(x) + exp(-x))
  // 2.  (1 - (2/(exp(2*x) + 1 )))
  // 3.  (exp(2*x) - 1)/(exp(2*x) + 1)
  // but computationally, some formulae are better on some ranges.

  // The point at which e^-x is insignificant compared to e^x = ln(2^27)
  const __CLC_GENTYPE large_threshold = 0x1.2b708872320e2p+4;

  __CLC_ULONGN ux = __CLC_AS_ULONGN(x);
  __CLC_ULONGN ax = ux & ~SIGNBIT_DP64;
  __CLC_ULONGN sx = ux ^ ax;
  __CLC_GENTYPE y = __CLC_AS_GENTYPE(ax);
  __CLC_GENTYPE y2 = y * y;

  // y < 0.9
  __CLC_GENTYPE znl =
      __clc_fma(y2,
                __clc_fma(y2,
                          __clc_fma(y2, -0.142077926378834722618091e-7,
                                    -0.200047621071909498730453e-3),
                          -0.176016349003044679402273e-1),
                -0.274030424656179760118928e0);

  __CLC_GENTYPE zdl =
      __clc_fma(y2,
                __clc_fma(y2,
                          __clc_fma(y2, 0.2091140262529164482568557e-3,
                                    0.201562166026937652780575e-1),
                          0.381641414288328849317962e0),
                0.822091273968539282568011e0);

  // 0.9 <= y <= 1
  __CLC_GENTYPE znm =
      __clc_fma(y2,
                __clc_fma(y2,
                          __clc_fma(y2, -0.115475878996143396378318e-7,
                                    -0.165597043903549960486816e-3),
                          -0.146173047288731678404066e-1),
                -0.227793870659088295252442e0);

  __CLC_GENTYPE zdm =
      __clc_fma(y2,
                __clc_fma(y2,
                          __clc_fma(y2, 0.173076050126225961768710e-3,
                                    0.167358775461896562588695e-1),
                          0.317204558977294374244770e0),
                0.683381611977295894959554e0);

  __CLC_LONGN c = y < 0.9;
  __CLC_GENTYPE zn = c ? znl : znm;
  __CLC_GENTYPE zd = c ? zdl : zdm;
  __CLC_GENTYPE z = y + y * y2 * MATH_DIVIDE(zn, zd);

  // y > 1
  __CLC_GENTYPE p = __clc_exp(2.0 * y) + 1.0;
  __CLC_GENTYPE zg = 1.0 - 2.0 / p;

  z = y > 1.0 ? zg : z;

  // Other cases
  z = y < 0x1.0p-28 || ax > PINFBITPATT_DP64 ? x : z;

  z = y > large_threshold ? 1.0 : z;

  return __CLC_AS_GENTYPE(sx | __CLC_AS_ULONGN(z));
}

#elif __CLC_FPSIZE == 16

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __clc_tanh(__CLC_GENTYPE x) {
  return __CLC_CONVERT_GENTYPE(__clc_tanh(__CLC_CONVERT_FLOATN(x)));
}

#endif
