//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/math/clc_mad.h>

// Evaluate single precisions in and cos of value in interval [-pi/4, pi/4]
_CLC_INLINE float2 __libclc__sincosf_piby4(float x) {
  // Taylor series for sin(x) is x - x^3/3! + x^5/5! - x^7/7! ...
  // = x * (1 - x^2/3! + x^4/5! - x^6/7! ...
  // = x * f(w)
  // where w = x*x and f(w) = (1 - w/3! + w^2/5! - w^3/7! ...
  // We use a minimax approximation of (f(w) - 1) / w
  // because this produces an expansion in even powers of x.

  // Taylor series for cos(x) is 1 - x^2/2! + x^4/4! - x^6/6! ...
  // = f(w)
  // where w = x*x and f(w) = (1 - w/2! + w^2/4! - w^3/6! ...
  // We use a minimax approximation of (f(w) - 1 + w/2) / (w*w)
  // because this produces an expansion in even powers of x.

  const float sc1 = -0.166666666638608441788607926e0F;
  const float sc2 = 0.833333187633086262120839299e-2F;
  const float sc3 = -0.198400874359527693921333720e-3F;
  const float sc4 = 0.272500015145584081596826911e-5F;

  const float cc1 = 0.41666666664325175238031e-1F;
  const float cc2 = -0.13888887673175665567647e-2F;
  const float cc3 = 0.24800600878112441958053e-4F;
  const float cc4 = -0.27301013343179832472841e-6F;

  float x2 = x * x;

  float2 ret;
  ret.x = __clc_mad(
      x * x2, __clc_mad(x2, __clc_mad(x2, __clc_mad(x2, sc4, sc3), sc2), sc1),
      x);
  ret.y = __clc_mad(
      x2 * x2, __clc_mad(x2, __clc_mad(x2, __clc_mad(x2, cc4, cc3), cc2), cc1),
      __clc_mad(x2, -0.5f, 1.0f));
  return ret;
}
