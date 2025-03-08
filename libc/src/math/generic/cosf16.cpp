//===-- Half-precision cos(x) function ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/cosf16.h"
#include "hdr/errno_macros.h"
#include "hdr/fenv_macros.h"
#include "sincosf16_utils.h"
#include "src/__support/FPUtil/FEnvImpl.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/FPUtil/cast.h"
#include "src/__support/FPUtil/except_value_utils.h"
#include "src/__support/FPUtil/multiply_add.h"
#include "src/__support/macros/optimization.h"

namespace LIBC_NAMESPACE_DECL {

constexpr size_t N_EXCEPTS = 4;

constexpr fputil::ExceptValues<float16, N_EXCEPTS> COSF16_EXCEPTS{{
    // (input, RZ output, RU offset, RD offset, RN offset)
    {0x2b7c, 0x3bfc, 1, 0, 1},
    {0x4ac1, 0x38b5, 1, 0, 0},
    {0x5c49, 0xb8c6, 0, 1, 0},
    {0x7acc, 0xa474, 0, 1, 0},
}};

LLVM_LIBC_FUNCTION(float16, cosf16, (float16 x)) {
  using FPBits = fputil::FPBits<float16>;
  FPBits xbits(x);

  uint16_t x_u = xbits.uintval();
  uint16_t x_abs = x_u & 0x7fff;
  float xf = x;

  // Range reduction:
  // For |x| > pi/32, we perform range reduction as follows:
  // Find k and y such that:
  //   x = (k + y) * pi/32
  //   k is an integer, |y| < 0.5
  //
  // This is done by performing:
  //   k = round(x * 32/pi)
  //   y = x * 32/pi - k
  //
  // Once k and y are computed, we then deduce the answer by the cosine of sum
  // formula:
  //   cos(x) = cos((k + y) * pi/32)
  //          = cos(k * pi/32) * cos(y * pi/32) -
  //            sin(k * pi/32) * sin(y * pi/32)

  // Handle exceptional values
  if (auto r = COSF16_EXCEPTS.lookup(x_abs); LIBC_UNLIKELY(r.has_value()))
    return r.value();

  // cos(+/-0) = 1
  if (LIBC_UNLIKELY(x_abs == 0U))
    return fputil::cast<float16>(1.0f);

  // cos(+/-inf) = NaN, and cos(NaN) = NaN
  if (xbits.is_inf_or_nan()) {
    if (xbits.is_inf()) {
      fputil::set_errno_if_required(EDOM);
      fputil::raise_except_if_required(FE_INVALID);
    }

    return x + FPBits::quiet_nan().get_val();
  }

  float sin_k, cos_k, sin_y, cosm1_y;
  sincosf16_eval(xf, sin_k, cos_k, sin_y, cosm1_y);
  // Since, cosm1_y = cos_y - 1, therefore:
  //   cos(x) = cos_k * cos_y - sin_k * sin_y
  //          = cos_k * (cos_y - 1 + 1) - sin_k * sin_y
  //          = cos_k * cosm1_y - sin_k * sin_y + cos_k
  return fputil::cast<float16>(fputil::multiply_add(
      cos_k, cosm1_y, fputil::multiply_add(-sin_k, sin_y, cos_k)));
}

} // namespace LIBC_NAMESPACE_DECL
