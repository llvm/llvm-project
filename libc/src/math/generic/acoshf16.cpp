//===-- Half-precision acoshf16 function ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/acoshf16.h"
#include "src/__support/FPUtil/FEnvImpl.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/FPUtil/cast.h"
#include "src/__support/FPUtil/except_value_utils.h"
#include "src/__support/FPUtil/generic/sqrt.h"
#include "src/__support/FPUtil/multiply_add.h"
#include "src/__support/macros/optimization.h"
#include "src/math/generic/explogxf.h"

namespace LIBC_NAMESPACE_DECL {

static constexpr size_t N_EXCEPTS = 1;
static constexpr fputil::ExceptValues<float16, N_EXCEPTS> ACOSHF16_EXCEPTS{
    {// (input, RZ output, RU offset, RD offset, RN offset)
     {0x41B7, 0x3ED8, 0, 1, 0}}};

LLVM_LIBC_FUNCTION(float16, acoshf16, (float16 x)) {
  using FPBits = fputil::FPBits<float16>;
  FPBits xbits(x);
  uint16_t x_u = xbits.uintval();
  uint16_t x_abs = x_u & 0x7fff;

  // Check for NaN input first.
  if (LIBC_UNLIKELY(xbits.is_nan())) {
    if (xbits.is_signaling_nan()) {
      fputil::raise_except_if_required(FE_INVALID);
      return FPBits::quiet_nan().get_val();
    }
    return x;
  }

  // Check for infinite inputs.
  if (LIBC_UNLIKELY(xbits.is_inf())) {
    if (xbits.is_neg()) {
      fputil::set_errno_if_required(EDOM);
      fputil::raise_except_if_required(FE_INVALID);
      return FPBits::quiet_nan().get_val();
    }
    return x;
  }

  // Domain error for inputs less than 1.0.
  if (LIBC_UNLIKELY(x_abs < 0x3c00U)) {
    fputil::set_errno_if_required(EDOM);
    fputil::raise_except_if_required(FE_INVALID);
    return FPBits::quiet_nan().get_val();
  }

  // acosh(1.0) exactly equals 0.0
  if (LIBC_UNLIKELY(x_u == 0x3c00U))
    return float16(0.0f);

  float xf32 = x;

  // High precision for inputs very close to 1.0
  if (LIBC_UNLIKELY(xf32 < 1.25f)) {
    float delta = xf32 - 1.0f;
    float sqrt_2_delta = fputil::sqrt<float>(2.0 * delta);
    float x2 = delta;
    float pe = fputil::polyeval(x2, 0x1.0000000000000p+0f,
                                -0x1.55551a83a9472p-4f, 0x1.331601c4b8ecfp-6f,
                                -0x1.6890f49eb0acbp-8f, 0x1.8f3a617040a6ap-10f);
    float approx = sqrt_2_delta * pe;
    return fputil::cast<float16>(approx);
  }

  if (auto r = ACOSHF16_EXCEPTS.lookup(xbits.uintval());
      LIBC_UNLIKELY(r.has_value()))
    return r.value();

  // Standard computation for general case.
  float sqrt_term =
      fputil::sqrt<float>(fputil::multiply_add(xf32, xf32, -1.0f));
  float result = static_cast<float>(log_eval(xf32 + sqrt_term));

  return fputil::cast<float16>(result);
}

} // namespace LIBC_NAMESPACE_DECL
