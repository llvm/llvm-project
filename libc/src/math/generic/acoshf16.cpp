//===-- Half-precision acoshf16 function -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/acoshf16.h"
#include "hdr/errno_macros.h"
#include "hdr/fenv_macros.h"
#include "src/__support/FPUtil/FEnvImpl.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/FPUtil/cast.h"
#include "src/__support/FPUtil/multiply_add.h"
#include "src/__support/FPUtil/sqrt.h"
#include "src/__support/macros/optimization.h"
#include "src/math/generic/common_constants.h"
#include "src/math/generic/explogxf.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(float16, acoshf16, (float16 x)) {
  using FPBits = fputil::FPBits<float16>;
  FPBits xbits(x);

  uint16_t x_u = xbits.uintval();
  uint16_t x_abs = x_u & 0x7fff;

  // Check for NaN input first.
  if (LIBC_UNLIKELY(xbits.is_nan())) {
    if (xbits.is_signaling_nan()) {
      fputil::raise_except(FE_INVALID);
      return FPBits::quiet_nan().get_val();
    }
    return x;
  }

  // Check for infinite inputs.
  if (LIBC_UNLIKELY(xbits.is_inf())) {
    if (xbits.is_neg()) {
      fputil::set_errno_if_required(EDOM);
      fputil::raise_except(FE_INVALID);
      return FPBits::quiet_nan().get_val();
    }
    return x;
  }

  // Domain error for inputs less than 1.0.
  if (LIBC_UNLIKELY(x_abs < 0x3c00)) {
    fputil::set_errno_if_required(EDOM);
    fputil::raise_except(FE_INVALID);
    return FPBits::quiet_nan().get_val();
  }

  // acosh(1.0) exactly equals 0.0
  if (LIBC_UNLIKELY(xbits.uintval() == 0x3c00U))
    return float16(0.0f);

  float xf32 = static_cast<float>(x);

  // High precision for inputs very close to 1.0
  if (LIBC_UNLIKELY(xf32 < 1.25f)) {
    float delta = xf32 - 1.0f;
    float sqrt_2_delta = fputil::sqrt<float>(2.0f * delta);
    double correction = (double)delta / 12.0 - (3.0 * (double)delta * delta) / 160.0;
    double precise_result = (double)sqrt_2_delta * (1.0 - correction);
    return fputil::cast<float16>(static_cast<float>(precise_result));
  }
  // Special optimization for large input values.
  if (LIBC_UNLIKELY(xf32 >= 32.0f)) {
    float result = static_cast<float>(log_eval(2.0f * xf32));
    return fputil::cast<float16>(result);
  }

  // Standard computation for general case.
  float sqrt_term = fputil::sqrt<float>(fputil::multiply_add(xf32, xf32, -1.0f));
  float result = static_cast<float>(log_eval(xf32 + sqrt_term));

  return fputil::cast<float16>(result);
}

} // namespace LIBC_NAMESPACE_DECL